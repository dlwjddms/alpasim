# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Turn one VAVAM prediction into the trajectories the Drive RPC returns.

The model predicts an ego-relative path.  When inference runs we convert that
path into the rollout's local (inertial) frame once, anchored at the current
ego pose, and cache it.

Between inferences we hand the controller the *same* cached path, just shorter:
each Drive call samples the cached plan on a grid fixed to the plan's own clock
and emits only the samples at or after "now".  The points never move until the
next inference -- the trajectory only shrinks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from alpasim_grpc.v0.common_pb2 import Pose, PoseAtTime, Quat, Trajectory, Vec3


@dataclass(frozen=True)
class CachedPlan:
    """One VAVAM prediction expressed in the rollout's local frame.

    ``times_s`` are seconds relative to ``created_time_us``; ``positions_xy``
    and ``yaws`` are absolute local-frame poses along the predicted path.
    """

    created_time_us: int
    times_s: np.ndarray
    positions_xy: np.ndarray
    yaws: np.ndarray


def yaw_from_quat(quat: Quat) -> float:
    siny_cosp = 2.0 * (quat.w * quat.z + quat.x * quat.y)
    cosy_cosp = 1.0 - 2.0 * (quat.y * quat.y + quat.z * quat.z)
    return math.atan2(siny_cosp, cosy_cosp)


def quat_from_yaw(yaw: float) -> Quat:
    half = 0.5 * yaw
    return Quat(w=float(math.cos(half)), x=0.0, y=0.0, z=float(math.sin(half)))


def rig_offsets_to_local_positions(
    anchor_pose: PoseAtTime,
    offsets_xy: np.ndarray,
) -> np.ndarray:
    """Project ego-relative (rig-frame) offsets into the rollout local frame."""
    offsets = np.asarray(offsets_xy, dtype=np.float64).reshape(-1, 2)
    yaw = yaw_from_quat(anchor_pose.pose.quat)
    c = math.cos(yaw)
    s = math.sin(yaw)
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    origin = np.array(
        [anchor_pose.pose.vec.x, anchor_pose.pose.vec.y], dtype=np.float64
    )
    return offsets @ rot.T + origin


def _wrap_to_pi(angle: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def stabilize_headings(
    *,
    raw_headings: np.ndarray,
    anchor_pose: PoseAtTime,
    previous_plan: "CachedPlan | None",
    new_created_time_us: int,
    ego_yaw_rate: float,
    source_frequency_hz: float,
    blend_steps: int = 2,
    yaw_rate_clamp_mult: float = 3.0,
    yaw_rate_floor: float = 0.7,
) -> np.ndarray:
    """Damp cross-inference heading discontinuities in raw model headings.

    ``raw_headings`` are rig-relative (added to the new anchor's yaw later in
    ``make_cached_plan``), one entry per future waypoint at
    ``1/source_frequency_hz`` spacing.  Two independent mechanisms, applied
    in sequence:

    1. Continuity blend: the outgoing ``previous_plan`` (if any) already
       committed to a heading trend; extrapolating that trend forward and
       blending it into the first ``blend_steps`` of the new headings avoids
       a hard jump right at the inference seam (every 500ms on the nuPlan
       track). Weight decays linearly from full at step 0 to 0 by
       ``blend_steps``.
    2. Yaw-rate plausibility clamp: using the ego's own just-measured yaw
       rate (``angular_velocity.z``, free signal, otherwise unused), clip
       any implied per-step heading delta that implausibly exceeds
       ``max(yaw_rate_clamp_mult * |ego_yaw_rate|, yaw_rate_floor)``. This is
       a loose defensive bound -- it should essentially never fire on
       ordinary cornering, only on outlier/hallucinated heading jumps.

    Returns headings in the same rig-relative frame as ``raw_headings``.
    """
    headings = np.asarray(raw_headings, dtype=np.float64).reshape(-1).copy()
    if headings.size == 0:
        return headings

    step_s = 1.0 / source_frequency_hz

    # --- 1. Continuity blend ---
    if (
        blend_steps > 0
        and previous_plan is not None
        and len(previous_plan.yaws) >= 2
    ):
        prev_d_yaw = float(
            _wrap_to_pi(previous_plan.yaws[-1] - previous_plan.yaws[-2])
        ) / max(1e-6, float(previous_plan.times_s[-1] - previous_plan.times_s[-2]))
        prev_last_abs_yaw = float(previous_plan.yaws[-1])
        prev_last_t_us = previous_plan.created_time_us + int(
            round(float(previous_plan.times_s[-1]) * 1_000_000)
        )
        anchor_yaw = yaw_from_quat(anchor_pose.pose.quat)

        n_blend = min(blend_steps, headings.size)
        for i in range(n_blend):
            future_us = new_created_time_us + int(round((i + 1) * step_s * 1_000_000))
            dt_s = (future_us - prev_last_t_us) / 1_000_000.0
            extrapolated_abs_yaw = prev_last_abs_yaw + prev_d_yaw * dt_s
            extrapolated_rig_heading = float(
                _wrap_to_pi(extrapolated_abs_yaw - anchor_yaw)
            )
            weight = max(0.0, 1.0 - i / blend_steps)
            delta = float(_wrap_to_pi(extrapolated_rig_heading - headings[i]))
            headings[i] = headings[i] + weight * delta

    # --- 2. Yaw-rate plausibility clamp ---
    max_rate = max(yaw_rate_clamp_mult * abs(ego_yaw_rate), yaw_rate_floor)
    prev_heading = 0.0  # rig-relative heading at t=0 is ego-forward by definition
    for i in range(headings.size):
        delta = float(_wrap_to_pi(headings[i] - prev_heading))
        max_delta = max_rate * step_s
        if abs(delta) > max_delta:
            delta = math.copysign(max_delta, delta)
            headings[i] = prev_heading + delta
        prev_heading = headings[i]

    return headings


def local_positions_to_rig_offsets(
    anchor_pose: PoseAtTime,
    local_positions_xy: np.ndarray,
) -> np.ndarray:
    """Inverse of ``rig_offsets_to_local_positions``: project local-frame
    positions back into ego-relative (rig-frame) offsets at ``anchor_pose``.
    """
    positions = np.asarray(local_positions_xy, dtype=np.float64).reshape(-1, 2)
    yaw = yaw_from_quat(anchor_pose.pose.quat)
    c = math.cos(yaw)
    s = math.sin(yaw)
    rot_inv = np.array([[c, s], [-s, c]], dtype=np.float64)
    origin = np.array(
        [anchor_pose.pose.vec.x, anchor_pose.pose.vec.y], dtype=np.float64
    )
    return (positions - origin) @ rot_inv.T


def apply_temporal_consistency(
    *,
    raw_trajectory_xy: np.ndarray,
    raw_headings: np.ndarray,
    anchor_pose: PoseAtTime,
    previous_plan: "CachedPlan | None",
    new_created_time_us: int,
    source_frequency_hz: float,
    blend_steps: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Candidate B: strong cross-inference temporal consistency.

    Unlike ``stabilize_headings`` (heading only), this blends both the
    predicted *position* and heading of the first ``blend_steps`` future
    waypoints toward a straight-line extrapolation of the outgoing plan's
    own last-observed velocity/yaw-rate trend. Targets position/speed-profile
    discontinuities between consecutive inferences directly -- the gap
    identified after Stage 1 (heading-only) showed no effect: plan_deviation
    is a position metric, and heading-only smoothing never touched xy.

    Both inputs/outputs are rig-relative offsets (as returned by the model,
    before ``make_cached_plan`` adds the anchor pose).
    """
    xy = np.asarray(raw_trajectory_xy, dtype=np.float64).reshape(-1, 2).copy()
    headings = np.asarray(raw_headings, dtype=np.float64).reshape(-1).copy()
    if xy.shape[0] == 0:
        return xy, headings
    if (
        blend_steps <= 0
        or previous_plan is None
        or len(previous_plan.positions_xy) < 2
    ):
        return xy, headings

    step_s = 1.0 / source_frequency_hz
    dt_prev = float(previous_plan.times_s[-1] - previous_plan.times_s[-2])
    if dt_prev <= 1e-6:
        return xy, headings

    v_trend = (previous_plan.positions_xy[-1] - previous_plan.positions_xy[-2]) / dt_prev
    prev_last_pos = previous_plan.positions_xy[-1]
    prev_last_t_us = previous_plan.created_time_us + int(
        round(float(previous_plan.times_s[-1]) * 1_000_000)
    )
    prev_d_yaw = float(
        _wrap_to_pi(previous_plan.yaws[-1] - previous_plan.yaws[-2])
    ) / dt_prev
    prev_last_yaw = float(previous_plan.yaws[-1])
    anchor_yaw = yaw_from_quat(anchor_pose.pose.quat)

    n_blend = min(blend_steps, xy.shape[0])
    for i in range(n_blend):
        future_us = new_created_time_us + int(round((i + 1) * step_s * 1_000_000))
        dt_s = (future_us - prev_last_t_us) / 1_000_000.0
        extrapolated_local_pos = prev_last_pos + v_trend * dt_s
        extrapolated_local_yaw = prev_last_yaw + prev_d_yaw * dt_s

        extrapolated_rig_xy = local_positions_to_rig_offsets(
            anchor_pose, extrapolated_local_pos[None, :]
        )[0]
        extrapolated_rig_heading = float(_wrap_to_pi(extrapolated_local_yaw - anchor_yaw))

        weight = max(0.0, 1.0 - i / blend_steps)
        xy[i] = xy[i] + weight * (extrapolated_rig_xy - xy[i])
        delta_h = float(_wrap_to_pi(extrapolated_rig_heading - headings[i]))
        headings[i] = headings[i] + weight * delta_h

    return xy, headings


def _resample_polyline_by_arclength(
    points: np.ndarray, s: np.ndarray, s_query: np.ndarray
) -> np.ndarray:
    """Linearly resample a 2D polyline at new arc-length coordinates.

    Query points beyond the polyline's own extent are extrapolated along the
    final segment's direction (not clamped to the last point) -- capping
    deceleration can imply covering *more* ground than the raw path's total
    length before coming to rest, and freezing at the last raw waypoint
    would silently reintroduce the abrupt stop this function exists to
    avoid.
    """
    xs = np.interp(s_query, s, points[:, 0])
    ys = np.interp(s_query, s, points[:, 1])

    s_max = float(s[-1])
    beyond = s_query > s_max
    if np.any(beyond):
        # Direction of the final non-degenerate segment.
        direction = np.array([1.0, 0.0])
        for k in range(len(points) - 1, 0, -1):
            seg = points[k] - points[k - 1]
            seg_len = float(np.hypot(seg[0], seg[1]))
            if seg_len > 1e-9:
                direction = seg / seg_len
                break
        extra = (s_query[beyond] - s_max)
        xs[beyond] = points[-1, 0] + direction[0] * extra
        ys[beyond] = points[-1, 1] + direction[1] * extra

    return np.column_stack([xs, ys])


def limit_braking_and_jerk(
    trajectory_xy: np.ndarray,
    time_step: float,
    *,
    max_decel_mps2: float = 3.0,
    max_jerk_mps3: float = 6.0,
) -> np.ndarray:
    """Candidate C: moderate brake/jerk limiter.

    A standalone, closed-form (non-optimization) re-timing pass: caps the
    implied longitudinal deceleration and jerk of the raw trajectory's speed
    profile, then re-samples the *same path* (shape unchanged) at the
    resulting, smoother arc-length schedule. Cheaper and more surgical than
    the full ``TrajectoryOptimizer`` -- targets only harsh/abrupt braking,
    which is a plausible contributor to surprising non-reactive trailing
    traffic into rear-end collisions.

    ``trajectory_xy`` is (T,2) rig-frame offsets (model output convention,
    NOT including the t=0 anchor point). Returns an adjusted (T,2) array.
    """
    xy = np.asarray(trajectory_xy, dtype=np.float64).reshape(-1, 2)
    if xy.shape[0] < 2 or time_step <= 1e-6:
        return xy.copy()

    pts = np.vstack(([[0.0, 0.0]], xy))  # prepend t=0 anchor (ego-relative origin)
    diffs = pts[1:] - pts[:-1]
    seg_len = np.hypot(diffs[:, 0], diffs[:, 1])
    s = np.concatenate(([0.0], np.cumsum(seg_len)))

    v = seg_len / time_step  # implied speed during each interval, length T

    # --- Cap deceleration: forward pass limiting how fast speed can drop ---
    v_capped = v.copy()
    max_drop = max_decel_mps2 * time_step
    for i in range(1, len(v_capped)):
        if v_capped[i] < v_capped[i - 1] - max_drop:
            v_capped[i] = v_capped[i - 1] - max_drop
        v_capped[i] = max(v_capped[i], 0.0)

    # --- Cap jerk: limit how fast acceleration itself can change ---
    if len(v_capped) >= 3:
        accel = np.diff(v_capped) / time_step
        max_da = max_jerk_mps3 * time_step
        for i in range(1, len(accel)):
            da = accel[i] - accel[i - 1]
            if abs(da) > max_da:
                accel[i] = accel[i - 1] + math.copysign(max_da, da)
        v_final = np.concatenate(([v_capped[0]], v_capped[0] + np.cumsum(accel) * time_step))
        v_final = np.clip(v_final, 0.0, None)
    else:
        v_final = v_capped

    s_new = np.concatenate(([0.0], np.cumsum(v_final) * time_step))
    new_xy = _resample_polyline_by_arclength(pts, s, s_new[1:])
    return new_xy


def make_cached_plan(
    *,
    created_time_us: int,
    anchor_pose: PoseAtTime,
    trajectory_xy: np.ndarray,
    headings: np.ndarray,
    source_frequency_hz: float = 2.0,
) -> CachedPlan | None:
    """Convert raw model output into a local-frame cached plan (or ``None``)."""
    offsets = np.asarray(trajectory_xy, dtype=np.float64).reshape(-1, 2)
    if offsets.shape[0] == 0:
        return None

    step_s = 1.0 / source_frequency_hz
    future_times = np.arange(1, offsets.shape[0] + 1, dtype=np.float64) * step_s
    times_s = np.concatenate(([0.0], future_times))

    anchor_xy = np.array(
        [[anchor_pose.pose.vec.x, anchor_pose.pose.vec.y]], dtype=np.float64
    )
    positions_xy = np.vstack(
        (anchor_xy, rig_offsets_to_local_positions(anchor_pose, offsets))
    )

    anchor_yaw = yaw_from_quat(anchor_pose.pose.quat)
    model_yaws = np.asarray(headings, dtype=np.float64).reshape(-1) + anchor_yaw
    if model_yaws.size != offsets.shape[0]:
        model_yaws = _headings_from_positions(positions_xy)[1:]
    yaws = np.concatenate(([anchor_yaw], model_yaws))

    return CachedPlan(created_time_us, times_s, positions_xy, yaws)


def build_trajectory_from_plan(
    plan: CachedPlan | None,
    current_pose: PoseAtTime | None,
    time_now_us: int,
    *,
    callback_frequency_hz: float = 10.0,
    fallback_speed_mps: float = 5.0,
    max_horizon_s: float = 5.0,
) -> Trajectory:
    if current_pose is None:
        current_pose = PoseAtTime(
            timestamp_us=time_now_us,
            pose=Pose(vec=Vec3(x=0.0, y=0.0, z=0.0), quat=Quat(w=1.0)),
        )

    dt_s = 1.0 / callback_frequency_hz
    if plan is None or len(plan.times_s) < 2:
        return build_straight_line_trajectory(
            current_pose,
            time_now_us,
            speed_mps=max(1.0, fallback_speed_mps),
            horizon_s=max_horizon_s,
            frequency_hz=callback_frequency_hz,
        )

    # How far along the cached plan we are.  The plan stays fixed in the local
    # frame; we only stop emitting the points the ego has already driven past.
    elapsed_s = max(0.0, (time_now_us - plan.created_time_us) / 1_000_000.0)
    horizon_end_s = min(float(plan.times_s[-1]), elapsed_s + max_horizon_s)

    # Sample the plan on a grid fixed to its own clock (multiples of dt from
    # created_time), so a given plan-time always maps to the same point.  We
    # keep only the samples at or after "now": the trajectory shrinks, but no
    # point moves until the next inference replaces the plan.
    first_step = int(math.ceil(elapsed_s / dt_s - 1e-9))
    last_step = int(math.floor(horizon_end_s / dt_s + 1e-9))
    if last_step - first_step < 1:
        return build_straight_line_trajectory(
            current_pose,
            time_now_us,
            speed_mps=max(1.0, fallback_speed_mps),
            horizon_s=max_horizon_s,
            frequency_hz=callback_frequency_hz,
        )

    sample_times = np.arange(first_step, last_step + 1, dtype=np.float64) * dt_s
    xs = np.interp(sample_times, plan.times_s, plan.positions_xy[:, 0])
    ys = np.interp(sample_times, plan.times_s, plan.positions_xy[:, 1])
    yaws = np.interp(sample_times, plan.times_s, np.unwrap(plan.yaws))

    cur_z = float(current_pose.pose.vec.z)
    trajectory = Trajectory()
    for t_s, x, y, yaw in zip(sample_times, xs, ys, yaws, strict=True):
        trajectory.poses.append(
            PoseAtTime(
                timestamp_us=plan.created_time_us + int(round(float(t_s) * 1_000_000)),
                pose=Pose(
                    vec=Vec3(x=float(x), y=float(y), z=cur_z),
                    quat=quat_from_yaw(float(yaw)),
                ),
            )
        )
    return trajectory


def build_straight_line_trajectory(
    start_pose: PoseAtTime,
    time_now_us: int,
    *,
    speed_mps: float,
    horizon_s: float,
    frequency_hz: float,
) -> Trajectory:
    yaw = yaw_from_quat(start_pose.pose.quat)
    dx = math.cos(yaw)
    dy = math.sin(yaw)
    dt_s = 1.0 / frequency_hz
    n_points = max(2, int(math.ceil(horizon_s * frequency_hz)) + 1)

    trajectory = Trajectory()
    for i in range(n_points):
        t_s = i * dt_s
        trajectory.poses.append(
            PoseAtTime(
                timestamp_us=time_now_us + int(round(t_s * 1_000_000)),
                pose=Pose(
                    vec=Vec3(
                        x=float(start_pose.pose.vec.x + dx * speed_mps * t_s),
                        y=float(start_pose.pose.vec.y + dy * speed_mps * t_s),
                        z=float(start_pose.pose.vec.z),
                    ),
                    quat=start_pose.pose.quat,
                ),
            )
        )
    return trajectory


def _headings_from_positions(positions_xy: np.ndarray) -> np.ndarray:
    prev = np.vstack((positions_xy[0:1], positions_xy[:-1]))
    deltas = positions_xy - prev
    return np.arctan2(deltas[:, 1], deltas[:, 0])
