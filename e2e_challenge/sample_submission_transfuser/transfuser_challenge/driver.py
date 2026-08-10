#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""AlpaSim e2e challenge driver backed by Transfuser (LTFv6/NAVSIM).

Targets the nuPlan/MTGS track. MTGS renders nuPlan cameras with real nuPlan
pinhole intrinsics (see plugins/mtgs/server/artifact_adapter.py), which already
match what LTFv6 was trained on -- unlike the PAI/NuRec track's f-theta
cameras, no rectification step is needed here.

Uses cameras CAM_L0, CAM_F0, CAM_R0, CAM_B0 (in that order), the exact 4-of-8
subset and ordering the checkpoint expects for NAVSIM_4CAMERAS
(see NUPLAN_CAMERA_CALIBRATION / camera_calibration() in transfuser_impl.py).
Runs model inference inline in ``drive`` at the model's native 2 Hz and caches
the resulting plan between inferences, mirroring sample_submission_vavam.
"""

from __future__ import annotations

import logging
import os
import signal
import threading
import time
from concurrent import futures
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any

import numpy as np
import torch
from alpasim_grpc import API_VERSION_MESSAGE
from alpasim_grpc.v0 import common_pb2, egodriver_pb2, egodriver_pb2_grpc
from PIL import Image

import grpc

from .model import NUPLAN_CAMERA_ORDER, DriveCommand, TransfuserModel
from .trajectory import (
    CachedPlan,
    apply_temporal_consistency,
    build_trajectory_from_plan,
    limit_braking_and_jerk,
    make_cached_plan,
    stabilize_headings,
)
from .trajectory_optimizer import TrajectoryOptimizer, VehicleConstraints

_RUNTIME_WRITE_DIR_DEFAULTS = {
    "XDG_CACHE_HOME": "/tmp/.cache",
    "TORCH_HOME": "/tmp/torch",
    "HF_HOME": "/tmp/huggingface",
    "MPLCONFIGDIR": "/tmp/matplotlib",
    "CUDA_CACHE_PATH": "/tmp/nv",
    "NUMBA_CACHE_DIR": "/tmp/numba",
    "ALPASIM_DRIVER_LOG_DIR": "/run/alpasim-driver",
}
for _key, _value in _RUNTIME_WRITE_DIR_DEFAULTS.items():
    os.environ.setdefault(_key, _value)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TORCH_NUM_THREADS", "1")
os.environ.setdefault("TORCH_NUM_INTEROP_THREADS", "1")

LOGGER = logging.getLogger("transfuser_challenge_driver")


def _configure_torch_threads() -> None:
    torch.set_num_threads(max(1, int(os.environ["TORCH_NUM_THREADS"])))
    torch.set_num_interop_threads(max(1, int(os.environ["TORCH_NUM_INTEROP_THREADS"])))


@dataclass
class SessionState:
    latest_pose: common_pb2.PoseAtTime | None = None
    latest_images: dict[str, np.ndarray] = field(default_factory=dict)
    poses: list[common_pb2.PoseAtTime] = field(default_factory=list)
    dynamic_states: list[tuple[int, common_pb2.DynamicState]] = field(
        default_factory=list
    )
    command: DriveCommand = DriveCommand.STRAIGHT
    cached_plan: CachedPlan | None = None


class TransfuserPolicyHandle:
    """Loads the heavyweight model after gRPC startup and exposes readiness."""

    def __init__(self, *, checkpoint_path: str, device: str) -> None:
        self._checkpoint_path = checkpoint_path
        self._device = device
        self._lock = threading.Lock()
        self._model: TransfuserModel | None = None
        self._load_error: BaseException | None = None

    def start(self) -> None:
        threading.Thread(
            target=self._load_model, name="transfuser-model-loader", daemon=True
        ).start()

    def get(self) -> TransfuserModel | None:
        with self._lock:
            return self._model

    def load_error(self) -> BaseException | None:
        with self._lock:
            return self._load_error

    def ready(self) -> bool:
        with self._lock:
            return self._model is not None

    def _load_model(self) -> None:
        try:
            LOGGER.info(
                "loading Transfuser checkpoint=%s device=%s",
                self._checkpoint_path,
                self._device,
            )
            model = TransfuserModel(
                checkpoint_path=self._checkpoint_path,
                device=torch.device(self._device),
                camera_ids=NUPLAN_CAMERA_ORDER,
            )
        except BaseException as exc:
            with self._lock:
                self._load_error = exc
            LOGGER.exception("Transfuser model load failed")
            return
        with self._lock:
            self._model = model
        LOGGER.info("Transfuser model load complete")


class TransfuserChallengeDriver(egodriver_pb2_grpc.EgodriverServiceServicer):
    """gRPC service that runs Transfuser inference inline during ``drive``."""

    def __init__(
        self,
        *,
        policy_handle: TransfuserPolicyHandle,
        inference_interval_us: int,
        callback_frequency_hz: float,
        heading_stabilize_enabled: bool = False,
        heading_blend_steps: int = 2,
        heading_yaw_rate_clamp_mult: float = 3.0,
        heading_yaw_rate_floor: float = 0.7,
        traj_optimizer_enabled: bool = False,
        traj_optimizer: TrajectoryOptimizer | None = None,
        vehicle_constraints: VehicleConstraints | None = None,
        temporal_consistency_enabled: bool = False,
        temporal_consistency_blend_steps: int = 4,
        brake_limiter_enabled: bool = False,
        brake_limiter_max_decel: float = 3.0,
        brake_limiter_max_jerk: float = 6.0,
    ) -> None:
        self._policy_handle = policy_handle
        self._inference_interval_us = inference_interval_us
        self._callback_frequency_hz = callback_frequency_hz
        self._heading_stabilize_enabled = heading_stabilize_enabled
        self._heading_blend_steps = heading_blend_steps
        self._heading_yaw_rate_clamp_mult = heading_yaw_rate_clamp_mult
        self._heading_yaw_rate_floor = heading_yaw_rate_floor
        self._traj_optimizer_enabled = traj_optimizer_enabled
        self._traj_optimizer = traj_optimizer
        self._vehicle_constraints = vehicle_constraints
        self._temporal_consistency_enabled = temporal_consistency_enabled
        self._temporal_consistency_blend_steps = temporal_consistency_blend_steps
        self._brake_limiter_enabled = brake_limiter_enabled
        self._brake_limiter_max_decel = brake_limiter_max_decel
        self._brake_limiter_max_jerk = brake_limiter_max_jerk
        self._sessions: dict[str, SessionState] = {}
        self._lock = threading.RLock()
        # Serializes GPU inference across concurrent Drive calls (one model,
        # one GPU): a second session waits its turn rather than running in
        # parallel.
        self._inference_lock = threading.Lock()
        self._server: grpc.Server | None = None

    def attach_server(self, server: grpc.Server) -> None:
        self._server = server

    def start_session(
        self,
        request: egodriver_pb2.DriveSessionRequest,
        context: grpc.ServicerContext,
    ) -> common_pb2.SessionRequestStatus:
        available = {
            cam.logical_id for cam in request.rollout_spec.vehicle.available_cameras
        }
        missing = set(NUPLAN_CAMERA_ORDER) - available
        if missing:
            LOGGER.warning(
                "session %s: expected cameras %s missing from available cameras %s",
                request.session_uuid,
                missing,
                available,
            )
        with self._lock:
            self._sessions[request.session_uuid] = SessionState()
        LOGGER.info("started session %s", request.session_uuid)
        return common_pb2.SessionRequestStatus()

    def close_session(
        self,
        request: egodriver_pb2.DriveSessionCloseRequest,
        context: grpc.ServicerContext,
    ) -> common_pb2.Empty:
        with self._lock:
            self._sessions.pop(request.session_uuid, None)
        LOGGER.info("closed session %s", request.session_uuid)
        return common_pb2.Empty()

    def submit_image_observation(
        self,
        request: egodriver_pb2.RolloutCameraImage,
        context: grpc.ServicerContext,
    ) -> common_pb2.Empty:
        grpc_image = request.camera_image
        if grpc_image.logical_id not in NUPLAN_CAMERA_ORDER:
            return common_pb2.Empty()

        session = self._get_session(request.session_uuid, context)
        image = Image.open(BytesIO(grpc_image.image_bytes)).convert("RGB")
        with self._lock:
            session.latest_images[grpc_image.logical_id] = np.array(image)
        return common_pb2.Empty()

    def submit_egomotion_observation(
        self,
        request: egodriver_pb2.RolloutEgoTrajectory,
        context: grpc.ServicerContext,
    ) -> common_pb2.Empty:
        session = self._get_session(request.session_uuid, context)
        with self._lock:
            if request.trajectory.poses:
                session.poses.extend(request.trajectory.poses)
                session.poses.sort(key=lambda pose: pose.timestamp_us)
                session.latest_pose = session.poses[-1]
            for idx, dynamic_state in enumerate(request.dynamic_states):
                if idx < len(request.trajectory.poses):
                    timestamp_us = int(request.trajectory.poses[idx].timestamp_us)
                    session.dynamic_states.append((timestamp_us, dynamic_state))
            session.dynamic_states.sort(key=lambda item: item[0])
            if len(session.poses) > 32:
                session.poses = session.poses[-32:]
            if len(session.dynamic_states) > 32:
                session.dynamic_states = session.dynamic_states[-32:]
        return common_pb2.Empty()

    def submit_route(
        self,
        request: egodriver_pb2.RouteRequest,
        context: grpc.ServicerContext,
    ) -> common_pb2.Empty:
        session = self._get_session(request.session_uuid, context)
        command = _command_from_route(request.route)
        with self._lock:
            session.command = command
        return common_pb2.Empty()

    def submit_recording_ground_truth(
        self,
        request: egodriver_pb2.GroundTruthRequest,
        context: grpc.ServicerContext,
    ) -> common_pb2.Empty:
        return common_pb2.Empty()

    def drive(
        self,
        request: egodriver_pb2.DriveRequest,
        context: grpc.ServicerContext,
    ) -> egodriver_pb2.DriveResponse:
        session = self._get_session(request.session_uuid, context)
        time_now_us = int(request.time_now_us)
        self._maybe_run_inference(session, time_now_us)

        with self._lock:
            pose = session.latest_pose
            plan = session.cached_plan
            speed = _estimate_speed_mps(session)

        trajectory = build_trajectory_from_plan(
            plan,
            pose,
            time_now_us,
            callback_frequency_hz=self._callback_frequency_hz,
            fallback_speed_mps=max(2.0, speed),
        )
        return egodriver_pb2.DriveResponse(trajectory=trajectory)

    def get_version(
        self,
        request: common_pb2.Empty,
        context: grpc.ServicerContext,
    ) -> common_pb2.VersionId:
        if os.environ.get("TRANSFUSER_REQUIRE_MODEL_FOR_VERSION", "1") == "1":
            load_error = self._policy_handle.load_error()
            if load_error is not None:
                context.abort(
                    grpc.StatusCode.UNAVAILABLE,
                    f"Transfuser model load failed: {load_error}",
                )
            if not self._policy_handle.ready():
                context.abort(
                    grpc.StatusCode.UNAVAILABLE, "Transfuser model is still loading"
                )
        return common_pb2.VersionId(
            version_id="transfuser-e2e-driver",
            git_hash="local",
            grpc_api_version=API_VERSION_MESSAGE,
        )

    def shut_down(
        self,
        request: common_pb2.Empty,
        context: grpc.ServicerContext,
    ) -> common_pb2.Empty:
        if self._server is not None:
            threading.Thread(target=self._stop_server, daemon=True).start()
        return common_pb2.Empty()

    def _stop_server(self) -> None:
        time.sleep(0.05)
        if self._server is not None:
            self._server.stop(grace=0.0)

    def _should_run_inference(self, session: SessionState, time_now_us: int) -> bool:
        if session.latest_pose is None:
            return False
        if set(session.latest_images) != set(NUPLAN_CAMERA_ORDER):
            return False
        plan = session.cached_plan
        if (
            plan is not None
            and time_now_us - plan.created_time_us < self._inference_interval_us
        ):
            return False
        return True

    def _maybe_run_inference(self, session: SessionState, time_now_us: int) -> None:
        model = self._policy_handle.get()
        if model is None:
            return

        with self._lock:
            if not self._should_run_inference(session, time_now_us):
                return
            images = dict(session.latest_images)
            anchor_pose = session.latest_pose
            command = session.command
            speed = _estimate_speed_mps(session)
            acceleration = _estimate_acceleration_mps2(session)
            yaw_rate = _estimate_yaw_rate(session)
            previous_plan = session.cached_plan

        with self._inference_lock:
            try:
                prediction = model.predict(images, command, speed, acceleration)
            except Exception:
                LOGGER.exception("Transfuser inference failed")
                return

        headings = prediction.headings
        trajectory_xy = prediction.trajectory_xy

        # Candidate B: strong cross-inference temporal consistency (position
        # + heading), applied first since it targets the seam discontinuity
        # directly -- everything downstream should operate on an already
        # cross-inference-consistent trajectory.
        if self._temporal_consistency_enabled:
            trajectory_xy, headings = apply_temporal_consistency(
                raw_trajectory_xy=trajectory_xy,
                raw_headings=headings,
                anchor_pose=anchor_pose,
                previous_plan=previous_plan,
                new_created_time_us=time_now_us,
                source_frequency_hz=model.output_frequency_hz,
                blend_steps=self._temporal_consistency_blend_steps,
            )

        if self._heading_stabilize_enabled:
            headings = stabilize_headings(
                raw_headings=headings,
                anchor_pose=anchor_pose,
                previous_plan=previous_plan,
                new_created_time_us=time_now_us,
                ego_yaw_rate=yaw_rate,
                source_frequency_hz=model.output_frequency_hz,
                blend_steps=self._heading_blend_steps,
                yaw_rate_clamp_mult=self._heading_yaw_rate_clamp_mult,
                yaw_rate_floor=self._heading_yaw_rate_floor,
            )

        # Candidate C: moderate brake/jerk limiter, applied before the
        # spatial optimizer so the optimizer's own comfort cost starts from
        # an already-physically-plausible speed profile.
        if self._brake_limiter_enabled:
            trajectory_xy = limit_braking_and_jerk(
                trajectory_xy,
                time_step=1.0 / model.output_frequency_hz,
                max_decel_mps2=self._brake_limiter_max_decel,
                max_jerk_mps3=self._brake_limiter_max_jerk,
            )

        if self._traj_optimizer_enabled and self._traj_optimizer is not None:
            # Optimizer operates on rig-frame [N,3] (x, y, heading); keep the
            # (possibly stabilized) headings as the output heading source
            # regardless of what the optimizer does to its own heading
            # column -- matches the one existing usage precedent in
            # alpasim_driver/main.py, which treats "heading source" and "xy
            # smoother" as separate concerns.
            rig_traj_3 = np.column_stack([trajectory_xy, headings])
            try:
                result = self._traj_optimizer.optimize(
                    trajectory=rig_traj_3,
                    time_step=1.0 / model.output_frequency_hz,
                    vehicle_constraints=self._vehicle_constraints,
                )
                if result.success:
                    trajectory_xy = result.trajectory[:, :2]
            except Exception:
                LOGGER.exception(
                    "Trajectory optimizer failed, using unoptimized trajectory"
                )

        plan = make_cached_plan(
            created_time_us=time_now_us,
            anchor_pose=anchor_pose,
            trajectory_xy=trajectory_xy,
            headings=headings,
            source_frequency_hz=model.output_frequency_hz,
        )
        if plan is not None:
            with self._lock:
                session.cached_plan = plan

    def _get_session(
        self,
        session_uuid: str,
        context: grpc.ServicerContext,
    ) -> SessionState:
        with self._lock:
            session = self._sessions.get(session_uuid)
        if session is None:
            context.abort(grpc.StatusCode.NOT_FOUND, f"unknown session {session_uuid}")
            raise AssertionError("unreachable")
        return session


def _command_from_route(route: egodriver_pb2.Route) -> DriveCommand:
    """Rig frame: X+ forward, Y+ left. Pick the first waypoint past the min
    lookahead distance and classify by its lateral offset."""
    threshold_m = float(os.environ.get("TRANSFUSER_ROUTE_LATERAL_THRESHOLD_M", "3.0"))
    min_lookahead_m = float(os.environ.get("TRANSFUSER_ROUTE_MIN_LOOKAHEAD_M", "20.0"))
    candidates = [wp for wp in route.waypoints if wp.x >= min_lookahead_m]
    waypoint = (
        candidates[0]
        if candidates
        else (route.waypoints[-1] if route.waypoints else None)
    )
    if waypoint is None:
        return DriveCommand.STRAIGHT
    if waypoint.y > threshold_m:
        return DriveCommand.LEFT
    if waypoint.y < -threshold_m:
        return DriveCommand.RIGHT
    return DriveCommand.STRAIGHT


def _estimate_speed_mps(session: SessionState) -> float:
    if session.dynamic_states:
        state = session.dynamic_states[-1][1]
        return float(np.hypot(state.linear_velocity.x, state.linear_velocity.y))
    if len(session.poses) >= 2:
        a, b = session.poses[-2], session.poses[-1]
        dt = (int(b.timestamp_us) - int(a.timestamp_us)) / 1_000_000.0
        if dt > 1e-6:
            return float(
                np.hypot(b.pose.vec.x - a.pose.vec.x, b.pose.vec.y - a.pose.vec.y) / dt
            )
    return 5.0


def _estimate_acceleration_mps2(session: SessionState) -> float:
    if session.dynamic_states:
        state = session.dynamic_states[-1][1]
        # Longitudinal (rig X-axis) acceleration.
        return float(state.linear_acceleration.x)
    return 0.0


def _estimate_yaw_rate(session: SessionState) -> float:
    """Ego's own measured yaw rate (rad/s), free signal from DynamicState."""
    if session.dynamic_states:
        state = session.dynamic_states[-1][1]
        return float(state.angular_velocity.z)
    return 0.0


def _configure_runtime_write_dirs() -> None:
    for key, value in _RUNTIME_WRITE_DIR_DEFAULTS.items():
        os.environ.setdefault(key, value)
    for key in _RUNTIME_WRITE_DIR_DEFAULTS:
        path = os.environ[key]
        try:
            os.makedirs(path, exist_ok=True)
        except OSError:
            if key != "ALPASIM_DRIVER_LOG_DIR":
                raise
            fallback = "/tmp/alpasim-driver"
            os.environ[key] = fallback
            os.makedirs(fallback, exist_ok=True)


def main() -> None:
    _configure_torch_threads()
    _configure_runtime_write_dirs()
    logging.basicConfig(
        level=os.environ.get("ALPASIM_DRIVER_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    host = os.environ.get("ALPASIM_DRIVER_HOST", "0.0.0.0")
    port = int(os.environ.get("ALPASIM_DRIVER_PORT", "6789"))
    checkpoint_path = os.environ.get(
        "TRANSFUSER_CHECKPOINT_PATH", "/app/assets/transfuser/model_0060.pth"
    )

    policy_handle = TransfuserPolicyHandle(
        checkpoint_path=checkpoint_path,
        device=os.environ.get("TRANSFUSER_DEVICE", "cuda"),
    )

    traj_optimizer_enabled = (
        os.environ.get("TRANSFUSER_TRAJ_OPTIMIZER_ENABLED", "1") == "1"
    )
    # max_deviation deliberately defaults far tighter than upstream's own 2.0m:
    # dist_to_gt_trajectory >= 4.0m triggers the same eval truncation as a
    # collision, and the raw model output is already close to GT -- letting
    # the optimizer redraw the path by meters per waypoint works against the
    # truncation-avoidance priority this driver is tuned for. See
    # experiments/transfuser_postprocessing/PLAN.md.
    vehicle_constraints = VehicleConstraints(
        max_deviation=float(
            os.environ.get("TRANSFUSER_TRAJ_OPTIMIZER_MAX_DEVIATION_M", "0.5")
        ),
        max_heading_change=float(
            os.environ.get("TRANSFUSER_TRAJ_OPTIMIZER_MAX_HEADING_CHANGE_RAD", "0.5236")
        ),
        max_abs_yaw_rate=float(
            os.environ.get("TRANSFUSER_TRAJ_OPTIMIZER_MAX_YAW_RATE", "0.95")
        ),
        max_abs_yaw_acc=float(
            os.environ.get("TRANSFUSER_TRAJ_OPTIMIZER_MAX_YAW_ACC", "1.93")
        ),
        max_lon_acc_pos=float(
            os.environ.get("TRANSFUSER_TRAJ_OPTIMIZER_MAX_LON_ACC_POS", "4.89")
        ),
        max_lon_acc_neg=float(
            os.environ.get("TRANSFUSER_TRAJ_OPTIMIZER_MAX_LON_ACC_NEG", "-4.05")
        ),
        max_abs_lon_jerk=float(
            os.environ.get("TRANSFUSER_TRAJ_OPTIMIZER_MAX_LON_JERK", "8.37")
        ),
    )
    traj_optimizer = TrajectoryOptimizer(
        smoothness_weight=float(
            os.environ.get("TRANSFUSER_TRAJ_OPTIMIZER_SMOOTHNESS_WEIGHT", "3.0")
        ),
        deviation_weight=float(
            os.environ.get("TRANSFUSER_TRAJ_OPTIMIZER_DEVIATION_WEIGHT", "0.1")
        ),
        comfort_weight=float(
            os.environ.get("TRANSFUSER_TRAJ_OPTIMIZER_COMFORT_WEIGHT", "2.0")
        ),
        max_iterations=int(
            os.environ.get("TRANSFUSER_TRAJ_OPTIMIZER_MAX_ITERATIONS", "100")
        ),
        enable_frenet_retiming=os.environ.get(
            "TRANSFUSER_TRAJ_OPTIMIZER_RETIME_IN_FRENET", "1"
        )
        == "1",
    )

    service = TransfuserChallengeDriver(
        policy_handle=policy_handle,
        inference_interval_us=int(
            os.environ.get("TRANSFUSER_INFERENCE_INTERVAL_US", "500000")
        ),
        callback_frequency_hz=float(
            os.environ.get("TRANSFUSER_CALLBACK_FREQUENCY_HZ", "10.0")
        ),
        heading_stabilize_enabled=os.environ.get(
            "TRANSFUSER_HEADING_STABILIZE_ENABLED", "0"
        )
        == "1",
        heading_blend_steps=int(os.environ.get("TRANSFUSER_HEADING_BLEND_STEPS", "2")),
        heading_yaw_rate_clamp_mult=float(
            os.environ.get("TRANSFUSER_HEADING_YAW_RATE_CLAMP_MULT", "3.0")
        ),
        heading_yaw_rate_floor=float(
            os.environ.get("TRANSFUSER_HEADING_YAW_RATE_FLOOR", "0.7")
        ),
        traj_optimizer_enabled=traj_optimizer_enabled,
        traj_optimizer=traj_optimizer,
        vehicle_constraints=vehicle_constraints,
        temporal_consistency_enabled=os.environ.get(
            "TRANSFUSER_TEMPORAL_CONSISTENCY_ENABLED", "0"
        )
        == "1",
        temporal_consistency_blend_steps=int(
            os.environ.get("TRANSFUSER_TEMPORAL_CONSISTENCY_BLEND_STEPS", "4")
        ),
        brake_limiter_enabled=os.environ.get("TRANSFUSER_BRAKE_LIMITER_ENABLED", "0")
        == "1",
        brake_limiter_max_decel=float(
            os.environ.get("TRANSFUSER_BRAKE_LIMITER_MAX_DECEL", "3.0")
        ),
        brake_limiter_max_jerk=float(
            os.environ.get("TRANSFUSER_BRAKE_LIMITER_MAX_JERK", "6.0")
        ),
    )

    grpc_workers = max(1, int(os.environ.get("ALPASIM_DRIVER_GRPC_WORKERS", "4")))
    # grpc-python's default max receive size is 4MB, and the runtime's client
    # channel (service_base.py) sets no custom options either -- raise the
    # cap so full-resolution camera images (1080x1920, up to 8 per step)
    # never get rejected. Matches MAX_GRPC_MESSAGE_BYTES in
    # alpasim_runtime/services/video_model_service.py.
    max_message_bytes = 64 * 1024 * 1024
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=grpc_workers),
        options=[
            ("grpc.max_receive_message_length", max_message_bytes),
            ("grpc.max_send_message_length", max_message_bytes),
        ],
    )
    egodriver_pb2_grpc.add_EgodriverServiceServicer_to_server(service, server)
    service.attach_server(server)

    bound_port = server.add_insecure_port(f"{host}:{port}")
    if bound_port == 0:
        raise RuntimeError(f"failed to bind {host}:{port}")

    def request_stop(signum: int, frame: object) -> None:
        LOGGER.info("received signal %s, stopping", signum)
        server.stop(grace=0.0)

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    server.start()
    LOGGER.info("Transfuser driver listening on %s:%d", host, bound_port)
    policy_handle.start()
    server.wait_for_termination()


if __name__ == "__main__":
    main()
