# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Standalone Transfuser (LTFv6/NAVSIM) model wrapper for the e2e challenge driver.

This mirrors ``plugins/transfuser_driver/alpasim_transfuser/transfuser_model.py``
but drops the dependency on ``alpasim_driver`` (which pulls in unrelated,
heavyweight, and partly gated model packages as hard dependencies). The only
in-repo file this reuses is ``transfuser_impl.py`` (copied verbatim), which is
self-contained (cv2/numpy/torch/timm/beartype/omegaconf only).

Camera order and mapping to raw nuPlan camera logical IDs (CAM_L0, CAM_F0,
CAM_R0, CAM_B0) come directly from ``NUPLAN_CAMERA_CALIBRATION`` /
``TargetDataset.NAVSIM_4CAMERAS`` in ``transfuser_impl.py`` -- that's the
camera set and order the checkpoint was trained on.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import torch
from PIL import Image

from .transfuser_impl import load_tf

logger = logging.getLogger(__name__)

# --- Stage 0: correctness-risk toggles (see experiments/transfuser_postprocessing/PLAN.md) ---
#
# TRANSFUSER_CHANNEL_ORDER: the training loader (NavsimData.__getitem__ in
# transfuser_impl.py) decodes images via cv2.imdecode(IMREAD_COLOR) -- BGR --
# into a dict key called "rgb", with zero cvtColor calls anywhere in that
# file. Our driver decodes true RGB via PIL. If the (not-in-this-repo)
# upstream feature-cache builder that produced the training crops also never
# converted BGR->RGB, the checkpoint expects BGR-ordered input. Set to "bgr"
# to feed the network channel-swapped input for A/B testing.
_CHANNEL_ORDER = os.environ.get("TRANSFUSER_CHANNEL_ORDER", "rgb").lower()

# TRANSFUSER_APPLY_BOTTOM_CROP: training geometry crops raw 1920x1120 ->
# 1920x1080 by removing 40px off the BOTTOM before downsampling (confirmed
# via NUPLAN_CAMERA_CALIBRATION / carla_crop_height_type in
# transfuser_impl.py). Our pipeline assumes MTGS's 1920x1080 render already
# is the post-crop FOV. Set to "1" to apply the equivalent bottom crop
# (~3.57% of height) before the resize, for A/B testing.
_APPLY_BOTTOM_CROP = os.environ.get("TRANSFUSER_APPLY_BOTTOM_CROP", "0") == "1"
_BOTTOM_CROP_KEEP_FRAC = 1080.0 / 1120.0

# Camera IDs in concatenation order, as used for NAVSIM_4CAMERAS training data
# (see NUPLAN_CAMERA_CALIBRATION / camera_calibration() in transfuser_impl.py).
NUPLAN_CAMERA_ORDER: tuple[str, ...] = ("CAM_L0", "CAM_F0", "CAM_R0", "CAM_B0")


class DriveCommand(IntEnum):
    """Canonical driving command. Values match Transfuser's own encoding
    (LEFT=0, FORWARD/STRAIGHT=1, RIGHT=2, UNDEFINED/UNKNOWN=3) 1:1, so no
    remapping is needed before one-hot encoding."""

    LEFT = 0
    STRAIGHT = 1
    RIGHT = 2
    UNKNOWN = 3


@dataclass
class ModelPrediction:
    trajectory_xy: np.ndarray  # (T, 2) x,y offsets in rig frame
    headings: np.ndarray  # (T,) headings in radians (rig frame)


def _resize_and_center_crop(
    image: np.ndarray, target_height: int, target_width: int
) -> np.ndarray:
    """Resize to target height (preserving aspect ratio) and center-crop width."""
    if _APPLY_BOTTOM_CROP:
        keep_rows = int(round(image.shape[0] * _BOTTOM_CROP_KEEP_FRAC))
        image = image[:keep_rows]

    h, w = image.shape[:2]
    if h == target_height and w == target_width:
        return image

    pil_img = Image.fromarray(image)
    scale = target_height / h
    new_w = int(w * scale)
    pil_img = pil_img.resize((new_w, target_height), Image.Resampling.BILINEAR)

    if new_w > target_width:
        left = (new_w - target_width) // 2
        pil_img = pil_img.crop((left, 0, left + target_width, target_height))
    elif new_w < target_width:
        raise ValueError(
            f"Image width {new_w} too small after resize, need {target_width}"
        )
    return np.array(pil_img)


def _compute_headings_from_trajectory(trajectory_xy: np.ndarray) -> np.ndarray:
    """Heading = direction of travel from the previous waypoint (origin for the first)."""
    prev = np.zeros_like(trajectory_xy)
    prev[1:] = trajectory_xy[:-1]
    deltas = trajectory_xy - prev
    return np.arctan2(deltas[:, 1], deltas[:, 0])


class TransfuserModel:
    """Loads LTFv6 and runs single-frame, 4-camera trajectory inference."""

    EXPECTED_HEIGHT = 270
    EXPECTED_WIDTH_PER_CAM = 480

    def __init__(
        self,
        checkpoint_path: str,
        device: torch.device,
        camera_ids: tuple[str, ...] = NUPLAN_CAMERA_ORDER,
    ) -> None:
        if len(camera_ids) != 4:
            raise ValueError(f"Transfuser requires exactly 4 cameras, got {camera_ids}")
        self._camera_ids = tuple(camera_ids)
        self._device = device
        self._model = load_tf(checkpoint_path, device)
        self._config = self._model.config
        logger.info(
            "Loaded Transfuser model from %s with cameras %s",
            checkpoint_path,
            self._camera_ids,
        )

    @property
    def camera_ids(self) -> tuple[str, ...]:
        return self._camera_ids

    @property
    def output_frequency_hz(self) -> float:
        return 2.0  # NAVSIM config: waypoints_spacing=10 at 20fps -> 2Hz

    def _concatenate_cameras(self, camera_images: dict[str, np.ndarray]) -> np.ndarray:
        resized = [
            _resize_and_center_crop(
                camera_images[cam_id], self.EXPECTED_HEIGHT, self.EXPECTED_WIDTH_PER_CAM
            )
            for cam_id in self._camera_ids
        ]
        concatenated = np.concatenate(resized, axis=1)
        if _CHANNEL_ORDER == "bgr":
            # torch.from_numpy rejects negative-stride views, hence the copy.
            concatenated = np.ascontiguousarray(concatenated[..., ::-1])
        return concatenated

    def predict(
        self,
        camera_images: dict[str, np.ndarray],
        command: DriveCommand,
        speed: float,
        acceleration: float,
    ) -> ModelPrediction:
        missing = set(self._camera_ids) - set(camera_images)
        if missing:
            raise ValueError(f"Missing camera images for: {missing}")

        concatenated = self._concatenate_cameras(camera_images)
        rgb = torch.from_numpy(concatenated).permute(2, 0, 1).unsqueeze(0).to(self._device)

        command_tensor = torch.nn.functional.one_hot(
            torch.tensor([int(command)], device=self._device, dtype=torch.long),
            num_classes=4,
        ).float()

        float_dtype = self._config.torch_float_type
        speed_tensor = torch.tensor([speed], device=self._device, dtype=float_dtype)
        accel_tensor = torch.tensor(
            [acceleration], device=self._device, dtype=float_dtype
        )

        data = {
            "rgb": rgb,
            "command": command_tensor,
            "speed": speed_tensor,
            "acceleration": accel_tensor,
        }

        with torch.no_grad():
            prediction = self._model(data)

        # CARLA: X+ forward, Y+ right; rig frame: X+ forward, Y+ left.
        waypoints = prediction.pred_future_waypoints[0].cpu().numpy()  # (N, 2)
        waypoints[:, 1] *= -1

        if prediction.pred_headings is not None:
            headings = prediction.pred_headings[0].cpu().numpy() * -1
        else:
            headings = _compute_headings_from_trajectory(waypoints)

        return ModelPrediction(trajectory_xy=waypoints, headings=headings)
