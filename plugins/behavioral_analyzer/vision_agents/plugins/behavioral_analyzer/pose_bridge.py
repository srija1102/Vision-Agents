"""PoseBridge — extends YOLOPoseProcessor to feed pose metrics into BehavioralAnalyzer.

Overrides the frame handler to intercept pose_data on every processed frame,
extract behavioral metrics from COCO keypoints, and forward them to the
BehavioralAnalyzer via ingest_video_event().

COCO keypoint indices (17 body points):
  0=nose  1=l_eye  2=r_eye  3=l_ear  4=r_ear
  5=l_shoulder  6=r_shoulder  7=l_elbow  8=r_elbow
  9=r_wrist  10=l_wrist  11=l_hip  12=r_hip
  13=l_knee  14=r_knee  15=l_ankle  16=r_ankle

Derived metrics:
  posture_stability   — variance of upper-body midpoint over rolling window
  shoulder_tension    — shoulder elevation relative to hip-shoulder distance
  head_jitter         — frame-to-frame nose displacement, normalised
  hand_movement       — wrist velocity over rolling window
"""

import logging
from collections import deque
from typing import TYPE_CHECKING, Any

import av

from vision_agents.plugins.ultralytics import YOLOPoseProcessor

from .analyzer import BehavioralAnalyzer

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_KP_NOSE = 0
_KP_L_SHOULDER = 5
_KP_R_SHOULDER = 6
_KP_R_WRIST = 9
_KP_L_WRIST = 10
_KP_L_HIP = 11
_KP_R_HIP = 12

_CONF_THRESHOLD = 0.30
_POSITION_HISTORY = 10
_MAX_JITTER_FRACTION = 0.05
_MAX_WRIST_MOVE_FRACTION = 0.10


class PoseBridge(YOLOPoseProcessor):
    """YOLOPoseProcessor that feeds live pose metrics to a BehavioralAnalyzer.

    Inherits all YOLO initialisation args. Pass the analyzer as the first arg.

    Args:
        analyzer: BehavioralAnalyzer instance to receive video metrics.
        **kwargs: Forwarded to YOLOPoseProcessor (model_path, fps, etc.).

    Raises:
        ValueError: If analyzer is None.
    """

    name = "pose_bridge"

    def __init__(self, analyzer: BehavioralAnalyzer, **kwargs) -> None:
        if analyzer is None:
            raise ValueError("analyzer must not be None")
        super().__init__(**kwargs)
        self._pose_analyzer = analyzer

        self._prev_nose_xy: tuple[float, float] | None = None
        self._shoulder_mid_history: deque[tuple[float, float]] = deque(
            maxlen=_POSITION_HISTORY
        )
        self._wrist_history: deque[tuple[float, float]] = deque(
            maxlen=_POSITION_HISTORY
        )

    async def _add_pose_and_add_frame(self, frame: av.VideoFrame) -> None:
        """Override to intercept pose_data before discarding it."""
        if self._shutdown:
            return

        try:
            frame_array = frame.to_ndarray(format="rgb24")
            annotated_array, pose_data = await self.add_pose_to_ndarray(frame_array)
            await self._video_track.add_frame(av.VideoFrame.from_ndarray(annotated_array))

            persons = pose_data.get("persons") if pose_data else None
            if persons:
                metrics = self._extract_metrics(
                    keypoints=persons[0]["keypoints"],
                    frame_w=frame_array.shape[1],
                    frame_h=frame_array.shape[0],
                )
                if metrics:
                    await self._pose_analyzer.ingest_video_event(**metrics)

        except (ValueError, RuntimeError):
            logger.exception("PoseBridge frame processing failed")
            await self._video_track.add_frame(frame)

    def _extract_metrics(
        self,
        keypoints: list[list[float]],
        frame_w: int,
        frame_h: int,
    ) -> dict[str, Any] | None:
        """Derive behavioural metrics from a single frame's COCO keypoints."""
        if len(keypoints) < 13:
            return None

        def kp(idx: int) -> tuple[float, float, float]:
            if idx < len(keypoints):
                row = keypoints[idx]
                return float(row[0]), float(row[1]), float(row[2])
            return 0.0, 0.0, 0.0

        nose_x, nose_y, nose_c = kp(_KP_NOSE)
        l_sh_x, l_sh_y, l_sh_c = kp(_KP_L_SHOULDER)
        r_sh_x, r_sh_y, r_sh_c = kp(_KP_R_SHOULDER)
        r_wr_x, r_wr_y, r_wr_c = kp(_KP_R_WRIST)
        l_wr_x, l_wr_y, l_wr_c = kp(_KP_L_WRIST)
        l_hp_x, l_hp_y, l_hp_c = kp(_KP_L_HIP)
        r_hp_x, r_hp_y, r_hp_c = kp(_KP_R_HIP)

        metrics: dict[str, Any] = {}

        # ── posture_stability ──────────────────────────────────────────────────
        # Variance of shoulder midpoint over rolling window → low variance = stable
        if l_sh_c > _CONF_THRESHOLD and r_sh_c > _CONF_THRESHOLD:
            sh_mid_x = (l_sh_x + r_sh_x) / 2.0
            sh_mid_y = (l_sh_y + r_sh_y) / 2.0
            self._shoulder_mid_history.append((sh_mid_x, sh_mid_y))

            if len(self._shoulder_mid_history) >= 3:
                xs = [p[0] for p in self._shoulder_mid_history]
                ys = [p[1] for p in self._shoulder_mid_history]
                mean_x = sum(xs) / len(xs)
                mean_y = sum(ys) / len(ys)
                var = sum(
                    (x - mean_x) ** 2 + (y - mean_y) ** 2
                    for x, y in self._shoulder_mid_history
                ) / len(self._shoulder_mid_history)
                max_var = (frame_w * 0.05) ** 2
                instability = min(var / max_var, 1.0)
                metrics["posture_stability"] = 1.0 - instability

        # ── shoulder_tension ───────────────────────────────────────────────────
        # Raised shoulders = shoulders closer to hips vertically relative to norm
        if l_sh_c > _CONF_THRESHOLD and l_hp_c > _CONF_THRESHOLD:
            sh_mid_y = (l_sh_y + r_sh_y) / 2.0 if r_sh_c > _CONF_THRESHOLD else l_sh_y
            hp_mid_y = (l_hp_y + r_hp_y) / 2.0 if r_hp_c > _CONF_THRESHOLD else l_hp_y
            body_height = abs(hp_mid_y - sh_mid_y)
            if body_height > 0:
                natural_ratio = 0.60
                actual_ratio = (hp_mid_y - sh_mid_y) / body_height
                tension = max(0.0, min(1.0, 1.0 - (actual_ratio / natural_ratio)))
                metrics["shoulder_tension"] = tension

        # ── head_jitter ────────────────────────────────────────────────────────
        # Frame-to-frame nose displacement, normalised by 5% of frame width
        if nose_c > _CONF_THRESHOLD:
            if self._prev_nose_xy is not None:
                px, py = self._prev_nose_xy
                dist = ((nose_x - px) ** 2 + (nose_y - py) ** 2) ** 0.5
                jitter = min(dist / (frame_w * _MAX_JITTER_FRACTION), 1.0)
                metrics["head_jitter"] = jitter
            self._prev_nose_xy = (nose_x, nose_y)

        # ── hand_movement ──────────────────────────────────────────────────────
        # Wrist velocity: dominant wrist (highest confidence) displacement
        if r_wr_c > _CONF_THRESHOLD or l_wr_c > _CONF_THRESHOLD:
            wx = r_wr_x if r_wr_c >= l_wr_c else l_wr_x
            wy = r_wr_y if r_wr_c >= l_wr_c else l_wr_y
            if self._wrist_history:
                px, py = self._wrist_history[-1]
                dist = ((wx - px) ** 2 + (wy - py) ** 2) ** 0.5
                movement = min(dist / (frame_w * _MAX_WRIST_MOVE_FRACTION), 1.0)
                metrics["hand_movement"] = movement
            self._wrist_history.append((wx, wy))

        return metrics if metrics else None
