#!/usr/bin/env python3
"""
Convert OpenDriveVLA public training dependencies into a single JSON file that is
as close as possible to the published OpenDriveVLA stage-3 planning format.

What this script does (officially aligned):
- Reads OpenDriveVLA public training dependencies:
  * nuscenes_infos_temporal_train.pkl / val.pkl (from UniAD)
  * cached_nuscenes_info.pkl (from GPT-Driver style cache)
- Merges records by sample token.
- Builds stage-3 planning prompts following the template described in the
  OpenDriveVLA paper appendix (VI-B1 and VI-B5).
- Converts trajectories from the common ego frame used by nuScenes/UniAD
  (x forward, y left) to the OpenDriveVLA prompt convention
  (x right, y front).
- Exports a single JSON file containing message-style samples + raw fields.

What this script cannot do (because the official training script is not public):
- Reconstruct OpenDriveVLA's internal projected <SCENE>/<TRACK>/<MAP> tokens.
  We therefore keep these segments as literal placeholders.
- Guarantee an exact byte-level recreation of the authors' final training set.

Recommended usage:
    python convert_opendrivevla_aligned_train_json.py \
      --train-pkl DriveVLA/data/infos/nuscenes_infos_temporal_train.pkl \
      --cache-pkl DriveVLA/data/nuscenes/cached_nuscenes_info.pkl \
      --out-json data/opendrivevla_stage3_aligned_train.json
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Clockwise order, matching a common multi-view prompt order used in driving VLMs.
CAM_ORDER = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_FRONT_LEFT",
]

OPEN_DRIVE_VLA_SYSTEM_PROMPT = (
    "You are Open-DriveVLA, an advanced vision-language driving model. "
    "Your core responsibilities include safe trajectory planning and interpretable decision-making. "
    "You generate collision-free driving plans while providing clear, logical explanations for user queries.\n\n"
    "Context:\n"
    "- Coordinates: X-axis is pointing to the right, and Y-axis is pointing to the front. "
    "You are at point (0,0). All coordinates are in meters.\n"
    "- Objective: Generate a 3-second safe driving plan consisting of 6 waypoints, one every 0.5 seconds. "
    "Provide logical responses to user queries.\n\n"
    "Task:\n"
    "- Perception & Prediction: Analyze the driving environment using visual data. "
    "Identify road users and hazards and predict their motion.\n"
    "- Thought Process: Determine critical objects and assess potential hazards. "
    "Consider road constraints and traffic rules.\n"
    "- Trajectory Planning: Define the driving objective. Generate a safe, feasible 3-second route consisting of 6 waypoints.\n"
    "- Explainability & User Interaction: If the user asks a question, provide a clear and logical response.\n\n"
    "Output Format:\n"
    "1. Trajectory (MOST IMPORTANT):\n"
    "- Format: <traj_start>[(x1,y1),(x2,y2),(x3,y3),(x4,y4),(x5,y5),(x6,y6)]<traj_end>\n"
    "2. User Question Response (OPTIONAL):\n"
    "- Format: <answer_start> Answer to the user's question <answer_end>"
)

USER_PROMPT_TEMPLATE = (
    "Scene information: <scene_start><SCENE><scene_end>\n"
    "Object-wise tracking information: <track_start><TRACK><track_end>\n"
    "Map information: <map_start><MAP><map_end>\n"
    "Ego states:\n"
    "- Velocity (vx,vy): {velocity}\n"
    "- Heading Angular Velocity (v_yaw): {yaw_rate}\n"
    "- Acceleration (ax,ay): {acceleration}\n"
    "- Can Bus: {can_bus}\n"
    "- Heading Speed: {speed}\n"
    "- Steering: {steering}\n"
    "Historical trajectory (last 2 seconds): {history}\n"
    "Mission goal: {command}\n"
    "Planning trajectory: <trajectory>"
)

FUTURE_STEPS = 6
HISTORY_STEPS = 4  # 2 seconds at 2 Hz


# ------------------------------ basic utils ------------------------------

def load_pickle(path: str) -> Any:
    p = Path(path)

    with p.open("rb") as f:
        try:
            return pickle.load(f)
        except Exception:
            pass

    with p.open("rb") as f:
        try:
            return pickle.load(f, encoding="latin1")
        except Exception:
            pass

    import torch

    try:
        return torch.load(str(p), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(p), map_location="cpu")


def to_jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [to_jsonable(v) for v in x]
    if isinstance(x, tuple):
        return [to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    return x


def maybe_round(v: float, ndigits: int = 4) -> float:
    return round(float(v), ndigits)


def as_list(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def extract_items(obj: Any) -> List[Dict[str, Any]]:
    if isinstance(obj, dict) and isinstance(obj.get("infos"), list):
        return [x for x in obj["infos"] if isinstance(x, dict)]
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        out: List[Dict[str, Any]] = []
        for k, v in obj.items():
            if not isinstance(v, dict):
                continue
            item = dict(v)
            if "token" not in item and "sample_token" not in item:
                item["token"] = str(k)
            out.append(item)
        if out:
            return out
    raise ValueError("Unsupported pickle structure. Expected dict with infos, list[dict], or dict[token]=info.")


def get_token(info: Dict[str, Any]) -> Optional[str]:
    for key in ("token", "sample_token", "id"):
        value = info.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def index_by_token(items: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for item in items:
        tok = get_token(item)
        if tok:
            out[tok] = item
    return out


def rel_path(path_str: str) -> str:
    s = str(path_str).replace("\\", "/")
    for prefix in ("samples/", "sweeps/"):
        idx = s.find(prefix)
        if idx >= 0:
            return s[idx:]
    return s


# --------------------------- quaternion / pose ---------------------------

def quat_wxyz_to_rotmat(q: Sequence[float]) -> np.ndarray:
    # nuScenes commonly stores quaternion as [w, x, y, z]
    if len(q) != 4:
        raise ValueError(f"Quaternion must have 4 values, got {q}")
    w, x, y, z = [float(v) for v in q]
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def yaw_from_quat_wxyz(q: Sequence[float]) -> float:
    # rotation around z-axis in radians
    w, x, y, z = [float(v) for v in q]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def normalize_angle(angle: float) -> float:
    return (float(angle) + math.pi) % (2 * math.pi) - math.pi


def try_extract_pose(info: Dict[str, Any]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    # common flat keys in nuScenes-style info dicts
    pairs = [
        ("ego2global_translation", "ego2global_rotation"),
        ("ego_translation", "ego_rotation"),
        ("translation", "rotation"),
    ]
    for t_key, r_key in pairs:
        t = info.get(t_key)
        r = info.get(r_key)
        if t is not None and r is not None:
            t_arr = np.asarray(t, dtype=np.float64).reshape(-1)
            r_arr = np.asarray(r, dtype=np.float64).reshape(-1)
            if t_arr.shape[0] >= 3 and r_arr.shape[0] == 4:
                return t_arr[:3], r_arr

    ego_pose = info.get("ego_pose")
    if isinstance(ego_pose, dict):
        t = ego_pose.get("translation")
        r = ego_pose.get("rotation")
        if t is not None and r is not None:
            t_arr = np.asarray(t, dtype=np.float64).reshape(-1)
            r_arr = np.asarray(r, dtype=np.float64).reshape(-1)
            if t_arr.shape[0] >= 3 and r_arr.shape[0] == 4:
                return t_arr[:3], r_arr

    # sometimes available under lidar path metadata
    lidar2ego_t = info.get("lidar2ego_translation")
    lidar2ego_r = info.get("lidar2ego_rotation")
    # not a global pose, so do not use it here
    _ = (lidar2ego_t, lidar2ego_r)
    return None


def global_to_current_ego(current_t: np.ndarray, current_q: np.ndarray, global_xyz: np.ndarray) -> np.ndarray:
    rot = quat_wxyz_to_rotmat(current_q)
    return rot.T @ (global_xyz - current_t)


# ----------------------------- field extraction -----------------------------

def extract_images(info: Dict[str, Any], cam_order: Sequence[str]) -> List[str]:
    # Primary format: info['cams'][cam]['data_path' / 'filename' / ...]
    cams = info.get("cams")
    if isinstance(cams, dict):
        out: List[str] = []
        for cam in cam_order:
            cam_info = cams.get(cam)
            if not isinstance(cam_info, dict):
                continue
            path = (
                cam_info.get("data_path")
                or cam_info.get("img_path")
                or cam_info.get("filename")
                or cam_info.get("path")
            )
            if path:
                out.append(rel_path(str(path)))
        if out:
            return out

    # Secondary format: list of image filenames
    img_filename = info.get("img_filename")
    if isinstance(img_filename, list):
        return [rel_path(str(x)) for x in img_filename if str(x).strip()]

    return []


def to_points_xy(value: Any) -> List[List[float]]:
    if value is None:
        return []
    arr = np.asarray(value, dtype=np.float32)
    if arr.size == 0:
        return []
    if arr.ndim == 1:
        if arr.shape[0] % 2 != 0:
            return []
        arr = arr.reshape(-1, 2)
    elif arr.ndim >= 3:
        arr = arr.reshape(arr.shape[0], -1)
    out: List[List[float]] = []
    for row in arr:
        if len(row) >= 2:
            out.append([float(row[0]), float(row[1])])
    return out


def to_mask_list(value: Any, n: int) -> List[int]:
    if value is None:
        return [1] * n
    arr = np.asarray(value).reshape(-1).tolist()
    out = [1 if float(v) > 0.5 else 0 for v in arr[:n]]
    if len(out) < n:
        out.extend([1] * (n - len(out)))
    return out


def forward_left_to_right_front(points: Sequence[Sequence[float]]) -> List[List[float]]:
    # input: x forward, y left  -> output: x right, y front
    out: List[List[float]] = []
    for p in points:
        if len(p) < 2:
            continue
        x_fwd = float(p[0])
        y_left = float(p[1])
        x_right = -y_left
        y_front = x_fwd
        out.append([maybe_round(x_right), maybe_round(y_front)])
    return out


def infer_command_from_future_forward_left(points: Sequence[Sequence[float]], threshold: float = 2.0) -> str:
    valid = [p for p in points if len(p) >= 2]
    if not valid:
        return "Go Straight"
    y_end = float(valid[min(len(valid), FUTURE_STEPS) - 1][1])
    if y_end <= -threshold:
        return "Turn Right"
    if y_end >= threshold:
        return "Turn Left"
    return "Go Straight"


def onehot_to_command(value: Any) -> Optional[str]:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.shape[0] < 3:
        return None
    idx = int(np.argmax(arr[:3]))
    return ["Turn Right", "Turn Left", "Go Straight"][idx]


def format_points_for_prompt(points: Sequence[Sequence[float]]) -> str:
    clean = []
    for p in points:
        if len(p) < 2:
            continue
        clean.append(f"({float(p[0]):.4f},{float(p[1]):.4f})")
    return "[" + ",".join(clean) + "]"


def format_traj_answer(points: Sequence[Sequence[float]]) -> str:
    return f"<traj_start>{format_points_for_prompt(points)}<traj_end>"


def compact_json(value: Any) -> str:
    return json.dumps(to_jsonable(value), ensure_ascii=False, separators=(",", ":"))


def extract_future_traj_open_drive_vla_frame(info: Dict[str, Any], require_full_future_6: bool) -> Optional[Dict[str, Any]]:
    raw = to_points_xy(info.get("gt_ego_fut_trajs"))
    if not raw:
        return None
    mask = to_mask_list(info.get("gt_ego_fut_masks"), len(raw))

    if require_full_future_6:
        if len(raw) < FUTURE_STEPS:
            return None
        if any(mask[i] == 0 for i in range(FUTURE_STEPS)):
            return None
        selected = raw[1:FUTURE_STEPS + 1]
    else:
        valid: List[List[float]] = []
        for i, p in enumerate(raw[:FUTURE_STEPS]):
            if mask[i] == 1:
                valid.append([float(p[0]), float(p[1])])
        if not valid:
            return None
        while len(valid) < FUTURE_STEPS:
            valid.append(valid[-1])
        selected = valid[:FUTURE_STEPS]

    return {
        "future_forward_left": [[maybe_round(p[0]), maybe_round(p[1])] for p in selected],
        "future_right_front": forward_left_to_right_front(selected),
        "mask_first6": mask[:FUTURE_STEPS],
    }


def build_history_from_prev_chain(
    info: Dict[str, Any],
    token_to_info: Dict[str, Dict[str, Any]],
    history_steps: int,
) -> List[List[float]]:
    current_pose = try_extract_pose(info)
    if current_pose is None:
        return [[0.0, 0.0] for _ in range(history_steps)]
    current_t, current_q = current_pose

    points_forward_left: List[List[float]] = []
    cursor = info
    for _ in range(history_steps):
        prev_tok = str(cursor.get("prev", "")).strip()
        if not prev_tok:
            break
        prev_info = token_to_info.get(prev_tok)
        if prev_info is None:
            break
        prev_pose = try_extract_pose(prev_info)
        if prev_pose is None:
            break
        prev_t, _prev_q = prev_pose
        rel = global_to_current_ego(current_t, current_q, prev_t)
        points_forward_left.append([maybe_round(rel[0]), maybe_round(rel[1])])
        cursor = prev_info

    points_forward_left.reverse()  # oldest -> newest
    while len(points_forward_left) < history_steps:
        points_forward_left.insert(0, [0.0, 0.0])
    return forward_left_to_right_front(points_forward_left[:history_steps])


def compute_prompt_ego_fields(
    info: Dict[str, Any],
    token_to_info: Dict[str, Dict[str, Any]],
    wheelbase: float,
) -> Dict[str, Any]:
    # Use pose differencing for prompt scalars. This is deterministic and avoids
    # assuming undocumented gt_ego_lcf_feat index semantics.
    pose = try_extract_pose(info)
    if pose is None:
        return {
            "vx_vy_right_front": [0.0, 0.0],
            "ax_ay_right_front": [0.0, 0.0],
            "yaw_rate": 0.0,
            "speed": 0.0,
            "steering_deg": 0.0,
        }

    curr_t, curr_q = pose
    curr_yaw = yaw_from_quat_wxyz(curr_q)

    prev_tok = str(info.get("prev", "")).strip()
    prev = token_to_info.get(prev_tok) if prev_tok else None
    if prev is None or try_extract_pose(prev) is None:
        return {
            "vx_vy_right_front": [0.0, 0.0],
            "ax_ay_right_front": [0.0, 0.0],
            "yaw_rate": 0.0,
            "speed": 0.0,
            "steering_deg": 0.0,
        }

    prev_t, prev_q = try_extract_pose(prev)  # type: ignore[misc]
    prev_yaw = yaw_from_quat_wxyz(prev_q)
    t_curr = float(info.get("timestamp", 0.0))
    t_prev = float(prev.get("timestamp", 0.0))
    dt = max((t_curr - t_prev) / 1e6, 1e-3) if t_curr and t_prev else 0.5

    rel_prev = global_to_current_ego(curr_t, curr_q, prev_t)
    # rel_prev is where the previous ego origin lies in the current ego frame.
    # Velocity of the current ego in its own current frame is the negative of that displacement / dt.
    vx_fwd = -float(rel_prev[0]) / dt
    vy_left = -float(rel_prev[1]) / dt
    speed = math.hypot(vx_fwd, vy_left)
    yaw_rate = normalize_angle(curr_yaw - prev_yaw) / dt

    prevprev_tok = str(prev.get("prev", "")).strip()
    prevprev = token_to_info.get(prevprev_tok) if prevprev_tok else None
    if prevprev is not None and try_extract_pose(prevprev) is not None:
        prevprev_t, prevprev_q = try_extract_pose(prevprev)  # type: ignore[misc]
        prevprev_yaw = yaw_from_quat_wxyz(prevprev_q)
        t_prevprev = float(prevprev.get("timestamp", 0.0))
        dt_prev = max((t_prev - t_prevprev) / 1e6, 1e-3) if t_prev and t_prevprev else 0.5
        rel_prevprev = global_to_current_ego(prev_t, prev_q, prevprev_t)
        vx_prev_fwd = -float(rel_prevprev[0]) / dt_prev
        vy_prev_left = -float(rel_prevprev[1]) / dt_prev
        ax_fwd = (vx_fwd - vx_prev_fwd) / max(dt, 1e-3)
        ay_left = (vy_left - vy_prev_left) / max(dt, 1e-3)
    else:
        ax_fwd = 0.0
        ay_left = 0.0

    # convert scalar vectors from forward-left to right-front
    vx_vy_right_front = [maybe_round(-vy_left), maybe_round(vx_fwd)]
    ax_ay_right_front = [maybe_round(-ay_left), maybe_round(ax_fwd)]
    steering = 0.0 if abs(speed) < 1e-6 else math.degrees(math.atan2(wheelbase * yaw_rate, speed))

    return {
        "vx_vy_right_front": vx_vy_right_front,
        "ax_ay_right_front": ax_ay_right_front,
        "yaw_rate": maybe_round(yaw_rate),
        "speed": maybe_round(speed),
        "steering_deg": maybe_round(steering),
    }


def find_can_bus_payload(merged: Dict[str, Any]) -> Any:
    for key in ("can_bus", "gt_ego_lcf_feat", "lcf_feat"):
        if key in merged and merged[key] is not None:
            return merged[key]
    return "N/A"


def build_sample(
    temporal_info: Dict[str, Any],
    cache_info: Optional[Dict[str, Any]],
    token_to_info: Dict[str, Dict[str, Any]],
    cam_order: Sequence[str],
    wheelbase: float,
    require_full_future_6: bool,
) -> Optional[Dict[str, Any]]:
    merged = dict(temporal_info)
    if cache_info:
        merged.update(cache_info)

    token = get_token(merged)
    if not token:
        return None

    images = extract_images(merged, cam_order)
    if len(images) < len(cam_order):
        return None

    future = extract_future_traj_open_drive_vla_frame(merged, require_full_future_6=require_full_future_6)
    if future is None:
        return None

    command = onehot_to_command(merged.get("gt_ego_fut_cmd"))
    if command is None:
        command = infer_command_from_future_forward_left(future["future_forward_left"])

    history = build_history_from_prev_chain(merged, token_to_info=token_to_info, history_steps=HISTORY_STEPS)
    ego_prompt = compute_prompt_ego_fields(merged, token_to_info=token_to_info, wheelbase=wheelbase)
    can_bus = find_can_bus_payload(merged)

    user_prompt = USER_PROMPT_TEMPLATE.format(
        velocity=compact_json(ego_prompt["vx_vy_right_front"]),
        yaw_rate=f"{float(ego_prompt['yaw_rate']):.4f}",
        acceleration=compact_json(ego_prompt["ax_ay_right_front"]),
        can_bus=compact_json(can_bus) if not (isinstance(can_bus, str) and can_bus == "N/A") else "N/A",
        speed=f"{float(ego_prompt['speed']):.4f}",
        steering=f"{float(ego_prompt['steering_deg']):.4f}",
        history=format_points_for_prompt(history),
        command=command,
    )
    assistant_answer = format_traj_answer(future["future_right_front"])

    return {
        "token": token,
        "image": images,
        "messages": [
            {"role": "system", "content": OPEN_DRIVE_VLA_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_answer},
        ],
        "system_prompt": OPEN_DRIVE_VLA_SYSTEM_PROMPT,
        "user_prompt": user_prompt,
        "assistant_answer": assistant_answer,
        "navigation": command,
        "history_traj_right_front": history,
        "future_waypoints_right_front": future["future_right_front"],
        "future_waypoints_forward_left": future["future_forward_left"],
        "future_mask_first6": future["mask_first6"],
        "ego_prompt_fields": {
            "velocity_vx_vy_right_front": ego_prompt["vx_vy_right_front"],
            "acceleration_ax_ay_right_front": ego_prompt["ax_ay_right_front"],
            "yaw_rate": ego_prompt["yaw_rate"],
            "speed": ego_prompt["speed"],
            "steering_deg": ego_prompt["steering_deg"],
            "can_bus": as_list(can_bus),
        },
        # raw fields preserved for debugging / downstream reuse
        "raw": {
            "gt_ego_fut_cmd": as_list(merged.get("gt_ego_fut_cmd")),
            "gt_ego_fut_trajs": as_list(merged.get("gt_ego_fut_trajs")),
            "gt_ego_fut_masks": as_list(merged.get("gt_ego_fut_masks")),
            "gt_ego_fut_yaw": as_list(merged.get("gt_ego_fut_yaw")),
            "gt_ego_lcf_feat": as_list(merged.get("gt_ego_lcf_feat")),
            "gt_boxes": as_list(merged.get("gt_boxes")),
            "gt_names": as_list(merged.get("gt_names")),
            "gt_velocity": as_list(merged.get("gt_velocity")),
            "gt_agent_fut_trajs": as_list(merged.get("gt_agent_fut_trajs")),
            "gt_agent_fut_masks": as_list(merged.get("gt_agent_fut_masks")),
            "gt_agent_fut_yaw": as_list(merged.get("gt_agent_fut_yaw")),
            "gt_agent_lcf_feat": as_list(merged.get("gt_agent_lcf_feat")),
            "scene_token": merged.get("scene_token"),
            "timestamp": merged.get("timestamp"),
            "prev": merged.get("prev"),
            "next": merged.get("next"),
        },
        "alignment_notes": {
            "scene_track_map_tokens": "Kept as literal placeholders because the official projected visual tokens are not publicly reconstructable from the released data files.",
            "coordinate_system": "Prompts and answers are exported in OpenDriveVLA paper convention: x=right, y=front.",
            "ego_prompt_scalars": "Computed from ego pose differencing unless an explicit can_bus field is available; gt_ego_lcf_feat is preserved as raw can_bus payload only.",
        },
    }


# --------------------------------- main ---------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert OpenDriveVLA public train data into an aligned planning JSON.")
    parser.add_argument("--train-pkl", required=True, help="Path to nuscenes_infos_temporal_train.pkl or val.pkl")
    parser.add_argument("--cache-pkl", required=True, help="Path to cached_nuscenes_info.pkl")
    parser.add_argument("--out-json", required=True, help="Output JSON file")
    parser.add_argument(
        "--wheelbase",
        type=float,
        default=2.84,
        help="Wheelbase used for steering approximation when only pose history is available.",
    )
    parser.add_argument(
        "--allow-short-future",
        action="store_true",
        help="Allow fewer than 6 valid future points and pad by repeating the last valid point.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional cap for debugging. 0 means export all samples.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    temporal_obj = load_pickle(args.train_pkl)
    cache_obj = load_pickle(args.cache_pkl)
    temporal_items = extract_items(temporal_obj)
    cache_items = extract_items(cache_obj)

    temporal_by_token = index_by_token(temporal_items)
    cache_by_token = index_by_token(cache_items)

    # history construction should prefer merged records when possible
    merged_for_history: Dict[str, Dict[str, Any]] = {}
    for tok, item in temporal_by_token.items():
        merged = dict(item)
        if tok in cache_by_token:
            merged.update(cache_by_token[tok])
        merged_for_history[tok] = merged

    samples: List[Dict[str, Any]] = []
    matched_cache = 0
    for tok, temporal_info in temporal_by_token.items():
        cache_info = cache_by_token.get(tok)
        if cache_info is not None:
            matched_cache += 1
        sample = build_sample(
            temporal_info=temporal_info,
            cache_info=cache_info,
            token_to_info=merged_for_history,
            cam_order=CAM_ORDER,
            wheelbase=float(args.wheelbase),
            require_full_future_6=not bool(args.allow_short_future),
        )
        if sample is None:
            continue
        samples.append(sample)
        if args.max_samples > 0 and len(samples) >= int(args.max_samples):
            break

    payload = {
        "metadata": {
            "format": "opendrivevla_stage3_aligned_json_v1",
            "num_samples": len(samples),
            "source_files": {
                "train_pkl": str(args.train_pkl),
                "cache_pkl": str(args.cache_pkl),
            },
            "cam_order": CAM_ORDER,
            "future_steps": FUTURE_STEPS,
            "history_steps": HISTORY_STEPS,
            "require_full_future_6": not bool(args.allow_short_future),
            "matched_cache_tokens": matched_cache,
            "alignment_level": "best_effort_publicly_reconstructable",
            "officially_aligned_parts": [
                "public data sources (UniAD temporal infos + GPT-Driver style cached_nuscenes_info)",
                "OpenDriveVLA system prompt text",
                "OpenDriveVLA stage-3 user prompt template",
                "trajectory answer tag format <traj_start>...<traj_end>",
                "coordinate convention x=right, y=front",
            ],
            "approximated_parts": [
                "projected visual tokens <SCENE>/<TRACK>/<MAP> cannot be reconstructed from public files",
                "exact sample filtering used by the authors is not publicly released",
                "ego prompt scalars are pose-derived unless an explicit can_bus field exists",
            ],
        },
        "samples": samples,
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(payload), f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(samples)} samples -> {out_path}")


if __name__ == "__main__":
    main()
