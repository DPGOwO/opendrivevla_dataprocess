#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

token_file = "/cache/wx1427092/keyobj/key_object_token_val.json"
with open(token_file, 'r', encoding='utf-8') as f:
    key_sample_token = json.load(f)

marker_root = "/cache/wx1427092/nuscenes_val_pred_marker_dist+keyobj+plan_new/"
nuscenes_root = "/cache/wx1427092/nuScenes/"
# marker_root = ""
# nuscenes_root = ""

CAM_ORDER_TEXT = (
    "CAM_FRONT, CAM_FRONT_RIGHT, CAM_BACK_RIGHT, "
    "CAM_BACK, CAM_BACK_LEFT, CAM_FRONT_LEFT"
)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(obj: Any, path: str) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def format_float(x: Any) -> str:
    return f"{float(x):.2f}"


def format_points(points: Sequence[Sequence[float]]) -> str:
    buf: List[str] = []
    for p in points:
        if not isinstance(p, (list, tuple)) or len(p) < 2:
            continue
        buf.append(f"({format_float(p[0])},{format_float(p[1])})")
    return "[" + ",".join(buf) + "]"


def format_vector(x: Any) -> str:
    if x is None:
        return "N/A"
    if not isinstance(x, (list, tuple)):
        return "N/A"
    vals = []
    for v in x:
        try:
            vals.append(float(v))
        except Exception:
            return "N/A"
    return "[" + ",".join(format_float(v) for v in vals) + "]"


def build_prompt(sample: Dict[str, Any]) -> str:
    ego = sample.get("ego_prompt_fields", {}) or {}

    velocity = ego.get("velocity_vx_vy_right_front")
    acceleration = ego.get("acceleration_ax_ay_right_front")
    yaw_rate = ego.get("yaw_rate")
    speed = ego.get("speed")
    steering = ego.get("steering_deg")
    history = sample.get("history_traj_right_front", [])
    navigation = sample.get("navigation", "Go Straight")

    velocity_str = format_vector(velocity)
    acceleration_str = format_vector(acceleration)
    yaw_rate_str = format_float(yaw_rate) if yaw_rate is not None else "N/A"
    speed_str = format_float(speed) if speed is not None else "N/A"
    steering_str = format_float(steering) if steering is not None else "N/A"
    history_str = format_points(history)

    prompt = (
        
        f"I will present you with six images representing different camera perspectives "
        f"of an ego vehicle in order: {CAM_ORDER_TEXT}.\n"
        f"Generate a safe 3-second driving trajectory consisting of exactly 6 future "
        f"waypoints in ego-centric coordinates, where x points to the right and y points "
        f"to the front.\n"
        f"Ego states:\n"
        f" - Longitudinal velocity: {velocity[1]} m/s"
        f" - Lateral velocity: {velocity[0]} m/s"
        f" - Longitudinal acceleration: {acceleration[1]} m/s^2"
        f" - Lateral acceleration: {acceleration[0]} m/s^2"
        f" - Yaw rate: {yaw_rate} rad/s"
        f" - Vehicle length: 4.08 m"
        f" - Vehicle width: 1.85 m"
        f" - Speed magnitude: {speed} m/s"
        f" - Steering angle: {steering} degrees"
        f" - Navigation Information = {navigation}"
        f" Please predict exactly 6 future waypoints."
        f" If there are objects marked by red boxes, you need to consider their impact on the waypoints."
        f" Output only the waypoint list in this format:\n"
        f" [(x1,y1),(x2,y2),(x3,y3),(x4,y4),(x5,y5),(x6,y6)]"
    )
    return prompt


def build_answer(sample: Dict[str, Any]) -> str:
    points = sample.get("future_waypoints_forward_left", [])
    return format_points(points)


def build_record(sample: Dict[str, Any]) -> Dict[str, Any] | None:
    image = sample.get("image", [])
    token = sample.get("token") or sample.get("sample_token")
    
    if token in key_sample_token:
        image = [marker_root + i for i in image]
    else:
        image = [nuscenes_root + i for i in image]
    
    if not isinstance(image, list) or len(image) != 6:
        return None
    if not token:
        return None

    prompt = build_prompt(sample)
    answer = build_answer(sample)

    human_value = "<image>\n<image>\n<image>\n<image>\n<image>\n<image>\n" + prompt

    return {
        "image": image,
        "conversations": [
            {
                "from": "human",
                "value": human_value,
            },
            {
                "from": "gpt",
                "value": answer,
            },
        ],
        "sample_token": token,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-json",
        required=True,
        help="Path to opendrivevla_stage3_aligned_train.json",
    )
    parser.add_argument(
        "--output-json",
        required=True,
        help="Path to output VQA json",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional max number of samples to export. 0 means all.",
    )
    args = parser.parse_args()

    data = load_json(args.input_json)

    if isinstance(data, dict) and "samples" in data:
        samples = data["samples"]
    elif isinstance(data, list):
        samples = data
    else:
        raise ValueError("Unsupported input JSON structure.")

    out: List[Dict[str, Any]] = []
    skipped = 0

    for sample in samples:
        rec = build_record(sample)
        if rec is None:
            skipped += 1
            continue
        out.append(rec)
        if args.max_samples > 0 and len(out) >= args.max_samples:
            break

    dump_json(out, args.output_json)
    print(f"Wrote {len(out)} samples to {args.output_json}, skipped {skipped} invalid samples.")


if __name__ == "__main__":
    main()