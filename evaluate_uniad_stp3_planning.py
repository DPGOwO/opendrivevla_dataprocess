#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import ast
import json
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from skimage.draw import polygon


FUTURE_STEPS = 6
X_BOUND = (-50.0, 50.0, 0.5)
Y_BOUND = (-50.0, 50.0, 0.5)
Z_BOUND = (-10.0, 10.0, 20.0)


def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.tensor([row[2] for row in [xbound, ybound, zbound]], dtype=torch.float32)
    bx = torch.tensor([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]], dtype=torch.float32)
    nx = torch.tensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]], dtype=torch.long)
    return dx, bx, nx


def calculate_birds_eye_view_parameters(x_bounds, y_bounds, z_bounds):
    bev_resolution = torch.tensor([row[2] for row in [x_bounds, y_bounds, z_bounds]], dtype=torch.float32)
    bev_start_position = torch.tensor([row[0] + row[2] / 2.0 for row in [x_bounds, y_bounds, z_bounds]], dtype=torch.float32)
    bev_dimension = torch.tensor(
        [(row[1] - row[0]) / row[2] for row in [x_bounds, y_bounds, z_bounds]],
        dtype=torch.long,
    )
    return bev_resolution, bev_start_position, bev_dimension


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        try:
            return pickle.load(f)
        except Exception:
            f.seek(0)
            return pickle.load(f, encoding="latin1")


def to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def normalize_waypoints_6(points: Any, strict_6: bool = False) -> Optional[np.ndarray]:
    if not isinstance(points, (list, tuple)):
        return None

    out: List[List[float]] = []
    for row in points:
        if isinstance(row, (list, tuple)) and len(row) >= 2:
            out.append([float(row[0]), float(row[1])])

    if strict_6:
        if len(out) != FUTURE_STEPS:
            return None
        return np.asarray(out, dtype=np.float32)

    if len(out) >= FUTURE_STEPS:
        out = out[:FUTURE_STEPS]
    elif len(out) == 0:
        return None
    else:
        last = out[-1]
        while len(out) < FUTURE_STEPS:
            out.append([last[0], last[1]])

    return np.asarray(out, dtype=np.float32)


def extract_python_waypoints_from_text(text: str, strict_6: bool = False) -> Optional[np.ndarray]:
    text = text.strip()
    if not text:
        return None

    candidates = [text]
    m = re.search(r"\[[\s\S]*\]", text)
    if m:
        candidates.append(m.group(0))

    for cand in candidates:
        try:
            value = ast.literal_eval(cand)
        except Exception:
            continue
        wp = normalize_waypoints_6(value, strict_6=strict_6)
        if wp is not None:
            return wp

    pairs = re.findall(r"\(\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*\)", text)
    if pairs:
        value = [[float(a), float(b)] for a, b in pairs]
        return normalize_waypoints_6(value, strict_6=strict_6)

    return None


def extract_json_obj_from_text(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    if not text:
        return None
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return None

    try:
        obj = json.loads(match.group(0))
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None


def parse_prediction_points(value: Any, strict_6: bool = False) -> Optional[np.ndarray]:
    if isinstance(value, (list, tuple)):
        return normalize_waypoints_6(value, strict_6=strict_6)

    if isinstance(value, dict):
        for key in ("future_waypoints", "output", "pred", "answer", "prediction", "text"):
            if key in value:
                return parse_prediction_points(value[key], strict_6=strict_6)

    if isinstance(value, str):
        obj = extract_json_obj_from_text(value)
        if obj is not None:
            wp = parse_prediction_points(obj, strict_6=strict_6)
            if wp is not None:
                return wp
        return extract_python_waypoints_from_text(value, strict_6=strict_6)

    return None


def load_predictions(pred_json: str, strict_6: bool = False) -> Dict[str, np.ndarray]:
    data = load_json(pred_json)
    pred_by_token: Dict[str, np.ndarray] = {}

    if isinstance(data, dict):
        if isinstance(data.get("predictions"), list):
            data = data["predictions"]
        elif isinstance(data.get("results"), list):
            data = data["results"]
        else:
            for k, v in data.items():
                wp = parse_prediction_points(v, strict_6=strict_6)
                if wp is not None:
                    pred_by_token[str(k)] = wp
            return pred_by_token

    if not isinstance(data, list):
        raise ValueError("--pred-json must be a JSON list or dict.")

    for item in data:
        if not isinstance(item, dict):
            continue

        token = None
        for key in ("sample_token", "token", "id"):
            if key in item and str(item[key]).strip():
                token = str(item[key]).strip()
                break

        if token is None:
            continue

        wp = None
        for key in ("output", "future_waypoints", "pred", "answer", "prediction", "text"):
            if key in item:
                wp = parse_prediction_points(item[key], strict_6=strict_6)
                if wp is not None:
                    break

        if wp is None and isinstance(item.get("conversations"), list):
            for msg in reversed(item["conversations"]):
                if isinstance(msg, dict) and str(msg.get("from", "")).lower() in ("gpt", "assistant"):
                    wp = parse_prediction_points(msg.get("value"), strict_6=strict_6)
                    if wp is not None:
                        break

        if wp is not None:
            pred_by_token[token] = wp

    return pred_by_token


def resolve_gt_cache_files(gt_cache_dir: Path, only_vehicle: bool) -> Dict[str, Path]:
    gt_traj_path = gt_cache_dir / "gt_traj.pkl"
    gt_mask_path = gt_cache_dir / "gt_traj_mask.pkl"

    if only_vehicle:
        occ_candidates = [
            gt_cache_dir / "planing_gt_segmentation_val",
            gt_cache_dir / "planing_gt_segmentation_val.pkl",
        ]
    else:
        occ_candidates = [gt_cache_dir / "vad_gt_seg.pkl"]

    occ_path = None
    for p in occ_candidates:
        if p.exists():
            occ_path = p
            break

    if not gt_traj_path.exists():
        raise FileNotFoundError(f"Missing GT trajectory cache: {gt_traj_path}")
    if not gt_mask_path.exists():
        raise FileNotFoundError(f"Missing GT trajectory mask cache: {gt_mask_path}")
    if occ_path is None:
        raise FileNotFoundError(f"Missing occupancy GT cache in {gt_cache_dir}")

    return {
        "gt_traj": gt_traj_path,
        "gt_mask": gt_mask_path,
        "gt_occ": occ_path,
    }


def normalize_gt_traj(traj: Any) -> np.ndarray:
    arr = to_numpy(traj).astype(np.float32)
    if arr.ndim == 2:
        arr = arr[None, :, :]
    if arr.ndim != 3 or arr.shape[-1] < 2:
        raise ValueError(f"Unexpected GT trajectory shape: {arr.shape}")
    return arr[:, :, :2]


def normalize_gt_mask(mask: Any) -> np.ndarray:
    arr = to_numpy(mask).astype(np.float32)
    if arr.ndim == 1:
        arr = arr[None, :, None]
    elif arr.ndim == 2:
        if arr.shape[-1] == 2:
            arr = arr[None, :, :]
        else:
            arr = arr[:, :, None]
    if arr.ndim != 3:
        raise ValueError(f"Unexpected GT mask shape: {arr.shape}")
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 2, axis=-1)
    return arr[:, :, :2]


def normalize_occ(occ: Any) -> np.ndarray:
    arr = to_numpy(occ).astype(np.float32)
    if arr.ndim == 3:
        arr = arr[None, :, :, :]
    if arr.ndim != 4:
        raise ValueError(f"Unexpected occupancy shape: {arr.shape}")
    return arr


class PlanningMetric:
    def __init__(self, n_future: int = FUTURE_STEPS):
        dx, bx, _ = gen_dx_bx(X_BOUND, Y_BOUND, Z_BOUND)
        dx, bx = dx[:2], bx[:2]
        _, _, bev_dimension = calculate_birds_eye_view_parameters(X_BOUND, Y_BOUND, Z_BOUND)

        self.dx = dx
        self.bx = bx
        self.bev_dimension = bev_dimension.numpy()
        self.W = 1.85
        self.H = 4.084
        self.n_future = n_future

        self.obj_col = torch.zeros(self.n_future, dtype=torch.float32)
        self.obj_box_col = torch.zeros(self.n_future, dtype=torch.float32)
        self.L2 = torch.zeros(self.n_future, dtype=torch.float32)
        self.total = 0

    def evaluate_single_coll(self, traj: torch.Tensor, segmentation: torch.Tensor) -> torch.Tensor:
        pts = np.array([
            [-self.H / 2.0 + 0.5,  self.W / 2.0],
            [ self.H / 2.0 + 0.5,  self.W / 2.0],
            [ self.H / 2.0 + 0.5, -self.W / 2.0],
            [-self.H / 2.0 + 0.5, -self.W / 2.0],
        ], dtype=np.float32)

        pts = (pts - self.bx.cpu().numpy()) / self.dx.cpu().numpy()
        pts[:, [0, 1]] = pts[:, [1, 0]]
        rr, cc = polygon(pts[:, 1], pts[:, 0])
        rc = np.concatenate([rr[:, None], cc[:, None]], axis=-1)

        n_future, _ = traj.shape
        trajs = traj.view(n_future, 1, 2).clone()
        trajs[:, :, [0, 1]] = trajs[:, :, [1, 0]]
        trajs = (trajs / self.dx).cpu().numpy() + rc

        r = np.clip(trajs[:, :, 0].astype(np.int32), 0, self.bev_dimension[0] - 1)
        c = np.clip(trajs[:, :, 1].astype(np.int32), 0, self.bev_dimension[1] - 1)

        collision = np.full(n_future, False)
        for t in range(n_future):
            rr_t = r[t]
            cc_t = c[t]
            valid = np.logical_and(
                np.logical_and(rr_t >= 0, rr_t < self.bev_dimension[0]),
                np.logical_and(cc_t >= 0, cc_t < self.bev_dimension[1]),
            )
            collision[t] = np.any(segmentation[t, rr_t[valid], cc_t[valid]].cpu().numpy())
        return torch.from_numpy(collision).to(device=traj.device)

    def evaluate_coll(
        self,
        trajs: torch.Tensor,
        gt_trajs: torch.Tensor,
        segmentation: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, n_future, _ = trajs.shape

        trajs = trajs * torch.tensor([-1, 1], device=trajs.device)
        gt_trajs = gt_trajs * torch.tensor([-1, 1], device=trajs.device)

        obj_coll_sum = torch.zeros(n_future, device=segmentation.device)
        obj_box_coll_sum = torch.zeros(n_future, device=segmentation.device)

        for i in range(B):
            gt_box_coll = self.evaluate_single_coll(gt_trajs[i], segmentation[i])

            xx, yy = trajs[i, :, 0], trajs[i, :, 1]
            yi = ((yy - self.bx[0]) / self.dx[0]).long()
            xi = ((xx - self.bx[1]) / self.dx[1]).long()

            m1 = torch.logical_and(
                torch.logical_and(yi >= 0, yi < self.bev_dimension[0]),
                torch.logical_and(xi >= 0, xi < self.bev_dimension[1]),
            )
            m1 = torch.logical_and(m1, torch.logical_not(gt_box_coll))

            ti = torch.arange(n_future, device=trajs.device)
            obj_coll_sum[ti[m1]] += segmentation[i, ti[m1], yi[m1], xi[m1]].long()

            box_coll = self.evaluate_single_coll(trajs[i], segmentation[i])
            m2 = torch.logical_not(gt_box_coll)
            obj_box_coll_sum[ti[m2]] += box_coll[ti[m2]].long()

        return obj_coll_sum, obj_box_coll_sum

    def compute_l2(
        self,
        trajs: torch.Tensor,
        gt_trajs: torch.Tensor,
        gt_trajs_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if gt_trajs_mask is None:
            gt_trajs_mask = torch.ones_like(gt_trajs)
        return torch.sqrt((((trajs[:, :, :2] - gt_trajs[:, :, :2]) ** 2) * gt_trajs_mask).sum(dim=-1))

    def update(
        self,
        trajs: np.ndarray,
        gt_trajs: np.ndarray,
        segmentation: np.ndarray,
        gt_trajs_mask: np.ndarray,
    ) -> None:
        trajs_t = torch.tensor(trajs, dtype=torch.float32)
        gt_t = torch.tensor(gt_trajs, dtype=torch.float32)
        seg_t = torch.tensor(segmentation, dtype=torch.float32)
        mask_t = torch.tensor(gt_trajs_mask, dtype=torch.float32)

        assert trajs_t.shape == gt_t.shape

        l2 = self.compute_l2(trajs_t, gt_t, mask_t)
        obj_coll_sum, obj_box_coll_sum = self.evaluate_coll(trajs_t[:, :, :2], gt_t[:, :, :2], seg_t)

        self.obj_col += obj_coll_sum.cpu()
        self.obj_box_col += obj_box_coll_sum.cpu()
        self.L2 += l2.sum(dim=0).cpu()
        self.total += len(trajs)

    def compute(self) -> Dict[str, np.ndarray]:
        if self.total == 0:
            raise RuntimeError("No valid samples were evaluated.")
        return {
            "obj_col": (self.obj_col / self.total).cpu().numpy(),
            "obj_box_col": (self.obj_box_col / self.total).cpu().numpy(),
            "L2": (self.L2 / self.total).cpu().numpy(),
        }


def load_gt_cache(gt_cache_dir: str, only_vehicle: bool) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    paths = resolve_gt_cache_files(Path(gt_cache_dir), only_vehicle=only_vehicle)

    gt_traj = load_pickle(paths["gt_traj"])
    gt_mask = load_pickle(paths["gt_mask"])
    gt_occ = load_pickle(paths["gt_occ"])

    for token, value in list(gt_occ.items()):
        occ = to_numpy(value)
        occ = np.flip(occ, axis=-1)
        if not only_vehicle:
            occ = np.flip(occ, axis=-2)
        gt_occ[token] = occ

    return gt_traj, gt_mask, gt_occ


def find_prediction_for_token(pred_by_token: Dict[str, np.ndarray], token: str) -> Optional[np.ndarray]:
    for candidate in (token, f"{token}_trajectory"):
        if candidate in pred_by_token:
            return pred_by_token[candidate]
    return None


def evaluate_opendrivevla(
    pred_by_token: Dict[str, np.ndarray],
    gt_trajs_dict: Dict[str, Any],
    gt_traj_mask_dict: Dict[str, Any],
    gt_occ_dict: Dict[str, Any],
) -> Dict[str, Any]:
    metric = PlanningMetric(FUTURE_STEPS)
    num_missing_pred = 0
    num_missing_gt_cache = 0

    for token, gt_traj_value in gt_trajs_dict.items():
        pred = find_prediction_for_token(pred_by_token, token)
        if pred is None:
            num_missing_pred += 1
            continue

        if token not in gt_traj_mask_dict or token not in gt_occ_dict:
            num_missing_gt_cache += 1
            continue

        gt_arr = normalize_gt_traj(gt_traj_value)
        mask_arr = normalize_gt_mask(gt_traj_mask_dict[token])
        occ_arr = normalize_occ(gt_occ_dict[token])

        pred_arr = pred.astype(np.float32)
        if pred_arr.ndim == 2:
            pred_arr = pred_arr[None, :, :2]
        elif pred_arr.ndim == 3:
            pred_arr = pred_arr[:, :, :2]
        else:
            raise ValueError(f"Unexpected prediction shape for token={token}: {pred_arr.shape}")

        if gt_arr.shape[1] % 2 == 1:
            gt_arr = gt_arr[:, 1:]
        if mask_arr.shape[1] % 2 == 1:
            mask_arr = mask_arr[:, 1:]
        if occ_arr.shape[1] % 2 == 1:
            occ_arr = occ_arr[:, 1:]
        if pred_arr.shape[1] % 2 == 1:
            pred_arr = pred_arr[:, 1:]

        gt_arr = gt_arr[:, :FUTURE_STEPS, :2]
        mask_arr = mask_arr[:, :FUTURE_STEPS, :2]
        occ_arr = occ_arr[:, :FUTURE_STEPS]
        pred_arr = pred_arr[:, :FUTURE_STEPS, :2]
        if pred_arr.shape != gt_arr.shape:
            if pred_arr.size == gt_arr.size:
                pred_arr = pred_arr.reshape(gt_arr.shape)
            else:
                raise ValueError(f"Prediction/GT shape mismatch for token={token}: {pred_arr.shape} vs {gt_arr.shape}")

        metric.update(pred_arr, gt_arr, occ_arr, mask_arr)

    scores = metric.compute()

    results: Dict[str, Any] = {
        "num_predictions_matched": int(metric.total),
        "num_missing_prediction": int(num_missing_pred),
        "num_missing_gt_cache": int(num_missing_gt_cache),
        "raw_scores": {
            "L2": scores["L2"].tolist(),
            "obj_col": scores["obj_col"].tolist(),
            "obj_box_col": scores["obj_box_col"].tolist(),
        },
    }

    for i in range(3):
        horizon = (i + 1) * 2
        results[f"plan_L2_{i+1}s"] = float(scores["L2"][:horizon].mean())
        results[f"plan_obj_col_{i+1}s"] = float(scores["obj_col"][:horizon].mean())
        results[f"plan_obj_box_col_{i+1}s"] = float(scores["obj_box_col"][:horizon].mean())

    results["plan_L2_avg"] = float(
        (results["plan_L2_1s"] + results["plan_L2_2s"] + results["plan_L2_3s"]) / 3.0
    )
    results["plan_obj_col_avg"] = float(
        (results["plan_obj_col_1s"] + results["plan_obj_col_2s"] + results["plan_obj_col_3s"]) / 3.0
    )
    results["plan_obj_box_col_avg"] = float(
        (results["plan_obj_box_col_1s"] + results["plan_obj_box_col_2s"] + results["plan_obj_box_col_3s"]) / 3.0
    )

    results["uniad"] = {
        "plan_L2_1s": float(scores["L2"][1]),
        "plan_L2_2s": float(scores["L2"][3]),
        "plan_L2_3s": float(scores["L2"][5]),
        "plan_L2_avg": float((scores["L2"][1] + scores["L2"][3] + scores["L2"][5]) / 3.0),
        "plan_obj_box_col_1s": float(scores["obj_box_col"][1]),
        "plan_obj_box_col_2s": float(scores["obj_box_col"][3]),
        "plan_obj_box_col_3s": float(scores["obj_box_col"][5]),
        "plan_obj_box_col_avg": float((scores["obj_box_col"][1] + scores["obj_box_col"][3] + scores["obj_box_col"][5]) / 3.0),
    }

    results["stp3"] = {
        "plan_L2_1s": results["plan_L2_1s"],
        "plan_L2_2s": results["plan_L2_2s"],
        "plan_L2_3s": results["plan_L2_3s"],
        "plan_L2_avg": results["plan_L2_avg"],
        "plan_obj_box_col_1s": results["plan_obj_box_col_1s"],
        "plan_obj_box_col_2s": results["plan_obj_box_col_2s"],
        "plan_obj_box_col_3s": results["plan_obj_box_col_3s"],
        "plan_obj_box_col_avg": results["plan_obj_box_col_avg"],
    }

    return results


def print_metrics(results: Dict[str, Any], only_vehicle: bool) -> None:
    print("===== OpenDriveVLA-aligned Planning Eval =====")
    print(f"only_vehicle: {only_vehicle}")
    print(f"num_predictions_matched: {results['num_predictions_matched']}")
    print(f"num_missing_prediction: {results['num_missing_prediction']}")
    print(f"num_missing_gt_cache: {results['num_missing_gt_cache']}")

    print("== UniAD evaluation ==")
    for key in (
        "plan_L2_1s",
        "plan_L2_2s",
        "plan_L2_3s",
        "plan_L2_avg",
        "plan_obj_box_col_1s",
        "plan_obj_box_col_2s",
        "plan_obj_box_col_3s",
        "plan_obj_box_col_avg",
    ):
        print(f"{key}: {results['uniad'][key]:.4f}")

    print("== STP-3 evaluation ==")
    for key in (
        "plan_L2_1s",
        "plan_L2_2s",
        "plan_L2_3s",
        "plan_L2_avg",
        "plan_obj_box_col_1s",
        "plan_obj_box_col_2s",
        "plan_obj_box_col_3s",
        "plan_obj_box_col_avg",
    ):
        print(f"{key}: {results['stp3'][key]:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate planning results by sample_token.")
    parser.add_argument("--pred-json", required=True, help="Prediction JSON path.")
    parser.add_argument("--gt-cache-dir", required=True, help="Directory containing gt_traj.pkl / gt_traj_mask.pkl / occupancy gt.")
    parser.add_argument(
        "--only-vehicle",
        dest="only_vehicle",
        action="store_true",
        default=True,
        help="Use vehicle-only occupancy (OpenDriveVLA / UniAD default).",
    )
    parser.add_argument(
        "--include-pedestrian",
        dest="only_vehicle",
        action="store_false",
        help="Use vehicle + pedestrian occupancy (STP-3 style).",
    )
    parser.add_argument("--strict-pred-6", action="store_true", help="Require predictions to contain exactly 6 future waypoints.")
    parser.add_argument("--save-json", default="", help="Optional metrics JSON output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pred_by_token = load_predictions(args.pred_json, strict_6=bool(args.strict_pred_6))
    gt_trajs_dict, gt_traj_mask_dict, gt_occ_dict = load_gt_cache(
        args.gt_cache_dir,
        only_vehicle=bool(args.only_vehicle),
    )

    results = evaluate_opendrivevla(
        pred_by_token=pred_by_token,
        gt_trajs_dict=gt_trajs_dict,
        gt_traj_mask_dict=gt_traj_mask_dict,
        gt_occ_dict=gt_occ_dict,
    )
    results["settings"] = {
        "pred_json": str(args.pred_json),
        "gt_cache_dir": str(args.gt_cache_dir),
        "only_vehicle": bool(args.only_vehicle),
        "strict_pred_6": bool(args.strict_pred_6),
    }

    print_metrics(results, only_vehicle=bool(args.only_vehicle))

    if args.save_json:
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Saved metrics: {out_path}")


if __name__ == "__main__":
    main()