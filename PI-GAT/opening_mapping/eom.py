from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import numpy as np
import pandas as pd


X_KNOTS_PCT_DEFAULT = np.array([0.0, 20.0, 40.0, 60.0, 80.0, 100.0], dtype=float)


def pav_isotonic(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Pool-Adjacent-Violators (PAV) for isotonic regression (nondecreasing)."""
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    n = int(y.size)
    if n == 0:
        return y

    starts: List[int] = []
    ends: List[int] = []
    sum_w: List[float] = []
    sum_wy: List[float] = []

    for i in range(n):
        starts.append(i)
        ends.append(i + 1)
        sw = float(max(w[i], 1e-12))
        sum_w.append(sw)
        sum_wy.append(sw * float(y[i]))

        while len(starts) >= 2:
            k = len(starts) - 1
            mean_prev = sum_wy[k - 1] / sum_w[k - 1]
            mean_curr = sum_wy[k] / sum_w[k]
            if mean_prev <= mean_curr + 1e-12:
                break
            ends[k - 1] = ends[k]
            sum_w[k - 1] += sum_w[k]
            sum_wy[k - 1] += sum_wy[k]
            starts.pop()
            ends.pop()
            sum_w.pop()
            sum_wy.pop()

    y_hat = np.empty(n, dtype=float)
    for s, e, sw, swy in zip(starts, ends, sum_w, sum_wy):
        y_hat[s:e] = swy / sw
    return y_hat


def load_mapping_json(path: str | Path) -> Dict[str, dict]:
    p = Path(path)
    with open(p, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Invalid mapping json: {p}")
    return obj


def save_mapping_json(mapping: Dict[str, dict], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)


def _parse_segments(spec: str) -> List[Tuple[float, float]]:
    segs: List[Tuple[float, float]] = []
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" not in part:
            raise ValueError(f"Invalid segment '{part}', expected 'a-b'")
        a_str, b_str = part.split("-", 1)
        a = float(a_str.strip())
        b = float(b_str.strip())
        if not (np.isfinite(a) and np.isfinite(b) and b > a):
            raise ValueError(f"Invalid segment '{part}', require finite and b>a")
        segs.append((a, b))
    if not segs:
        raise ValueError("No valid segments")
    segs = sorted(segs, key=lambda t: (t[0], t[1]))
    if segs[0][0] > 0.0 + 1e-6 or segs[-1][1] < 100.0 - 1e-6:
        raise ValueError("Segments must cover 0..100")
    return segs


def _filter_valid_windows(df, required_columns: List[str]) -> pd.DataFrame:
    """Exclude records with missing required values or all required values equal to zero."""
    valid_rows = []
    for i in range(len(df)):
        row = df.iloc[i]
        has_missing = False
        all_zero = True
        for c in required_columns:
            if c not in df.columns:
                continue
            raw = row.get(c, "")
            if raw is None:
                has_missing = True
                break
            raw = str(raw).strip()
            if raw == "" or raw.lower() == "nan":
                has_missing = True
                break
            try:
                v = float(raw)
            except ValueError:
                has_missing = True
                break
            if not np.isfinite(v):
                has_missing = True
                break
            if v != 0:
                all_zero = False
        if not has_missing and not all_zero:
            valid_rows.append(i)
    return df.iloc[valid_rows]


def _build_simple_od_q_curve(
    df,
    od_col: str,
    q_col: str,
    segments: List[Tuple[float, float]],
    min_samples_per_segment: int = 30,
    cap_quantile: float = 0.95,
):
    """Simple OD->Q curve builder using quantile-based piecewise linear approximation."""
    class SimpleCurve:
        def __init__(self, od_points, q_points):
            self.od_points = np.asarray(od_points, dtype=float)
            self.q_points = np.asarray(q_points, dtype=float)

        def __call__(self, od):
            return np.interp(od, self.od_points, self.q_points, left=0.0, right=self.q_points[-1])

        def inv(self, q):
            return np.interp(q, self.q_points, self.od_points, left=0.0, right=self.od_points[-1])

    od = pd.to_numeric(df[od_col], errors="coerce").to_numpy(dtype=float)
    q = pd.to_numeric(df[q_col], errors="coerce").to_numpy(dtype=float)
    m = np.isfinite(od) & np.isfinite(q)
    od = od[m]
    q = q[m]

    od_points = []
    q_points = []

    for lo, hi in segments:
        if len(segments) == 1 or lo == 0.0:
            sel = (od >= lo) & (od <= hi)
        else:
            sel = (od > lo) & (od <= hi)

        if np.sum(sel) >= min_samples_per_segment:
            od_seg = od[sel]
            q_seg = q[sel]
            q_cap = np.quantile(q_seg, cap_quantile)
            od_median = np.median(od_seg[q_seg <= q_cap])
            q_median = np.median(q_seg[q_seg <= q_cap])
            od_points.append(od_median)
            q_points.append(q_median)

    if not od_points:
        od_points = [0.0, 50.0, 100.0]
        q_points = [0.0, np.median(q), np.quantile(q, 0.95)]

    od_points = sorted(od_points)
    q_points = [q for _, q in sorted(zip(od_points, q_points))]

    return SimpleCurve(od_points, q_points)


def learn_effective_opening_mapping_from_flow(
    *,
    flow_csv: str | Path,
    valves: List[str],
    segments: List[Tuple[float, float]] | None = None,
    cap_quantile: float = 0.95,
    min_samples_per_segment: int = 30,
    delta_quantile: float = 0.8,
    max_shift_pct: float = 60.0,
    x_knots_pct: np.ndarray | None = None,
    row_indices: Optional[np.ndarray] = None,
    df_override: Optional["pd.DataFrame"] = None,
    data_mode_override: Optional[str] = None,
) -> Dict[str, dict]:
    """Learn per-valve effective-opening mapping g_v(OD)->OD_eff from historical OD/Q.

    This does NOT change K(open). It only provides a monotone mapping to be applied
    before K(open) when computing energy loss.

    Parameters
    ----------
    row_indices : optional
        0-based integer positions in ``flow_csv``; when provided only those rows
        are used (high-credibility mode).
    df_override : optional
        Pre-processed DataFrame to use directly, bypassing CSV reading and
        filtering entirely.  ``row_indices`` is ignored when this is set.
    data_mode_override : optional
        String recorded in ``_meta.data_mode`` when ``df_override`` is used.
    """
    flow_csv = Path(flow_csv)

    if segments is None:
        segments = [(0.0, 20.0), (21.0, 40.0), (41.0, 60.0), (61.0, 80.0), (81.0, 100.0)]

    if x_knots_pct is None:
        x_knots_pct = X_KNOTS_PCT_DEFAULT
    x_knots_pct = np.asarray(x_knots_pct, dtype=float)

    if df_override is not None:
        df = df_override.reset_index(drop=True)
        data_mode = data_mode_override if data_mode_override else "df_override"
        print(f"Using the supplied DataFrame: {len(df)} rows (data_mode={data_mode})")
    else:
        if not flow_csv.exists():
            raise FileNotFoundError(f"flow CSV not found: {flow_csv}")

        needed = ["date"]
        for v in valves:
            needed.append(f"OD_{v}")
            needed.append(v)

        required_columns = ["Total", "B01", "B02", "B03", "B04", "B05", "B06"]

        df_head = pd.read_csv(flow_csv, nrows=1)
        all_needed = list(set(needed + required_columns))
        usecols = [c for c in all_needed if c in df_head.columns]
        df = pd.read_csv(flow_csv, usecols=usecols)

        if row_indices is not None:
            idx_arr = np.asarray(row_indices, dtype=int)
            idx_arr = idx_arr[(idx_arr >= 0) & (idx_arr < len(df))]
            df = df.iloc[idx_arr].reset_index(drop=True)
            data_mode = "high_credibility"
            print(f"High-credibility subset: selected {len(df)} rows (from {len(row_indices)} input indices)")
        else:
            df = _filter_valid_windows(df, required_columns)
            data_mode = "full"
            print(f"Number of valid windows after filtering: {len(df)}")

    mapping: Dict[str, dict] = {
        "_meta": {
            "flow_csv": str(flow_csv),
            "data_mode": data_mode,
            "n_rows_used": int(len(df)),
            "segments": [list(s) for s in segments],
            "cap_quantile": float(cap_quantile),
            "min_samples_per_segment": int(min_samples_per_segment),
            "delta_quantile": float(delta_quantile),
            "max_shift_pct": float(max_shift_pct),
            "x_knots_pct": [float(x) for x in x_knots_pct.tolist()],
        }
    }

    for v in valves:
        od_col = f"OD_{v}"
        q_col = v
        if od_col not in df.columns or q_col not in df.columns:
            continue

        curve = _build_simple_od_q_curve(
            df,
            od_col,
            q_col,
            segments=segments,
            min_samples_per_segment=int(min_samples_per_segment),
            cap_quantile=float(cap_quantile),
        )

        od = pd.to_numeric(df[od_col], errors="coerce").to_numpy(dtype=float)
        q = pd.to_numeric(df[q_col], errors="coerce").to_numpy(dtype=float)
        m = np.isfinite(od) & np.isfinite(q)
        od = np.clip(od[m], 0.0, 100.0)
        q = np.clip(q[m], 0.0, None)

        inv = getattr(curve, "inv")
        implied = np.array([float(inv(float(qq))) if (np.isfinite(qq) and qq > 0) else 0.0 for qq in q], dtype=float)
        implied = np.clip(implied, 0.0, 100.0)

        delta = implied - od
        delta = np.where(np.isfinite(delta), delta, 0.0)
        delta_pos = np.clip(delta, 0.0, float(max_shift_pct))

        n_seg: Dict[str, int] = {}
        shift_at_hi: List[float] = [0.0]  # for x=0
        weights: List[float] = [1000.0]
        for i, (lo, hi) in enumerate(segments):
            if i == 0:
                sel = (od >= lo) & (od <= hi) & (q > 0.0)
            else:
                sel = (od > lo) & (od <= hi) & (q > 0.0)
            n_sel = int(np.sum(sel))
            n_seg[f"{lo:.0f}-{hi:.0f}"] = n_sel
            if n_sel < 10:
                sh = 0.0
                w = 1.0
            else:
                sh = float(np.quantile(delta_pos[sel], float(delta_quantile)))
                w = float(np.sum(sel))
            shift_at_hi.append(sh)
            weights.append(w)

        if len(shift_at_hi) != len(x_knots_pct):
            xp = np.linspace(0.0, 100.0, num=len(shift_at_hi), dtype=float)
            shift_knot = np.interp(x_knots_pct, xp, np.asarray(shift_at_hi, dtype=float))
        else:
            shift_knot = np.asarray(shift_at_hi, dtype=float)

        y_knots = x_knots_pct + shift_knot
        y_knots[0] = 0.0
        y_knots[-1] = 100.0

        y_knots = np.maximum(y_knots, x_knots_pct)
        w = np.asarray(weights[: len(x_knots_pct)], dtype=float)
        if w.size != x_knots_pct.size:
            w = np.ones_like(x_knots_pct)
        y_iso = pav_isotonic(y_knots, w)

        y_iso = np.clip(y_iso, 0.0, 100.0)
        y_iso[0] = 0.0
        y_iso[-1] = 100.0

        mapping[v] = {
            "x_knots_pct": [float(x) for x in x_knots_pct.tolist()],
            "y_knots_pct": [float(y) for y in y_iso.tolist()],
            "n_samples": int(len(od)),
            "n_samples_per_segment": n_seg,
        }

    return mapping


def compare_eom_full_vs_high(
    full_mapping: Dict[str, dict],
    high_mapping: Dict[str, dict],
) -> Dict[str, dict]:
    """Compare per-valve EOM curves from full data vs high-credibility subset.

    For each valve the result contains both curves side-by-side, a delta
    (high - full) at each x-knot, and a simple agreement label:
      'good'     : max_abs_delta < 5 pct-pts  — curves basically agree
      'moderate' : 5 <= max_abs_delta < 15     — noticeable but bounded shift
      'poor'     : max_abs_delta >= 15         — significant divergence
    """
    comparison: Dict[str, dict] = {
        "_meta": {
            "full_meta": full_mapping.get("_meta", {}),
            "high_meta": high_mapping.get("_meta", {}),
        }
    }
    valves = [k for k in full_mapping if not k.startswith("_")]
    for v in valves:
        if v not in high_mapping:
            continue
        fe = full_mapping[v]
        he = high_mapping[v]
        x = fe.get("x_knots_pct", [])
        fy = fe.get("y_knots_pct", [])
        hy = he.get("y_knots_pct", [])
        if not fy or len(fy) != len(hy):
            continue
        delta = [round(h - f, 6) for h, f in zip(hy, fy)]
        max_abs = max(abs(d) for d in delta)
        mean_abs = sum(abs(d) for d in delta) / len(delta)
        if max_abs < 5.0:
            agreement = "good"
        elif max_abs < 15.0:
            agreement = "moderate"
        else:
            agreement = "poor"
        comparison[v] = {
            "x_knots_pct": x,
            "full": {
                "y_knots_pct": fy,
                "n_samples": fe.get("n_samples"),
                "n_samples_per_segment": fe.get("n_samples_per_segment", {}),
            },
            "high": {
                "y_knots_pct": hy,
                "n_samples": he.get("n_samples"),
                "n_samples_per_segment": he.get("n_samples_per_segment", {}),
            },
            "delta_y_knots_pct": delta,
            "interpretation": {
                "max_abs_delta_pct": round(max_abs, 4),
                "mean_abs_delta_pct": round(mean_abs, 4),
                "agreement": agreement,
            },
        }
    return comparison


def compare_eom_three_way(
    full_mapping: Dict[str, dict],
    high_mapping: Dict[str, dict],
    debiased_mapping: Dict[str, dict],
) -> Dict[str, dict]:
    """Three-way comparison of EOM curves: full / high-credibility / debiased-full.

    For each valve the result contains all three y_knots curves, two delta arrays
    (high-full, debiased-full) and a per-curve agreement label using the same
    thresholds as compare_eom_full_vs_high.
    """

    def _label(max_abs: float) -> str:
        if max_abs < 5.0:
            return "good"
        elif max_abs < 15.0:
            return "moderate"
        return "poor"

    comparison: Dict[str, dict] = {
        "_meta": {
            "full_meta": full_mapping.get("_meta", {}),
            "high_meta": high_mapping.get("_meta", {}),
            "debiased_meta": debiased_mapping.get("_meta", {}),
        }
    }
    valves = [k for k in full_mapping if not k.startswith("_")]
    for v in valves:
        if v not in high_mapping or v not in debiased_mapping:
            continue
        fe = full_mapping[v]
        he = high_mapping[v]
        de = debiased_mapping[v]
        x = fe.get("x_knots_pct", [])
        fy = fe.get("y_knots_pct", [])
        hy = he.get("y_knots_pct", [])
        dy = de.get("y_knots_pct", [])
        if not fy or len(fy) != len(hy) or len(fy) != len(dy):
            continue

        delta_high_vs_full = [round(h - f, 6) for h, f in zip(hy, fy)]
        delta_debiased_vs_full = [round(d - f, 6) for d, f in zip(dy, fy)]

        max_abs_hf = max(abs(d) for d in delta_high_vs_full)
        max_abs_df = max(abs(d) for d in delta_debiased_vs_full)
        mean_abs_hf = sum(abs(d) for d in delta_high_vs_full) / len(delta_high_vs_full)
        mean_abs_df = sum(abs(d) for d in delta_debiased_vs_full) / len(delta_debiased_vs_full)

        comparison[v] = {
            "x_knots_pct": x,
            "full": {
                "y_knots_pct": fy,
                "n_samples": fe.get("n_samples"),
                "n_samples_per_segment": fe.get("n_samples_per_segment", {}),
            },
            "high": {
                "y_knots_pct": hy,
                "n_samples": he.get("n_samples"),
                "n_samples_per_segment": he.get("n_samples_per_segment", {}),
            },
            "debiased": {
                "y_knots_pct": dy,
                "n_samples": de.get("n_samples"),
                "n_samples_per_segment": de.get("n_samples_per_segment", {}),
            },
            "delta_high_vs_full": delta_high_vs_full,
            "delta_debiased_vs_full": delta_debiased_vs_full,
            "interpretation": {
                "high_vs_full": {
                    "max_abs_delta_pct": round(max_abs_hf, 4),
                    "mean_abs_delta_pct": round(mean_abs_hf, 4),
                    "agreement": _label(max_abs_hf),
                },
                "debiased_vs_full": {
                    "max_abs_delta_pct": round(max_abs_df, 4),
                    "mean_abs_delta_pct": round(mean_abs_df, 4),
                    "agreement": _label(max_abs_df),
                },
            },
        }
    return comparison


def build_edge_y_knots01(
    *,
    sensor2edge_idx: Dict[str, int],
    num_edges: int,
    mapping_json: Dict[str, dict],
    x_knots_pct: Optional[Iterable[float]] = None,
):
    """Build per-edge y_knots in [0,1] for torch-side mapping."""
    import torch

    if x_knots_pct is None:
        x_knots_pct = mapping_json.get("_meta", {}).get("x_knots_pct", X_KNOTS_PCT_DEFAULT.tolist())
    assert x_knots_pct is not None
    x_knots_pct = np.asarray(list(x_knots_pct), dtype=float)
    x_knots01 = torch.tensor(np.divide(x_knots_pct, 100.0), dtype=torch.float32)

    y = x_knots01.view(1, -1).repeat(int(num_edges), 1).clone()

    for valve, cfg in mapping_json.items():
        if valve.startswith("_"):
            continue
        if valve not in sensor2edge_idx:
            continue
        ei = int(sensor2edge_idx[valve])
        if ei < 0 or ei >= int(num_edges):
            continue
        y_knots_pct = cfg.get("y_knots_pct", None)
        if not isinstance(y_knots_pct, list) or len(y_knots_pct) != int(x_knots01.numel()):
            continue
        y_knots01 = torch.tensor([float(v) / 100.0 for v in y_knots_pct], dtype=torch.float32)
        y[ei, :] = torch.clamp(y_knots01, 0.0, 1.0)

    return x_knots01, y
