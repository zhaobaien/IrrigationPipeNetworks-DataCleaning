from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path


def _load_high_credibility_indices(credibility_csv: Path) -> list[int]:
    """Read the indices of high-credibility time steps."""
    indices: list[int] = []
    with open(credibility_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("credibility", "")).strip() == "high":
                try:
                    indices.append(int(row["t_index"]))
                except (KeyError, ValueError):
                    pass
    return indices


def main() -> None:
    here = Path(__file__).resolve()
    pi_gat_dir = here.parent.parent  # .../PI-GAT
    repo_root = pi_gat_dir.parent

    if str(pi_gat_dir) not in sys.path:
        sys.path.insert(0, str(pi_gat_dir))

    # Imports that work both when run as a script and when imported as a package.
    try:
        from opening_mapping.eom import (
            learn_effective_opening_mapping_from_flow,
            save_mapping_json,
            compare_eom_full_vs_high,
        )
        from opening_mapping.eom import _parse_segments as parse_segments
    except Exception:  # pragma: no cover
        from eom import (
            learn_effective_opening_mapping_from_flow,
            save_mapping_json,
            compare_eom_full_vs_high,
        )
        from eom import _parse_segments as parse_segments

    default_credibility_windows = str(
        pi_gat_dir / "preprocessing" / "credibility_windows.csv"
    )

    parser = argparse.ArgumentParser(
        description=(
            "Build per-valve effective-opening mappings (OD->OD_eff) from flow_5min_imputed.csv. "
            "Default outputs: eom_full.json (all windows), eom_high.json (high credibility) and eom_high_vs_full.json (comparison)."
        )
    )
    parser.add_argument(
        "--flow",
        type=str,
        default=str(repo_root / "Preprocess_Timeseries" / "outputs" / "preprocessed" / "flow_5min_imputed.csv"),
        help="Path to flow_5min_imputed.csv.",
    )
    parser.add_argument(
        "--valves",
        type=str,
        default="B01,B02,B03,B04,B05,B06",
        help="Comma-separated valves to build mappings for.",
    )
    parser.add_argument(
        "--segments",
        type=str,
        default="0-20,21-40,41-60,61-80,81-100",
        help="Segments for robust OD->Q curve learning.",
    )
    parser.add_argument(
        "--cap_quantile",
        type=float,
        default=0.95,
        help="Quantile used for segment caps when learning OD->Q.",
    )
    parser.add_argument(
        "--min_samples_per_segment",
        type=int,
        default=30,
        help="Minimum samples per segment.",
    )
    parser.add_argument(
        "--delta_quantile",
        type=float,
        default=0.8,
        help="Quantile of positive (implied_OD-OD) used as shift in each segment.",
    )
    parser.add_argument(
        "--max_shift_pct",
        type=float,
        default=60.0,
        help="Maximum allowed upward shift in percent points.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(here.parent / "eom_full.json"),
        help="Output JSON path for full-data EOM (eom_full.json).",
    )
    parser.add_argument(
        "--out_high",
        type=str,
        default=None,
        help="Output JSON path for high-credibility EOM (default: eom_high.json next to --out).",
    )
    parser.add_argument(
        "--credibility_windows",
        type=str,
        default=default_credibility_windows,
        help="Path to credibility_windows.csv (output of preprocess_imputed.py).",
    )
    parser.add_argument(
        "--out_comparison",
        type=str,
        default=None,
        help="Output JSON path for comparison (default: eom_high_vs_full.json next to --out).",
    )

    args = parser.parse_args()

    flow = Path(str(args.flow))
    if not flow.exists():
        raise FileNotFoundError(f"Flow CSV not found: {flow}")

    valves = [v.strip() for v in str(args.valves).split(",") if v.strip()]
    if not valves:
        raise ValueError("No valves specified")

    segments = None
    if str(args.segments).strip():
        segments = parse_segments(str(args.segments))

    out = Path(str(args.out))
    out_high = Path(
        args.out_high
        if args.out_high
        else str(out.parent / "eom_high.json")
    )
    out_comparison = Path(
        args.out_comparison
        if args.out_comparison
        else str(out.parent / "eom_high_vs_full.json")
    )

    cap_quantile = float(args.cap_quantile)
    min_samples_per_segment = int(args.min_samples_per_segment)
    delta_quantile = float(args.delta_quantile)
    max_shift_pct = float(args.max_shift_pct)

    print("=" * 80)
    print("[1/2] Fitting EOM using all valid windows ...")
    full_mapping = learn_effective_opening_mapping_from_flow(
        flow_csv=flow,
        valves=valves,
        segments=segments,
        cap_quantile=cap_quantile,
        min_samples_per_segment=min_samples_per_segment,
        delta_quantile=delta_quantile,
        max_shift_pct=max_shift_pct,
    )
    save_mapping_json(full_mapping, out)
    print(f"All-window EOM saved to: {out}")

    credibility_csv = Path(str(args.credibility_windows))
    high_mapping = None
    if not credibility_csv.exists():
        print(f"\nWarning: credibility_windows.csv missing. Skipping the high-credibility fit: {credibility_csv}")
    else:
        print(f"\n[2/2] Fitting EOM using high-credibility windows ...")
        high_indices = _load_high_credibility_indices(credibility_csv)
        if not high_indices:
            print("Warning: No high-credibility rows. Skipping the high-credibility fit.")
        else:
            print(f"Number of high-credibility windows: {len(high_indices)}")
            import numpy as np
            high_mapping = learn_effective_opening_mapping_from_flow(
                flow_csv=flow,
                valves=valves,
                segments=segments,
                cap_quantile=cap_quantile,
                min_samples_per_segment=min_samples_per_segment,
                delta_quantile=delta_quantile,
                max_shift_pct=max_shift_pct,
                row_indices=np.asarray(high_indices, dtype=int),
            )
            save_mapping_json(high_mapping, out_high)
            print(f"High-credibility EOM saved to: {out_high}")

    if high_mapping is not None:
        comparison = compare_eom_full_vs_high(full_mapping, high_mapping)
        save_mapping_json(comparison, out_comparison)
        print(f"\nComparison JSON saved to: {out_comparison}")

        print("\n" + "=" * 80)
        print("EOM curve comparison (all windows versus high credibility)")
        print(f"{'Valve':8s}  {'max_abs_delta':>14s}  {'mean_abs_delta':>15s}  {'agreement'}")
        print("-" * 56)
        for v in valves:
            if v not in comparison:
                continue
            interp = comparison[v]["interpretation"]
            print(
                f"  {v:6s}  {interp['max_abs_delta_pct']:>14.2f}%  "
                f"{interp['mean_abs_delta_pct']:>14.2f}%  "
                f"{interp['agreement']}"
            )
        print("=" * 80)
        print(
            "\nAgreement categories:\n"
            "  agreement=good     → The two curves closely agree\n"
            "  agreement=moderate → The curves differ. Inspect coverage in the upper OD range\n"
            "  agreement=poor     → The curves differ substantially. Prefer the high-credibility fit"
        )
        print(f"\nTip: See {out_comparison.name} for both curves and their differences by valve.")
        print(f"     Pass either {out.name} (all windows) or {out_high.name} (high credibility) to train_pi_gat.py via --effective_opening_mapping_json.")
    else:
        print("\nComparison skipped because no high-credibility data were available.")
        print(f"\nTip: pass {out} to train_pi_gat.py via --effective_opening_mapping_json")


if __name__ == "__main__":
    main()
