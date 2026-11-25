import csv
import json
import os
import numpy as np

def _quantile_stats(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
            "p01": None,
            "p05": None,
            "p25": None,
            "p75": None,
            "p95": None,
            "p99": None,
        }

    return {
        "n": int(x.size),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "p01": float(np.percentile(x, 1)),
        "p05": float(np.percentile(x, 5)),
        "p25": float(np.percentile(x, 25)),
        "p75": float(np.percentile(x, 75)),
        "p95": float(np.percentile(x, 95)),
        "p99": float(np.percentile(x, 99)),
    }


def _print_stats(title: str, x: np.ndarray) -> None:
    s = _quantile_stats(x)
    print(f"\n[{title}]")
    print(f"  Sample count n = {s['n']}")
    if s["n"] == 0:
        return
    print(
        "  "
        + f"mean={s['mean']:.6g}, median={s['median']:.6g}, std={s['std']:.6g}, min={s['min']:.6g}, max={s['max']:.6g}"
    )
    print(
        "  "
        + f"p01={s['p01']:.6g}, p05={s['p05']:.6g}, p25={s['p25']:.6g}, p75={s['p75']:.6g}, p95={s['p95']:.6g}, p99={s['p99']:.6g}"
    )


def _mad_sigma(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return 0.0
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return float(1.4826 * mad)


def _fit_gmm_1d_two_components(
    x: np.ndarray,
    max_iter: int = 200,
    tol: float = 1e-6,
    seed: int = 42,
) -> dict:
    """Fit a two-component univariate Gaussian mixture by expectation maximisation.

    The dominant component characterises the positive residuals used for screening.
    Return mixture weights, means, standard deviations and log likelihood."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 30:
        raise ValueError("Too few samples to fit the Gaussian mixture model reliably.")

    q30, q70 = np.percentile(x, [30, 70])
    means = np.array([q30, q70], dtype=float)
    sig0 = max(_mad_sigma(x), 1e-6)
    sigmas = np.array([sig0, sig0], dtype=float)
    weights = np.array([0.5, 0.5], dtype=float)

    def norm_pdf(xx: np.ndarray, mu: float, sigma: float) -> np.ndarray:
        sigma = max(float(sigma), 1e-12)
        z = (xx - mu) / sigma
        return (1.0 / (np.sqrt(2.0 * np.pi) * sigma)) * np.exp(-0.5 * z * z)

    prev_ll = None
    for _ in range(max_iter):
        # E-step
        p0 = weights[0] * norm_pdf(x, means[0], sigmas[0])
        p1 = weights[1] * norm_pdf(x, means[1], sigmas[1])
        denom = p0 + p1 + 1e-18
        r0 = p0 / denom
        r1 = p1 / denom

        # M-step
        n0 = float(np.sum(r0))
        n1 = float(np.sum(r1))
        weights = np.array([n0, n1], dtype=float) / float(x.size)

        means = np.array(
            [float(np.sum(r0 * x) / (n0 + 1e-18)), float(np.sum(r1 * x) / (n1 + 1e-18))],
            dtype=float,
        )
        var0 = float(np.sum(r0 * (x - means[0]) ** 2) / (n0 + 1e-18))
        var1 = float(np.sum(r1 * (x - means[1]) ** 2) / (n1 + 1e-18))
        sigmas = np.array([np.sqrt(max(var0, 1e-12)), np.sqrt(max(var1, 1e-12))], dtype=float)

        sigmas = np.maximum(sigmas, 1e-6)
        weights = np.clip(weights, 1e-6, 1.0)
        weights = weights / np.sum(weights)

        # log-likelihood
        mix = weights[0] * norm_pdf(x, means[0], sigmas[0]) + weights[1] * norm_pdf(x, means[1], sigmas[1])
        ll = float(np.sum(np.log(mix + 1e-18)))

        if prev_ll is not None and abs(ll - prev_ll) < tol * (1.0 + abs(prev_ll)):
            prev_ll = ll
            break
        prev_ll = ll

    p0 = weights[0] * norm_pdf(x, means[0], sigmas[0])
    p1 = weights[1] * norm_pdf(x, means[1], sigmas[1])
    denom = p0 + p1 + 1e-18
    resp = np.stack([p0 / denom, p1 / denom], axis=1)

    return {
        "weights": weights,
        "means": means,
        "sigmas": sigmas,
        "loglik": prev_ll,
        "responsibilities": resp,
        "x": x,
    }


def _histogram_brief(x: np.ndarray, bins: int = 20) -> list[dict]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return []
    hist, edges = np.histogram(x, bins=bins)
    out = []
    for i in range(hist.size):
        out.append(
            {
                "bin_left": float(edges[i]),
                "bin_right": float(edges[i + 1]),
                "count": int(hist[i]),
            }
        )
    return out


def _print_histogram_brief(title: str, x: np.ndarray, bins: int = 20) -> None:
    rows = _histogram_brief(x, bins=bins)
    print(f"\n[{title} | Histogram bins={bins}]")
    if not rows:
        print("  (no data)")
        return
    for r in rows:
        print(f"  [{r['bin_left']:.6g}, {r['bin_right']:.6g}) : {r['count']}")


def main() -> None:
    required_columns = ["Total", "B01", "B02", "B03", "B04", "B05", "B06"]

    # Classify positive residuals by distance from the dominant component.
    k_high = 1.5
    k_mid = 3.0

    enable_plot = True

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_windows_csv = os.path.join(script_dir, "credibility_windows.csv")
    out_summary_json = os.path.join(script_dir, "credibility_summary.json")
    out_hist_png = os.path.join(script_dir, "distribution_histogram.png")

    file_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "Preprocess_Timeseries",
        "outputs",
        "preprocessed",
        "flow_5min_imputed.csv",
    )

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    print("Reading data from CSV...")

    def _open_csv(path: str):
        try:
            return open(path, "r", encoding="utf-8-sig", newline="")
        except UnicodeDecodeError:
            return open(path, "r", encoding="gbk", newline="")

    print("Calculating the closure residual (Total minus the branch sum) for each window...")
    included_indices: list[int] = []
    residuals: list[float] = []
    residuals_plot: list[float] = []

    k_sum_over_total: list[float] = []
    non_positive_entries: list[tuple] = []

    skipped = {
        "missing_any": [],
        "all_zero": [],
        "non_positive_residual": [],
    }

    total_rows = 0
    with _open_csv(file_path) as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("The CSV is missing its header.")
        missing_columns = [c for c in required_columns if c not in reader.fieldnames]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        for row_idx, row in enumerate(reader):
            total_rows += 1
            values: list[float] = []
            missing = False
            for c in required_columns:
                raw = row.get(c, "")
                if raw is None:
                    missing = True
                    break
                raw = str(raw).strip()
                if raw == "" or raw.lower() == "nan":
                    missing = True
                    break
                try:
                    v = float(raw)
                except ValueError:
                    missing = True
                    break
                if not np.isfinite(v):
                    missing = True
                    break
                values.append(v)

            if missing:
                skipped["missing_any"].append(int(row_idx))
                continue

            if all(v == 0.0 for v in values):
                skipped["all_zero"].append(int(row_idx))
                continue

            total = values[0]
            f_sum = sum(values[1:])
            r = total - f_sum

            residuals_plot.append(float(r))

            if r <= 0.0:
                _k = float(f_sum / total) if total > 0.0 else float("nan")
                non_positive_entries.append((int(row_idx), float(r), float(_k)))
                skipped["non_positive_residual"].append(int(row_idx))
                continue

            k = float(f_sum / total) if total > 0.0 else float("nan")
            included_indices.append(int(row_idx))
            residuals.append(float(r))
            k_sum_over_total.append(float(k))

    residuals_all = np.asarray(residuals, dtype=float)
    residuals_plot_all = np.asarray(residuals_plot, dtype=float)
    included_indices_arr = np.asarray(included_indices, dtype=int)
    non_positive_residuals = (
        np.asarray([e[1] for e in non_positive_entries], dtype=float)
        if non_positive_entries else np.empty(0, dtype=float)
    )
    k_ratio_arr = np.asarray(k_sum_over_total, dtype=float)

    print("\nExcluded windows:")
    print(f"  Incomplete windows: {len(skipped['missing_any'])}")
    print(f"  All required columns zero: {len(skipped['all_zero'])}")
    print(f"  Non-positive residual (Total at or below the branch sum): {len(skipped['non_positive_residual'])}")
    print(f"  Windows included in the residual summary: {residuals_all.size}")

    _print_stats("Residuals over all included windows", residuals_all)

    peak_mu = None
    peak_sigma = None
    gmm_info = None
    peak_method = None

    if residuals_all.size >= 30:
        try:
            gmm_info = _fit_gmm_1d_two_components(residuals_all)
            w = gmm_info["weights"]
            m = gmm_info["means"]
            s = gmm_info["sigmas"]
            peak_idx = int(np.argmax(w))
            peak_mu = float(m[peak_idx])
            peak_sigma = float(s[peak_idx])
            peak_method = "gmm2_em"
        except Exception as e:
            print(f"\nGMM fit failed. Falling back to the median absolute deviation: {e}")

    if peak_mu is None or peak_sigma is None or not np.isfinite(peak_mu) or not np.isfinite(peak_sigma) or peak_sigma <= 0:
        peak_mu = float(np.median(residuals_all)) if residuals_all.size else 0.0
        peak_sigma = float(max(_mad_sigma(residuals_all), 1e-6))
        peak_method = "median_mad_fallback"

    print("\nDominant component fitted to all included residuals:")
    print(f"  method={peak_method}, mu_peak={peak_mu:.6g}, sigma_peak={peak_sigma:.6g}")
    print(f"  Credibility thresholds: high |r-mu|<= {k_high}σ, mid {k_high}σ<|r-mu|<= {k_mid}σ, and low otherwise")

    dist = np.abs(residuals_all - peak_mu)
    high_mask = dist <= k_high * peak_sigma
    mid_mask = (dist > k_high * peak_sigma) & (dist <= k_mid * peak_sigma)
    low_mask = ~high_mask & ~mid_mask

    residuals_high = residuals_all[high_mask]
    residuals_mid = residuals_all[mid_mask]
    residuals_low_pos = residuals_all[low_mask]
    residuals_low = (
        np.concatenate([residuals_low_pos, non_positive_residuals])
        if non_positive_residuals.size > 0 else residuals_low_pos
    )

    _print_stats("High-credibility residual", residuals_high)
    _print_stats("Mid-credibility residual", residuals_mid)
    _print_stats("Low-credibility residual", residuals_low)

    weights = np.full(residuals_all.shape, 0.1, dtype=float)
    weights[mid_mask] = 0.5
    weights[high_mask] = 1.0

    credibility = np.full(residuals_all.shape, "low", dtype=object)
    credibility[mid_mask] = "mid"
    credibility[high_mask] = "high"

    with open(out_windows_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t_index", "residual", "credibility", "weight", "k_sum_over_total"])
        for ti, r, c, w, k in zip(
            included_indices_arr.tolist(),
            residuals_all.tolist(),
            credibility.tolist(),
            weights.tolist(),
            k_ratio_arr.tolist(),
        ):
            writer.writerow([
                int(ti),
                float(r),
                str(c),
                float(w),
                (float(k) if np.isfinite(k) else ""),
            ])
        for ti, r_val, k_val in non_positive_entries:
            writer.writerow([
                ti,
                r_val,
                "low",
                0.1,
                (float(k_val) if np.isfinite(k_val) else ""),
            ])
    print(f"\nWindow credibility saved to: {out_windows_csv}")

    summary = {
        "skipped_stats": {
            "missing_windows": len(skipped["missing_any"]),
            "all_zero_windows": len(skipped["all_zero"]),
            "non_positive_residual_windows": len(skipped["non_positive_residual"]),
            "used_samples": int(residuals_all.size)
        },
        "residual_all": _quantile_stats(residuals_all),
        "peak_estimation": {
            "method": peak_method,
            "mu_peak": float(peak_mu),
            "sigma_peak": float(peak_sigma),
            "k_high": float(k_high),
            "k_mid": float(k_mid)
        },
        "credibility_windows": {
            "high": _quantile_stats(residuals_high),
            "mid": _quantile_stats(residuals_mid),
            "low": _quantile_stats(residuals_low)
        }
    }

    with open(out_summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Summary saved to: {out_summary_json}")

    if enable_plot:
        try:
            import matplotlib
            import matplotlib.pyplot as plt

            matplotlib.rcParams["font.sans-serif"] = ["SimSun"]
            matplotlib.rcParams["axes.unicode_minus"] = False

            plt.figure(figsize=(12, 6))
            plt.hist(residuals_plot_all, bins=60, alpha=0.7, color="steelblue",
                     label=f"residual all (n={residuals_plot_all.size})")

            left_h = peak_mu - k_high * peak_sigma
            right_h = peak_mu + k_high * peak_sigma
            left_m = peak_mu - k_mid * peak_sigma
            right_m = peak_mu + k_mid * peak_sigma

            plt.axvline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.8, label="0")
            plt.axvline(peak_mu, color="darkorange", linewidth=2.0, alpha=0.9, label="mu_peak")
            plt.axvspan(left_h, right_h, color="green", alpha=0.15, label="high band")
            plt.axvspan(left_m, right_m, color="gold", alpha=0.12, label="mid band")

            plt.title("Residual distribution and dominant-component thresholds")
            plt.xlabel("Total - (B01 + B02 + B03 + B04 + B05 + B06)")
            plt.ylabel("Frequency")
            plt.grid(True, alpha=0.25)
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_hist_png)
            plt.close()
            print(f"Histogram saved to: {out_hist_png}")
        except Exception as e:
            print(f"Could not generate the optional histogram: {e}")

    print("\nCredibility summary:")
    print(f"  High-credibility windows available to train the model: {int(np.sum(high_mask))} / {int(residuals_all.size)}")

    pressure_head_file = os.path.join(
        os.path.dirname(file_path), "pressure_5min_head.csv"
    )
    out_press_mask_csv = os.path.join(script_dir, "press_mask.csv")

    if os.path.exists(pressure_head_file):
        try:
            import pandas as pd

            ph_df = pd.read_csv(pressure_head_file, index_col=0, parse_dates=True)
            press_mask_df = (~ph_df.isna()).astype(int)
            press_mask_df.to_csv(out_press_mask_csv, encoding="utf-8")

            total_steps = len(press_mask_df)
            coverage = {
                col: int(press_mask_df[col].sum()) for col in press_mask_df.columns
            }
            print(f"\nPressure-head mask saved to: {out_press_mask_csv}")
            print(f"  Total time steps: {total_steps}")
            for col, cnt in coverage.items():
                print(f"  {col}: {cnt}/{total_steps} valid ({cnt / total_steps:.1%})")

            summary["pressure_coverage"] = {
                "total_timesteps": total_steps,
                "conversion_factor": 101.9716,
                "press_head_file": pressure_head_file,
                "press_mask_file": out_press_mask_csv,
                "coverage_per_sensor": coverage,
            }
            with open(out_summary_json, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Warning: Could not create the optional pressure-head mask: {e}")
    else:
        print(
            f"\nWarning: Pressure-head file missing. Skipping mask generation: {pressure_head_file}"
        )


if __name__ == "__main__":
    main()
