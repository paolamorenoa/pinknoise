import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from scipy import optimize
from scipy.stats import norm

embeddings_folder = 
output_folder = os.path.join(embeddings_folder, "dfa_results")
file_pattern_suffix = "_embeddings.csv"

min_length_for_dfa = 100       # minimum series length
min_window = 4                 # minimum window size
n_scales = 20                  # number of log-spaced scales
max_window_fraction = 4        # maximum window = len(series) // max_window_fraction
detrend_order = 1              # linear
n_shuffle_surrogates = 500     # number of shuffles
loglog_linearity_threshold = 0.90    # threshold log-log

sns.set(style="whitegrid", font_scale=1.05)

def compute_fluctuations(cum_signal, nvals, order=1):
    fluctuations = []
    L = len(cum_signal)
    for n in nvals:
        # non-overlapping windows of size n
        segments = [cum_signal[i:i+n] for i in range(0, L, n) if len(cum_signal[i:i+n]) == n]

        # inverse (mitigate edge effects + stability)
        segments_inv = [np.flip(cum_signal)[i:i+n] for i in range(0, L, n) if len(cum_signal[i:i+n]) == n]
        segments.extend(segments_inv)

        if len(segments) == 0:
            fluctuations.append(np.nan)
            continue

        detrended_rms = []
        for seg in segments:
            # fitting of polynomial in each segment
            coeffs = np.polyfit(np.arange(n), seg, order)
            trend = np.polyval(coeffs, np.arange(n))
            detrended = seg - trend
            # RMS fluctuation after detrending
            detrended_rms.append(np.sqrt(np.mean(detrended**2)))

        fluctuations.append(np.mean(detrended_rms))
    return np.array(fluctuations)


def dfa(signal, min_window=4, n_scales=20, order=1, return_diagnostics=False):

    s = np.asarray(signal).astype(float)
    if len(s) < max(min_length_for_dfa, 2 * min_window):
        return np.nan, None, None, np.nan
    # integrate
    cum_sig = np.cumsum(s - np.mean(s))

    max_window = max(min_window + 1, len(cum_sig) // max(2, max_window_fraction))
    nvals = np.unique(np.logspace(np.log10(min_window), np.log10(max_window), num=n_scales).astype(int))
    nvals = nvals[nvals >= min_window]
    if len(nvals) < 3:
        return np.nan, None, None, np.nan

    # RMS fluctuations at each window scale
    fluctuations = compute_fluctuations(cum_sig, nvals, order=order)
    valid = ~np.isnan(fluctuations) & (fluctuations > 0)
    if valid.sum() < 2:
        return np.nan, nvals, fluctuations, np.nan

    nvals_v = nvals[valid]
    Fv = fluctuations[valid]

    # fit slope in log-log space
    if loglog_corr < loglog_linearity_threshold:
        return np.nan, nvals_v, Fv, loglog_corr

    alpha = np.polyfit(log_n, log_F, 1)[0]

        if return_diagnostics:
            return alpha, nvals_v, Fv, loglog_corr
        return alpha, nvals_v, Fv, loglog_corr

# shuffled nulls
def generate_scramble_nulls(series, n_iter=500):
    alphas = []
    for _ in range(n_iter):
        s = np.random.permutation(series)
        a, _, _, _ = dfa(s, min_window=min_window, n_scales=n_scales, order=detrend_order)
        alphas.append(a)
    return np.array(alphas)

# plots
def plot_diagnostics(participant, signal, nvals, fluctuations, alpha, loglog_corr, outdir, nulls=None):
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    plt.figure(figsize=(18, 12))

    # raw cosine similarity
    plt.subplot(2, 2, 1)
    plt.plot(signal, color='gray', lw=1)
    plt.xlabel("Token index")
    plt.ylabel("Cosine similarity")
    plt.title(f"{participant}: similarity time series")

    #linear
    plt.subplot(2, 2, 2)
    if nvals is not None and fluctuations is not None:
        plt.plot(nvals, fluctuations, 'o-', color='C0', linewidth=1.5)
        plt.xlabel("Window size (n)")
        plt.ylabel("Fluctuation F(n)")
        plt.title("Fluctuation function (linear scale)")
        try:
            coeffs = np.polyfit(np.log10(nvals), np.log10(fluctuations), 1)
            fit_line = 10 ** np.polyval(coeffs, np.log10(nvals))
            plt.plot(nvals, fit_line, 'r--', linewidth=1.2, label=f"power-law fit α={coeffs[0]:.2f}")
            plt.legend()
        except Exception:
            pass
    else:
        plt.text(0.5, 0.5, 'Insufficient data', ha='center')

    # log-log
    plt.subplot(2, 2, 3)
    if nvals is not None and fluctuations is not None:
        plt.plot(nvals, fluctuations, 'o-', color='C1', linewidth=1.5)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel("log(n)")
        plt.ylabel("log(F(n))")
        plt.title(f"DFA log–log (α={alpha:.3f}, r={loglog_corr:.2f})")
        try:
            coeffs = np.polyfit(np.log10(nvals), np.log10(fluctuations), 1)
            fit_line = 10 ** np.polyval(coeffs, np.log10(nvals))
            plt.plot(nvals, fit_line, 'r--', linewidth=1.2)
        except Exception:
            pass

    # null distribution
    plt.subplot(2, 2, 4)
    if nulls is not None and len(nulls) > 0:
        sns.kdeplot(nulls, fill=True, color='orange')
        plt.axvline(alpha, color='blue', linestyle='--', label=f"Participant α={alpha:.3f}")
        plt.xlabel("α (scaling exponent)")
        plt.ylabel("Density")
        plt.title("Shuffle α null distribution")
        plt.legend()
    else:
        plt.text(0.5, 0.5, 'No nulls', ha='center')

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{participant}_dfa_diag.png"), dpi=200)
    plt.close()

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

all_files = [f for f in os.listdir(embeddings_folder) if f.endswith(file_pattern_suffix)]

results = []
print(f"Found {len(all_files)} embedding files.")

for idx, fname in enumerate(all_files, start=1):
    try:
        path = os.path.join(embeddings_folder, fname)
        participant = os.path.splitext(fname)[0]
        print(f"[{idx}/{len(all_files)}] {participant}")

        df = pd.read_csv(path, header=0)

        # alt csv
        # embedding_cols = [c for c in df.columns if c.startswith("dim_")]

        # if len(embedding_cols) == 0:
        #     print(f"  Skipping {participant}: embedding columns not found.")
        #     continue
        # arr = df[embedding_cols].astype(float).values

        arr = df.iloc[:, 1:].values

        if arr.shape[1] > 1 and not np.issubdtype(arr[:, 0].dtype, np.number):
            arr = arr[:, 1:].astype(float)
        else:
            arr = arr.astype(float)

        # array cosine similarity continuous
        sims = [cosine_similarity(arr[i:i+1], arr[i-1:i])[0, 0] for i in range(1, arr.shape[0])]
        sims = np.array(sims)

        alpha, nvals, fluctuations, loglog_corr = dfa(
            sims,
            min_window=min_window,
            n_scales=n_scales,
            order=detrend_order,
            return_diagnostics=True
        )

        if np.isnan(alpha):
            print(f"  Skipped ppt: {participant}.")
            continue

        null_shuffle = generate_scramble_nulls(sims, n_iter=n_shuffle_surrogates)
        mean_null = np.nanmean(null_shuffle)
        std_null = np.nanstd(null_shuffle)

        z_score = (alpha - mean_null) / std_null if std_null > 0 else np.nan
        p_value = 2 * (1 - norm.cdf(abs(z_score))) if not np.isnan(z_score) else np.nan

        pnr = np.sum(null_shuffle < alpha) / max(1, len(null_shuffle))

        results.append({
            "participant": participant,
            "alpha": float(alpha),
            "loglog_r": float(loglog_corr),
            "n_points": len(sims),
            "mean_null": float(mean_null),
            "std_null": float(std_null),
            "z_score": float(z_score),
            "p_value": float(p_value),
            "pnr": float(pnr)
        })

        #individual plots
        plot_diagnostics(participant, sims, nvals, fluctuations, alpha, loglog_corr, output_folder, nulls=null_shuffle)

    except Exception as e:
        print(f"  errorr {fname}: {e}")

results_df = pd.DataFrame(results)
results_csv = os.path.join(output_folder, "dfa_summary.csv")
results_df.to_csv(results_csv, index=False)
print(f"\n Saved to {results_csv}")

# group-level plots
# if not results_df.empty and results_df["alpha"].notna().any():
#     plt.figure(figsize=(10, 6))
#     sns.kdeplot(data=results_df, x="alpha", fill=True, color="blue", alpha=0.3, linewidth=2,
#                 label=f"Real α (mean={results_df['alpha'].mean():.2f})")
#     sns.rugplot(
#     data=results_df, x="alpha",
#     color="blue", height=0.05, alpha=0.4
# )
#     sns.kdeplot(data=results_df, x="mean_null", fill=True, color="gray", alpha=0.4, linewidth=2,
#                 label=f"Shuffled α (mean={results_df['mean_null'].mean():.2f})")
#     plt.axvline(results_df["alpha"].mean(), color="blue", linestyle="--", linewidth=2)
#     plt.axvline(results_df["mean_null"].mean(), color="gray", linestyle="--", linewidth=2)
#     plt.xlabel("DFA Scaling Coefficient (α)")
#     plt.ylabel("Density")
#     plt.title("Distribution of Real vs Shuffled DFA Scaling Coefficients")
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_folder, "dfa_real_vs_shuffled_distribution_kde.png"), dpi=200)
#     plt.close()
#     plt.figure(figsize=(10, 6))
#     sns.kdeplot(data=results_df, x="pnr", fill=True, color="orange", alpha=0.3, linewidth=2,
#                 label=f"PNR (mean={results_df['pnr'].mean():.2f})")
#     plt.axvline(results_df["pnr"].mean(), color="orange", linestyle="--", linewidth=2)
#     plt.axvline(0.95, color="red", linestyle="--", linewidth=1.5, label="PNR=0.95 threshold")
#     plt.xlabel("Proportion of null α smaller than real α (PNR)")
#     plt.ylabel("Density")
#     plt.title("Distribution of PNR Across Participants")
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_folder, "pnr_distribution_kde.png"), dpi=200)
#     plt.close()

