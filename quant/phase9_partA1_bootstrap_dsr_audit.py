"""
Phase 9, Part A step 1 (highest priority): Bootstrap / Monte Carlo significance
test + Deflated Sharpe Ratio (DSR, Bailey & Lopez de Prado 2014) + Probability
of Backtest Overfitting (PBO) audit of Sleeve M4 -- the ONE live, certified,
non-crypto profitable sleeve the entire 21+-sleeve project currently rests on.

This closes the single most important acknowledged gap from Phase 5
("Deflated Sharpe Ratio / PBO / combinatorial purged CV not formally
computed... multiple-testing correction has not been formally applied to
the project's overall sleeve count") and Phase 7's "section 6" coverage
table. If M4 does not survive this audit, nothing downstream matters.

Three independent checks:
  1. IID bootstrap of the 47 trade-level R-multiples -> confidence interval
     on the trade-level Sharpe-like statistic (mean(R)/std(R)*sqrt(n)) and a
     one-sided p-value for H0: true mean R <= 0.
  2. Stationary block bootstrap (quarter-blocks) as a robustness check in
     case trade outcomes cluster within a quarter (same market regime).
  3. Deflated Sharpe Ratio: corrects the single-trial Sharpe for (a) the
     N=49 total sleeve/variant configurations this entire project has ever
     run (the honest multiple-testing universe -- not just the 6 universe
     variants M/M2-M6), and (b) the non-normality (skew/kurtosis) of the
     actual trade returns, using the full project's own observed Sharpe
     distribution as the empirical null.
"""
from __future__ import annotations
import json
import numpy as np
from scipy import stats as sps

RNG = np.random.default_rng(42)
N_BOOT = 20000


def load_trades(path):
    recs = json.load(open(path))
    r = np.array([t["r_multiple"] for t in recs], dtype=float)
    pnl = np.array([t["net_pnl"] for t in recs], dtype=float)
    qtr = np.array([t["exit_time"][:7] for t in recs])  # YYYY-MM as a coarse quarter-ish block key
    return r, pnl, qtr


def sharpe_like(r):
    n = len(r)
    if n < 2:
        return 0.0
    return (r.mean() / (r.std(ddof=1) + 1e-9)) * np.sqrt(n)


def iid_bootstrap(r, n_boot=N_BOOT):
    n = len(r)
    idx = RNG.integers(0, n, size=(n_boot, n))
    samples = r[idx]
    sharpes = samples.mean(axis=1) / (samples.std(axis=1, ddof=1) + 1e-9) * np.sqrt(n)
    means = samples.mean(axis=1)
    return sharpes, means


def block_bootstrap_by_quarter(r, qtr, n_boot=N_BOOT):
    """Resample whole year-month blocks (with replacement) to preserve any
    within-block (same-regime) correlation of trade outcomes, then flatten."""
    blocks = {}
    for k, v in zip(qtr, r):
        blocks.setdefault(k, []).append(v)
    block_list = list(blocks.values())
    nblocks = len(block_list)
    sharpes = []
    means = []
    for _ in range(n_boot):
        chosen = [block_list[i] for i in RNG.integers(0, nblocks, size=nblocks)]
        flat = np.concatenate(chosen)
        sharpes.append(sharpe_like(flat))
        means.append(flat.mean())
    return np.array(sharpes), np.array(means)


def permutation_sign_flip_test(r, n_perm=N_BOOT):
    """H0: the sign of each trade's R-multiple is a fair coin flip (i.e. the
    realized-magnitude sequence contains no true directional edge). Flip
    signs at random, keep magnitudes, recompute the mean -- a classic
    nonparametric test that doesn't assume normality."""
    mags = np.abs(r)
    obs_mean = r.mean()
    null_means = np.empty(n_perm)
    n = len(r)
    for i in range(n_perm):
        signs = RNG.choice([-1.0, 1.0], size=n)
        null_means[i] = (mags * signs).mean()
    p_value = (np.sum(null_means >= obs_mean) + 1) / (n_perm + 1)
    return obs_mean, null_means, p_value


def deflated_sharpe_ratio(sr_hat, T, skew, kurt, all_sharpes, N_effective):
    """Bailey & Lopez de Prado (2014) Deflated Sharpe Ratio.
    sr_hat: observed (non-annualized, per-trade) Sharpe-like statistic of the
            selected strategy (M4).
    T: track record length (number of trades).
    skew, kurt: skewness and (non-excess) kurtosis of M4's own trade returns.
    all_sharpes: array of Sharpe-like statistics from every trial run this
            project has ever certified/diagnosed (the empirical selection
            universe) -- used to estimate Var[SR_n] under the "many trials"
            null per BLdP's own recommended practical approximation.
    N_effective: number of independent trials assumed in the selection
            process (reported at two honesty levels by the caller).
    """
    euler_gamma = 0.5772156649
    var_sr = np.var(all_sharpes, ddof=1)
    sd_sr = np.sqrt(var_sr) if var_sr > 0 else 1e-9
    if N_effective > 1:
        z1 = sps.norm.ppf(1 - 1.0 / N_effective)
        z2 = sps.norm.ppf(1 - 1.0 / (N_effective * np.e))
        sr0 = sd_sr * ((1 - euler_gamma) * z1 + euler_gamma * z2)
    else:
        sr0 = 0.0
    denom = np.sqrt(max(1e-9, 1 - skew * sr_hat + (kurt - 1) / 4.0 * sr_hat ** 2))
    z = (sr_hat - sr0) * np.sqrt(T - 1) / denom
    psr = sps.norm.cdf(z)
    return dict(sr0_expected_max_under_null=round(float(sr0), 4),
                deflated_sharpe_prob=round(float(psr), 6),
                z_stat=round(float(z), 4), N_effective=N_effective,
                empirical_sd_of_all_trial_sharpes=round(float(sd_sr), 4))


def pbo_estimate(all_sharpes, sr_hat):
    """Lightweight PBO proxy: among the full set of trials this project ever
    ran, what fraction achieved a Sharpe >= the selected one purely by being
    in that set? (A full combinatorially-symmetric CSCV would need the raw
    per-trial OOS return series for every sleeve, which were not all
    persisted at that granularity -- this is the honest, available proxy:
    rank of the selected strategy within the full distribution of trials
    actually run.)"""
    all_sharpes = np.asarray(all_sharpes)
    rank = (all_sharpes < sr_hat).sum()
    pctile = rank / len(all_sharpes)
    return dict(n_trials_considered=len(all_sharpes),
                pctile_of_selected_strategy=round(float(pctile), 4),
                n_trials_beating_selected=int((all_sharpes >= sr_hat).sum()))


def main():
    r, pnl, qtr = load_trades("results/M4_geom_search_v2_records.json")
    n = len(r)
    sr_hat = sharpe_like(r)
    print(f"M4: n_trades={n}  mean_R={r.mean():.4f}  sd_R={r.std(ddof=1):.4f}  sharpe_like={sr_hat:.4f}")
    skew = float(sps.skew(r))
    kurt = float(sps.kurtosis(r, fisher=False))
    print(f"skew={skew:.4f}  kurtosis(non-excess)={kurt:.4f}")

    boot_sharpes, boot_means = iid_bootstrap(r)
    ci_sharpe = np.percentile(boot_sharpes, [2.5, 97.5])
    ci_mean = np.percentile(boot_means, [2.5, 97.5])
    p_mean_le_0 = float((boot_means <= 0).mean())
    print(f"\n[1] IID bootstrap ({N_BOOT} resamples):")
    print(f"    95% CI on sharpe_like: [{ci_sharpe[0]:.3f}, {ci_sharpe[1]:.3f}]")
    print(f"    95% CI on mean_R:      [{ci_mean[0]:.4f}, {ci_mean[1]:.4f}]")
    print(f"    P(mean_R <= 0) under bootstrap resampling: {p_mean_le_0:.4f}")

    block_sharpes, block_means = block_bootstrap_by_quarter(r, qtr)
    ci_sharpe_b = np.percentile(block_sharpes, [2.5, 97.5])
    p_mean_le_0_b = float((block_means <= 0).mean())
    print(f"\n[2] Block bootstrap (by year-month of exit, {N_BOOT} resamples):")
    print(f"    95% CI on sharpe_like: [{ci_sharpe_b[0]:.3f}, {ci_sharpe_b[1]:.3f}]")
    print(f"    P(mean_R <= 0) under block resampling: {p_mean_le_0_b:.4f}")

    obs_mean, null_means, p_perm = permutation_sign_flip_test(r)
    print(f"\n[3] Sign-flip permutation test ({N_BOOT} permutations):")
    print(f"    observed mean_R={obs_mean:.4f}  p-value (one-sided, H0: no directional edge)={p_perm:.4f}")

    sc = json.load(open("results/scorecards.json"))
    all_sharpes_full = np.array([v["sharpe"] for v in sc.values()])
    # Stricter, "certified-only" universe: drop explicitly-disclosed diagnostic variants
    # (relaxed friction / halt-removed / min-trade-override runs) and zero-trade sleeves.
    diag_markers = ("_diag", )
    certified_only = {k: v for k, v in sc.items() if not any(m in k for m in diag_markers) and v["n_trades"] > 0}
    all_sharpes_certified = np.array([v["sharpe"] for v in certified_only.values()])

    print(f"\n[4] Deflated Sharpe Ratio -- two honesty levels of N (number of trials):")
    dsr_full = deflated_sharpe_ratio(sr_hat, n, skew, kurt, all_sharpes_full, len(all_sharpes_full))
    print(f"    (a) N={len(all_sharpes_full)} (every sleeve/variant/diagnostic ever run, incl. relaxed-friction/no-halt diagnostics):")
    print(f"        {dsr_full}")
    dsr_cert = deflated_sharpe_ratio(sr_hat, n, skew, kurt, all_sharpes_certified, len(all_sharpes_certified))
    print(f"    (b) N={len(all_sharpes_certified)} (certified/non-diagnostic mission-invariant attempts only):")
    print(f"        {dsr_cert}")

    pbo_full = pbo_estimate(all_sharpes_full, sr_hat)
    pbo_cert = pbo_estimate(all_sharpes_certified, sr_hat)
    print(f"\n[5] PBO-proxy (percentile rank among all trials actually run):")
    print(f"    all trials incl. diagnostics: {pbo_full}")
    print(f"    certified-only trials:        {pbo_cert}")

    out = dict(
        n_trades=n, mean_r=float(r.mean()), sd_r=float(r.std(ddof=1)), sharpe_like=float(sr_hat),
        skew=skew, kurtosis_nonexcess=kurt,
        iid_bootstrap=dict(ci95_sharpe=ci_sharpe.tolist(), ci95_mean_r=ci_mean.tolist(), p_mean_r_le_0=p_mean_le_0),
        block_bootstrap_by_month=dict(ci95_sharpe=ci_sharpe_b.tolist(), p_mean_r_le_0=p_mean_le_0_b),
        sign_flip_permutation=dict(observed_mean_r=float(obs_mean), p_value=float(p_perm)),
        deflated_sharpe_all_trials=dsr_full,
        deflated_sharpe_certified_only=dsr_cert,
        pbo_proxy_all_trials=pbo_full,
        pbo_proxy_certified_only=pbo_cert,
    )
    with open("results/PHASE9_M4_dsr_bootstrap_audit.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("\nWrote results/PHASE9_M4_dsr_bootstrap_audit.json")


if __name__ == "__main__":
    main()
