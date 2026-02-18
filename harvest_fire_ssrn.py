"""
================================================================================
Harvest FIRE: A Dynamic Withdrawal Framework for Lifetime Consumption Maximization
================================================================================

Replication Code for SSRN Working Paper

Authors:  [Author Name]
Version:  1.0
Date:     2026-02

Description:
    This script replicates all simulations and figures reported in the paper.
    Five withdrawal strategies are compared across 20,000 Monte Carlo paths
    and 47 rolling 30-year historical windows (S&P 500, 1950–2024).

    Strategies:
        1. Traditional 4% Rule (60/40 stock-bond portfolio)
        2. Three-layer Bucket Strategy (Evensky & Katz, 2006)
        3. Harvest 15%  (reward ratio r = 0.15, no Seeding)
        4. Harvest 30%  (reward ratio r = 0.30, no Seeding)
        5. Harvest 30% + Seeding (r = 0.30, Seeding enabled)

    Sensitivity analyses:
        A. Reward ratio: 10%–50% in 5% increments
        B. Cash pool interest rate: 0%, 1%, 3%, 5%
        C. Initial cash pool size × reward ratio (2-D heatmap)
        D. Cash pool depletion and equity rescue tracking

Design note on inflation:
    base_withdrawal is inflation-adjusted (real purchasing power guaranteed).
    depletion_target is fixed in nominal terms (W0-based).
    surplus and bonus are computed in nominal terms.
    This means inflation causes nominal assets to drift above the depletion
    path more readily, making Harvest modestly favorable in inflationary
    environments — a deliberate design property noted in the paper.

Requirements:
    numpy, pandas, matplotlib, scipy

Usage:
    python harvest_fire_ssrn.py

Outputs (saved to ./outputs/):
    harvest_fire_table1.csv          — Monte Carlo summary (Table 1)
    harvest_fire_historical.csv      — Historical backtest (all windows)
    harvest_fire_reward_sensitivity.csv
    harvest_fire_cash_sensitivity.csv
    harvest_fire_figures.png         — Main figures (Fig 1–6)
    harvest_fire_boxplot.png         — Boxplot (Fig 1 & 2, publication style)
================================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings('ignore')

os.makedirs('./outputs', exist_ok=True)

# ── Global parameters ─────────────────────────────────────────────────────────
np.random.seed(42)

N_PATHS    = 20_000    # Monte Carlo paths
T          = 30        # Planning horizon (years)
W0         = 1_000_000 # Initial portfolio ($)
INFLATION  = 0.02      # Annual inflation rate
MU_E       = 0.11      # Equity mean return (Monte Carlo)
SIGMA_E    = 0.17      # Equity return std dev
MU_B       = 0.03      # Bond mean return
SIGMA_B    = 0.05      # Bond return std dev
CASH_RATE  = 0.03      # Cash pool interest rate (baseline)
INIT_CASH  = 5 * 0.04 * W0   # Initial cash pool = 5 years of base withdrawal

# Seeding parameters
SEED_THRESH   = -0.15  # Equity return threshold to trigger Seeding
SEED_RATIO    = 0.40   # Fraction of cash pool redeployed to equity
SEED_CASH_MIN = 0.20   # Minimum cash/total-assets ratio required for Seeding

# Historical S&P 500 total returns 1950–2024 (Damodaran, NYU Stern)
SP500 = np.array([
    0.3181, 0.2402, 0.1837,-0.0099, 0.5262, 0.3156, 0.0656,-0.1078, 0.4336, 0.1196,
    0.0062, 0.2664,-0.0881, 0.2268, 0.1645, 0.1242, 0.0306, 0.2397, 0.1106,-0.0850,
    0.0401, 0.1431, 0.1898,-0.1466,-0.2647, 0.3720, 0.2384,-0.0718, 0.0656, 0.1844,
    0.3242,-0.0491, 0.2141, 0.2251, 0.0627, 0.3216, 0.1847, 0.0543, 0.1661, 0.3129,
    0.0761,-0.0310, 0.3047, 0.0762, 0.1008, 0.0132, 0.3723, 0.2368, 0.3111, 0.2645,
    0.1900,-0.0910,-0.1189,-0.2210, 0.2868, 0.1088, 0.0491, 0.1579, 0.0549,-0.3700,
    0.2646, 0.1506, 0.0211, 0.1600, 0.3239, 0.1369, 0.0138, 0.1196, 0.2183,-0.0438,
    0.3149, 0.1840, 0.2883, 0.2661, 0.0265, 0.1689
])

# ── Helper functions ──────────────────────────────────────────────────────────

def base_withdrawal(t):
    """Inflation-adjusted guaranteed base withdrawal (4% of W0 in real terms)."""
    return 0.04 * W0 * (1 + INFLATION) ** t

def depletion_target(t):
    """
    Nominal depletion path: linear glide from W0 to 0 over T years.
    Fixed in nominal terms; surplus computed relative to this path.
    """
    return W0 * (1 - t / T)

# ── Strategy 1: Traditional 4% Rule (60/40) ──────────────────────────────────

def simulate_4pct_rule(re, rb):
    """
    Annual withdrawal = base_withdrawal(t), fixed in real terms.
    Portfolio: 60% equity, 40% bonds, rebalanced annually.
    """
    portfolio = W0
    total_consumption = 0
    for t in range(T):
        portfolio *= (0.60 * (1 + re[t]) + 0.40 * (1 + rb[t]))
        withdrawal = min(portfolio, base_withdrawal(t))
        portfolio -= withdrawal
        total_consumption += withdrawal
        if portfolio <= 0:
            return total_consumption, 0, False
    return total_consumption, portfolio, True

# ── Strategy 2: Three-layer Bucket (Evensky & Katz, 2006) ────────────────────

def simulate_bucket(re, rb):
    """
    Cash bucket:  5 years of base withdrawal ($200K)
    Bond bucket:  5 years of base withdrawal ($200K)
    Equity bucket: remainder ($600K)
    Replenishment rules:
      - Cash < 2yr floor → replenish from bonds up to 5yr target
      - Bond < 3yr floor and equity return > 0 → replenish from equity up to 5yr target
    """
    cash_b   = 5 * base_withdrawal(0)
    bond_b   = 5 * base_withdrawal(0)
    equity_b = W0 - cash_b - bond_b
    total_consumption = 0

    for t in range(T):
        equity_b *= (1 + re[t])
        bond_b   *= (1 + rb[t])
        bwd = base_withdrawal(t)
        fc = min(cash_b, bwd);   cash_b -= fc;   rem = bwd - fc
        fb = min(bond_b, rem);   bond_b -= fb;   rem -= fb
        fe = min(equity_b, rem); equity_b -= fe
        total_consumption += fc + fb + fe
        # Replenish cash from bonds
        if cash_b < 2 * base_withdrawal(t):
            r = min(bond_b, 5 * base_withdrawal(t) - cash_b)
            bond_b -= r; cash_b += r
        # Replenish bonds from equity (only on up years)
        if bond_b < 3 * base_withdrawal(t) and re[t] > 0:
            r = min(equity_b, 5 * base_withdrawal(t) - bond_b)
            equity_b -= r; bond_b += r
        if equity_b + bond_b + cash_b <= 0:
            return total_consumption, 0, False
    return total_consumption, equity_b + bond_b + cash_b, True

# ── Strategies 3–5: Harvest Framework ────────────────────────────────────────

def simulate_harvest(re, reward_ratio, seeding=False,
                     cash_rate=CASH_RATE, init_cash=INIT_CASH,
                     track_cash=False):
    """
    Harvest FIRE withdrawal framework.

    Parameters
    ----------
    re           : array of equity returns (length T)
    reward_ratio : bonus = reward_ratio * surplus (r in paper)
    seeding      : if True, redeploy cash to equity on large drawdowns
    cash_rate    : annual interest rate on cash pool (nominal)
    init_cash    : initial cash pool size
    track_cash   : if True, return depletion/rescue year lists

    Design note
    -----------
    base_withdrawal is inflation-adjusted (real guarantee).
    depletion_target and surplus are in nominal terms.
    bonus = reward_ratio * nominal surplus (also nominal).
    Inflation causes nominal assets to exceed the fixed nominal depletion
    path more readily, making surplus harvesting slightly more frequent
    in inflationary environments.

    Returns
    -------
    total_consumption, terminal_wealth, success
    (+ cash_depletion_years, equity_rescue_years if track_cash=True)
    """
    E = W0 - init_cash
    C = init_cash
    total_consumption = 0
    cash_depletion_years = []
    equity_rescue_years  = []

    for t in range(T):
        # Step 1: Growth
        E *= (1 + re[t])
        C *= (1 + cash_rate)

        # Step 2: Surplus harvest — transfer entire surplus to cash pool
        surplus = max(E - depletion_target(t + 1), 0)
        if surplus > 0:
            E -= surplus
            C += surplus

        # Step 3: Seeding (optional) — redeploy cash on large drawdowns
        if seeding and re[t] < SEED_THRESH:
            total_assets = E + C
            if total_assets > 0 and C / total_assets > SEED_CASH_MIN:
                seed = SEED_RATIO * C
                C -= seed
                E += seed

        # Step 4: Withdrawal
        bwd      = base_withdrawal(t)          # inflation-adjusted base
        bonus    = reward_ratio * surplus       # nominal bonus
        total_wd = bwd + bonus

        from_cash = min(C, total_wd);     C -= from_cash
        shortfall = total_wd - from_cash
        from_eq   = min(E, shortfall);    E -= from_eq
        withdrawal = from_cash + from_eq
        total_consumption += withdrawal

        if track_cash:
            if C <= 1:
                cash_depletion_years.append(t)
            if shortfall > 0 and from_eq > 0:
                equity_rescue_years.append(t)

        if E + C <= 0:
            if track_cash:
                return total_consumption, 0, False, cash_depletion_years, equity_rescue_years
            return total_consumption, 0, False

    if track_cash:
        return total_consumption, E + C, True, cash_depletion_years, equity_rescue_years
    return total_consumption, E + C, True

# ── Monte Carlo simulation ────────────────────────────────────────────────────

def run_monte_carlo(n=N_PATHS):
    """Run all 5 strategies across n Monte Carlo paths."""
    re = np.random.normal(MU_E, SIGMA_E, (n, T))
    rb = np.random.normal(MU_B, SIGMA_B, (n, T))

    strategies = {
        '4% Rule (60/40)':     lambda i: simulate_4pct_rule(re[i], rb[i]),
        'Bucket (3-layer)':    lambda i: simulate_bucket(re[i], rb[i]),
        'Harvest 15%':         lambda i: simulate_harvest(re[i], 0.15),
        'Harvest 30%':         lambda i: simulate_harvest(re[i], 0.30),
        'Harvest 30%+Seeding': lambda i: simulate_harvest(re[i], 0.30, seeding=True),
    }

    results = {}
    for name, fn in strategies.items():
        consumptions, terminals, successes = [], [], []
        for i in range(n):
            c, tw, s = fn(i)
            consumptions.append(c); terminals.append(tw); successes.append(s)
        ca = np.array(consumptions); ta = np.array(terminals)
        results[name] = {
            'success_rate':       np.mean(successes),
            'median_consumption': np.median(ca),
            'median_terminal':    np.median(ta),
            'iqr_terminal':       np.percentile(ta,75) - np.percentile(ta,25),
            'consumptions':       ca,
            'terminals':          ta,
        }
    return results, re, rb

# ── Sensitivity analyses ──────────────────────────────────────────────────────

def run_reward_sensitivity(n=N_PATHS):
    """Reward ratio sensitivity: 10%–50% in 5% steps."""
    re = np.random.normal(MU_E, SIGMA_E, (n, T))
    rows = []
    for r in np.arange(0.10, 0.55, 0.05):
        c_list, s_list = [], []
        for i in range(n):
            c, _, s = simulate_harvest(re[i], r)
            c_list.append(c); s_list.append(s)
        rows.append({
            'Reward Ratio':          f"{int(r*100)}%",
            'Success Rate':          np.mean(s_list),
            'Median Consumption':    np.median(c_list),
            'vs 4% Rule':            f"+{(np.median(c_list)/1_622_723-1)*100:.0f}%",
        })
    return pd.DataFrame(rows)

def run_cash_rate_sensitivity(n=N_PATHS):
    """Cash pool interest rate sensitivity: 0%, 1%, 3%, 5%."""
    re = np.random.normal(MU_E, SIGMA_E, (n, T))
    rows = []
    for cr in [0.00, 0.01, 0.03, 0.05]:
        c_list, s_list = [], []
        for i in range(n):
            c, _, s = simulate_harvest(re[i], 0.30, cash_rate=cr)
            c_list.append(c); s_list.append(s)
        rows.append({
            'Cash Rate':          f"{int(cr*100)}%",
            'Success Rate':       np.mean(s_list),
            'Median Consumption': np.median(c_list),
        })
    return pd.DataFrame(rows)

def run_2d_sensitivity(n=5_000):
    """2-D heatmap: initial cash pool size × reward ratio."""
    re = np.random.normal(MU_E, SIGMA_E, (n, T))
    init_cash_years = [1, 2, 3, 5, 7, 10]
    reward_ratios   = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
    succ  = np.zeros((len(init_cash_years), len(reward_ratios)))
    cons  = np.zeros_like(succ)
    depl  = np.zeros_like(succ)
    for i, yrs in enumerate(init_cash_years):
        ic = yrs * base_withdrawal(0)
        for j, rr in enumerate(reward_ratios):
            c_l, s_l, d_l = [], [], []
            for k in range(n):
                c, _, s, dep, _ = simulate_harvest(re[k], rr, init_cash=ic, track_cash=True)
                c_l.append(c); s_l.append(s); d_l.append(1 if dep else 0)
            succ[i,j] = np.mean(s_l)
            cons[i,j] = np.median(c_l) / 1e6
            depl[i,j] = np.mean(d_l)
    return init_cash_years, reward_ratios, succ, cons, depl

def run_cash_depletion_analysis(n=N_PATHS):
    """Track cash pool depletion and equity rescue by year."""
    re = np.random.normal(MU_E, SIGMA_E, (n, T))
    results = {}
    for label, rr, seed in [
        ('Harvest 15%', 0.15, False),
        ('Harvest 30%', 0.30, False),
        ('Harvest 30%+Seeding', 0.30, True),
    ]:
        dep_rate, resc_rate = 0, 0
        dep_by_yr  = np.zeros(T)
        resc_by_yr = np.zeros(T)
        resc_durs  = []
        for i in range(n):
            _, _, _, dep_yrs, resc_yrs = simulate_harvest(
                re[i], rr, seeding=seed, track_cash=True)
            if dep_yrs:
                dep_rate += 1
                for y in dep_yrs: dep_by_yr[y] += 1
            if resc_yrs:
                resc_rate += 1
                resc_durs.append(len(resc_yrs))
                for y in resc_yrs: resc_by_yr[y] += 1
        results[label] = {
            'depletion_rate':    dep_rate / n,
            'rescue_rate':       resc_rate / n,
            'depletion_by_year': dep_by_yr / n,
            'rescue_by_year':    resc_by_yr / n,
            'median_rescue_dur': np.median(resc_durs) if resc_durs else 0,
            'max_rescue_dur':    np.max(resc_durs) if resc_durs else 0,
        }
    return results

# ── Historical backtest ───────────────────────────────────────────────────────

def run_historical_backtest():
    """47 rolling 30-year windows, S&P 500 1950–2024."""
    rows = []
    rb = np.full(T, 0.03)
    for start in range(len(SP500) - T + 1):
        ret = SP500[start:start+T]
        window = f"{1950+start}–{1950+start+T-1}"
        c4,  _, s4   = simulate_4pct_rule(ret, rb)
        ch15,_, sh15 = simulate_harvest(ret, 0.15)
        ch30,_, sh30 = simulate_harvest(ret, 0.30)
        chs, _, shs  = simulate_harvest(ret, 0.30, seeding=True)
        rows.append({
            'Window':           window,
            '4% Consumption':   c4,   '4% Success':   s4,
            'H15 Consumption':  ch15, 'H15 Success':  sh15,
            'H30 Consumption':  ch30, 'H30 Success':  sh30,
            'H30S Consumption': chs,  'H30S Success': shs,
            'H15 vs 4% (%)':    round((ch15/c4-1)*100, 1),
            'H30 vs 4% (%)':    round((ch30/c4-1)*100, 1),
            'H30S vs 4% (%)':   round((chs/c4-1)*100, 1),
        })
    return pd.DataFrame(rows)

# ── Figures ───────────────────────────────────────────────────────────────────

COLORS = {
    '4% Rule (60/40)':     '#4C72B0',
    'Bucket (3-layer)':    '#DD8452',
    'Harvest 15%':         '#55A868',
    'Harvest 30%':         '#C44E52',
    'Harvest 30%+Seeding': '#8172B2',
}

def plot_main_figures(mc_results, reward_df, cash_df, hist_df):
    fig = plt.figure(figsize=(18, 22))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    # Fig 1: KDE of lifetime consumption
    ax1 = fig.add_subplot(gs[0, 0])
    for name, res in mc_results.items():
        vals = res['consumptions'] / 1e6
        kde  = gaussian_kde(vals, bw_method=0.15)
        xr   = np.linspace(0.5, 4.5, 500)
        ax1.plot(xr, kde(xr), color=COLORS[name], lw=2.2, label=name)
        ax1.fill_between(xr, kde(xr), alpha=0.15, color=COLORS[name])
        ax1.axvline(np.median(vals), color=COLORS[name], lw=1.5, ls='--', alpha=0.8)
    ax1.set(xlabel='Total Lifetime Consumption ($M)', ylabel='Density',
            title='Figure 1: Distribution of Total Lifetime Consumption\n(KDE; dashed = medians)',
            xlim=(0.5, 4.5))
    ax1.legend(fontsize=8)

    # Fig 2: KDE of terminal wealth
    ax2 = fig.add_subplot(gs[0, 1])
    for name, res in mc_results.items():
        vals = np.clip(res['terminals'] / 1e6, 0, 15)
        kde  = gaussian_kde(vals, bw_method=0.15)
        xr   = np.linspace(0, 15, 500)
        ax2.plot(xr, kde(xr), color=COLORS[name], lw=2.2, label=name)
        ax2.fill_between(xr, kde(xr), alpha=0.15, color=COLORS[name])
    ax2.set(xlabel='Terminal Wealth ($M, clipped at 15M)', ylabel='Density',
            title='Figure 2: Distribution of Terminal Wealth')
    ax2.legend(fontsize=8)

    # Fig 3: Reward ratio sensitivity
    ax3  = fig.add_subplot(gs[1, 0])
    ax3b = ax3.twinx()
    x3   = range(len(reward_df))
    ax3.plot(x3, reward_df['Median Consumption']/1e6, 'o-', color='#2ecc71', lw=2, label='Median Consumption')
    ax3b.plot(x3, reward_df['Success Rate']*100, 's--', color='#e74c3c', lw=2, label='Success Rate')
    ax3.axhline(1.622723*2, color='gray', ls=':', lw=1.5, label='2× baseline ($3.25M)')
    ax3.set_xticks(x3); ax3.set_xticklabels(reward_df['Reward Ratio'], fontsize=9)
    ax3.set(xlabel='Reward Ratio', ylabel='Median Consumption ($M)',
            title='Figure 3: Reward Ratio Sensitivity (10%–50%)')
    ax3b.set_ylabel('Success Rate (%)', color='#e74c3c')
    lines = ax3.get_legend_handles_labels(); lines2 = ax3b.get_legend_handles_labels()
    ax3.legend(lines[0]+lines2[0], lines[1]+lines2[1], fontsize=8)

    # Fig 4: Cash rate sensitivity
    ax4  = fig.add_subplot(gs[1, 1])
    ax4b = ax4.twinx()
    x4   = range(len(cash_df))
    ax4.bar(x4, cash_df['Median Consumption']/1e6, color='#3498db', alpha=0.7, label='Median Consumption')
    ax4b.plot(x4, cash_df['Success Rate']*100, 'ro-', lw=2, label='Success Rate')
    ax4.set_xticks(x4); ax4.set_xticklabels([f"Cash {r}" for r in cash_df['Cash Rate']])
    ax4.set(ylabel='Median Consumption ($M)',
            title='Figure 4: Cash Pool Interest Rate Sensitivity (Harvest 30%)')
    ax4b.set_ylabel('Success Rate (%)', color='red')
    lines = ax4.get_legend_handles_labels(); lines2 = ax4b.get_legend_handles_labels()
    ax4.legend(lines[0]+lines2[0], lines[1]+lines2[1], fontsize=9)

    # Fig 5: Historical backtest
    ax5 = fig.add_subplot(gs[2, 0])
    x5  = range(len(hist_df))
    ax5.plot(x5, hist_df['4% Consumption']/1e6,      'b-o', ms=3, lw=1.5, label='4% Rule')
    ax5.plot(x5, hist_df['H15 Consumption']/1e6,     'g-^', ms=3, lw=1.5, label='Harvest 15%')
    ax5.plot(x5, hist_df['H30S Consumption']/1e6,    'r-s', ms=3, lw=1.5, label='Harvest 30%+Seeding')
    step = max(1, len(hist_df)//8)
    ax5.set_xticks(range(0, len(hist_df), step))
    ax5.set_xticklabels([hist_df['Window'].iloc[i].split('–')[0]
                         for i in range(0, len(hist_df), step)], rotation=45, fontsize=9)
    ax5.set(ylabel='Total Lifetime Consumption ($M)',
            title='Figure 5: Historical Backtest (Rolling 30-Year Windows, 1950–2024)')
    ax5.legend(fontsize=9); ax5.grid(alpha=0.3)

    # Fig 6: Summary table
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis('off')
    table_data = [[name,
                   f"{res['success_rate']*100:.1f}%",
                   f"${res['median_consumption']/1e6:.2f}M",
                   f"${res['median_terminal']/1e6:.2f}M",
                   f"${res['iqr_terminal']/1e6:.2f}M"]
                  for name, res in mc_results.items()]
    tbl = ax6.table(cellText=table_data,
                    colLabels=['Strategy','Success','Median\nConsumption',
                               'Median\nTerminal','Terminal\nIQR'],
                    cellLoc='center', loc='center', bbox=[0, 0.1, 1, 0.85])
    tbl.auto_set_font_size(False); tbl.set_fontsize(8)
    ax6.set_title('Table 1: Monte Carlo Summary (20,000 paths)', fontsize=11, fontweight='bold')

    plt.suptitle('Harvest FIRE: Simulation Results', fontsize=16, fontweight='bold', y=1.01)
    plt.savefig('./outputs/harvest_fire_figures.png', dpi=150, bbox_inches='tight')
    plt.close()

def plot_boxplots(mc_results):
    """Publication-style boxplots: consumption and terminal wealth."""
    labels_short = ['4%', 'Bucket', 'H15', 'H30', 'H30+Seed']
    keys         = list(mc_results.keys())
    data_c = [mc_results[k]['consumptions']/1e6 for k in keys]
    data_t = [mc_results[k]['terminals']/1e6    for k in keys]

    box_props = dict(
        patch_artist=True,
        medianprops=dict(color='#4FC3F7', lw=2),
        whiskerprops=dict(color='#555', lw=1.2),
        capprops=dict(color='#555', lw=1.2),
        flierprops=dict(marker='+', markersize=3, alpha=0.15, color='#999'),
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor('white')

    bp1 = ax1.boxplot(data_c, labels=labels_short, **box_props)
    for p in bp1['boxes']:
        p.set_facecolor('white'); p.set_edgecolor('#555'); p.set_linewidth(1.2)
    ax1.set_ylim(bottom=1.4)   # floor: focus on variance above baseline
    ax1.set(title='Figure 1: Total Lifetime Consumption (Millions USD)',
            ylabel='Millions USD')
    ax1.yaxis.grid(True, linestyle='--', alpha=0.5); ax1.set_axisbelow(True)
    ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)

    bp2 = ax2.boxplot(data_t, labels=labels_short, **box_props)
    for p in bp2['boxes']:
        p.set_facecolor('white'); p.set_edgecolor('#555'); p.set_linewidth(1.2)
    ax2.set(title='Figure 2: Terminal Wealth (Millions USD)', ylabel='Millions USD')
    ax2.yaxis.grid(True, linestyle='--', alpha=0.5); ax2.set_axisbelow(True)
    ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)

    plt.tight_layout(pad=3)
    plt.savefig('./outputs/harvest_fire_boxplot.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    print("=" * 60)
    print("Harvest FIRE Replication Code")
    print("=" * 60)

    print("\n[1/6] Monte Carlo simulation (20,000 paths)...")
    mc_results, _, _ = run_monte_carlo()
    rows = []
    for name, res in mc_results.items():
        rows.append({'Strategy': name,
                     'Success Rate': f"{res['success_rate']*100:.1f}%",
                     'Median Consumption': f"${res['median_consumption']:,.0f}",
                     'Median Terminal':    f"${res['median_terminal']:,.0f}",
                     'Terminal IQR':       f"${res['iqr_terminal']:,.0f}"})
    df_table1 = pd.DataFrame(rows)
    print(df_table1.to_string(index=False))
    df_table1.to_csv('./outputs/harvest_fire_table1.csv', index=False)

    print("\n[2/6] Reward ratio sensitivity (10%–50%)...")
    reward_df = run_reward_sensitivity()
    print(reward_df.to_string(index=False))
    reward_df.to_csv('./outputs/harvest_fire_reward_sensitivity.csv', index=False)

    print("\n[3/6] Cash pool interest rate sensitivity...")
    cash_df = run_cash_rate_sensitivity()
    print(cash_df.to_string(index=False))
    cash_df.to_csv('./outputs/harvest_fire_cash_sensitivity.csv', index=False)

    print("\n[4/6] Cash depletion analysis...")
    dep_results = run_cash_depletion_analysis()
    for label, res in dep_results.items():
        print(f"  {label}: depletion={res['depletion_rate']*100:.1f}%  "
              f"rescue={res['rescue_rate']*100:.1f}%  "
              f"median rescue duration={res['median_rescue_dur']:.0f}yr  "
              f"max={res['max_rescue_dur']:.0f}yr")

    print("\n[5/6] Historical backtest (47 windows, 1950–2024)...")
    hist_df = run_historical_backtest()
    for col, label in [('4% Consumption','4% Rule'), ('H15 Consumption','Harvest 15%'),
                       ('H30 Consumption','Harvest 30%'), ('H30S Consumption','H30+Seeding')]:
        print(f"  {label}: median=${hist_df[col].median():,.0f}  "
              f"min=${hist_df[col].min():,.0f}  max=${hist_df[col].max():,.0f}")
    med4 = hist_df['4% Consumption'].median()
    print(f"\n  Harvest premium (historical median):")
    for col, label in [('H15 Consumption','H15'), ('H30 Consumption','H30'),
                       ('H30S Consumption','H30+Seed')]:
        print(f"    {label}: +{(hist_df[col].median()/med4-1)*100:.1f}%")
    hist_df.to_csv('./outputs/harvest_fire_historical.csv', index=False)

    print("\n[6/6] Generating figures...")
    plot_main_figures(mc_results, reward_df, cash_df, hist_df)
    plot_boxplots(mc_results)

    print("\nAll outputs saved to ./outputs/")
    print("=" * 60)
