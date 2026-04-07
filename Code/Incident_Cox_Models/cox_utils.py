import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter, KaplanMeierFitter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Scripts.get_all_data import read_dfs, build_new_dfs

FOLDER_PATH = '/Users/daniellancet/Desktop/Spring_2026/CDSS_170/CANS_INCIDENTS_PROJECT/Aspiranet_Data'
CATEGORY_MAPPING_PATH = '/Users/daniellancet/Desktop/Spring_2026/CDSS_170/CANS_INCIDENTS_PROJECT/Novel_Data/category_mapping.csv'
DEMO_COLS = ['Black', 'White', 'Latino', 'Other', 'M', 'AgeWhenAssessed']


def load_data(folder_path=FOLDER_PATH, filter_amount=200, filter_interval=True):
    """Load and process all data. Returns dfs, top_n_cols, base_cols, incident_cols_dict, group_cols."""
    dfs = read_dfs(folder_path=folder_path)
    dfs, top_n_cols, base_cols, incident_cols_dict, group_cols = build_new_dfs(
        dfs, filter_amount=filter_amount, filter_interval=filter_interval
    )
    return dfs, top_n_cols, base_cols, incident_cols_dict, group_cols


def cox_incident_table(incident_cat, dfs, base_cols, incident_cols_dict):
    """Join base columns with incident-specific columns."""
    full_hazard_table = dfs['FULL_HAZARD_TABLE']
    base_table = full_hazard_table[base_cols]
    incident_table = full_hazard_table[incident_cols_dict[incident_cat]]
    return base_table.join(incident_table, how='inner')


def build_cox_df(incident_related_cox_table, incident_cat, group_cols, demo_cols=None):
    """Select and return model columns from the hazard table."""
    if demo_cols is None:
        demo_cols = DEMO_COLS
    event_col = f'{incident_cat}_event'
    stop_col = f'{incident_cat}_stop_days'
    cox_cols = (
        [event_col, 'start_days', 'OptionsNumber', stop_col]
        + group_cols
        + demo_cols
        + [f'{incident_cat}_days_since_last_incident', f'{incident_cat}_prior_incident_count']
    )
    cox_df = incident_related_cox_table[cox_cols].copy()
    print(f"Shape: {cox_df.shape}")
    print(f"Events: {cox_df[event_col].sum()}")
    return cox_df


def fit_cox_model(cox_df, incident_cat):
    """Fit a Cox PH model with entry/exit times clustered by individual."""
    cph = CoxPHFitter()
    cph.fit(
        cox_df,
        duration_col=f'{incident_cat}_stop_days',
        event_col=f'{incident_cat}_event',
        entry_col='start_days',
        cluster_col='OptionsNumber'
    )
    return cph


def check_cox_assumptions(cph, cox_df, covariates=None):
    """
    Check proportional hazards assumption via complementary log-log plots.

    lifelines' built-in check_assumptions() raises NotImplementedError for models
    fit with entry_col (left-truncated data). This function uses stratified
    Kaplan-Meier curves instead — KaplanMeierFitter accepts entry=, so it works
    correctly with delayed-entry data.

    For each covariate, the data is split into two groups (binary covariates are
    used as-is; continuous/ordinal covariates are split at the median). log(−log(S(t)))
    is plotted against log(t) for each group. Parallel lines support the PH assumption;
    diverging or crossing lines suggest a violation.

    Also prints the model's concordance index as a discrimination metric.

    Parameters
    ----------
    cph : fitted CoxPHFitter
    cox_df : pd.DataFrame
        The same dataframe used to fit cph.
    covariates : list, optional
        Subset of model covariates to check. Defaults to all covariates in the model.
    """
    duration_col = cph.duration_col
    event_col = cph.event_col
    entry_col = cph.entry_col

    covariates = covariates or cph.params_.index.tolist()
    n = len(covariates)
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    axes_flat = np.array(axes).flatten()

    for i, col in enumerate(covariates):
        ax = axes_flat[i]
        vals = cox_df[col].dropna()
        unique_vals = sorted(vals.unique())

        if len(unique_vals) == 2:
            strata = {str(v): cox_df[cox_df[col] == v] for v in unique_vals}
        else:
            med = vals.median()
            strata = {
                f'≤{med:.1f}': cox_df[cox_df[col] <= med],
                f'>{med:.1f}': cox_df[cox_df[col] > med],
            }

        for label, sub in strata.items():
            if sub[event_col].sum() < 3:
                continue
            kmf = KaplanMeierFitter()
            kmf.fit(
                durations=sub[duration_col],
                event_observed=sub[event_col],
                entry=sub[entry_col] if entry_col else None,
                label=label
            )
            sf = kmf.survival_function_
            t = sf.index.values
            s = sf[label].values
            mask = (s > 0) & (s < 1) & (t > 0)
            if mask.sum() < 2:
                continue
            ax.plot(np.log(t[mask]), np.log(-np.log(s[mask])), label=label)

        ax.set_title(col.replace('_', ' '), fontsize=9)
        ax.set_xlabel('log(t)', fontsize=8)
        ax.set_ylabel('log(−log(S))', fontsize=8)
        ax.legend(fontsize=7)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        'Proportional Hazards Check — Complementary Log-Log Plots\n'
        'Parallel lines → PH holds   |   Crossing/diverging lines → possible violation',
        fontsize=11
    )
    plt.tight_layout()
    plt.show()

    print(f"Concordance Index: {cph.concordance_index_:.4f}  "
          "(0.5 = random, 1.0 = perfect discrimination)")


def plot_cox_coefs(cph_model, exclude_vars=None, sort_by="coef", figsize_scale=0.5,
                   fade_nonsignificant=True, title=None):
    """Horizontal bar chart of log hazard ratios with 95% CI and center dots."""
    summary = cph_model.summary.copy()

    if exclude_vars:
        summary = summary.drop(index=[v for v in exclude_vars if v in summary.index])

    if sort_by == "abs_coef":
        summary = summary.reindex(summary["coef"].abs().sort_values().index)
    else:
        summary = summary.sort_values(by="coef")

    neg_color = "#a8ddb5"
    pos_color = "#f4a6a6"

    if fade_nonsignificant:
        colors = [
            neg_color if (coef < 0 and p < 0.05)
            else pos_color if (coef > 0 and p < 0.05)
            else "#d3d3d3"
            for coef, p in zip(summary["coef"], summary["p"])
        ]
    else:
        colors = [neg_color if coef < 0 else pos_color for coef in summary["coef"]]

    group_names = [name.replace('_', ' ') for name in summary.index.tolist()]

    fig, ax = plt.subplots(figsize=(9, figsize_scale * len(summary)))
    ax.barh(group_names, summary["coef"], color=colors, zorder=2)

    lower_err = summary["coef"] - summary["coef lower 95%"]
    upper_err = summary["coef upper 95%"] - summary["coef"]
    ax.errorbar(
        summary["coef"], group_names,
        xerr=[lower_err, upper_err],
        fmt='none', ecolor='black', capsize=3, zorder=3
    )
    ax.scatter(summary["coef"], group_names, color='black', zorder=4, s=20)
    ax.axvline(0, color='black', linestyle='--')
    ax.set_title(title or "Log Hazard Ratio (coef)")
    ax.set_xlabel("Effect Size")
    plt.tight_layout()
    plt.show()


def plot_prior_incident_risk_curve(cph, incident_cat, incident_display, values=None):
    """Survival curves showing risk by number of prior incidents."""
    if values is None:
        values = [0, 1, 2, 3]
    col = f'{incident_cat}_prior_incident_count'
    fig, ax = plt.subplots(figsize=(8, 5))
    cph.plot_partial_effects_on_outcome(
        covariates=col,
        values=values,
        ax=ax,
        plot_baseline=False
    )
    ax.set_title(f"Survival by Prior Incident Count\n({incident_display})")
    ax.set_xlabel("Days")
    ax.set_ylabel("Survival Probability (No Incident)")
    ax.set_xlim(0, 365 * 3)
    ax.legend(title="Prior Incidents", labels=[f"{v} prior incident(s)" for v in values])
    plt.tight_layout()
    plt.show()


def drill_down(cph, group_cols, incident_related_cox_table, incident_cat, incident_display,
               demo_cols=None, category_mapping_path=CATEGORY_MAPPING_PATH, top_n=3):
    """
    Identify the top risk group, fit a Cox model on its individual CANS items,
    plot the coefficients, and print the top_n most impactful items.

    Returns: cph_drill, drill_df, top_items (list of up to top_n item names)
    """
    if demo_cols is None:
        demo_cols = DEMO_COLS

    top_group = cph.summary.loc[group_cols, 'coef'].idxmax()
    print(f"Top risk group: {top_group}")

    category_mapping = pd.read_csv(category_mapping_path)
    group_items = category_mapping.loc[category_mapping['group'] == top_group, 'variable'].tolist()
    group_items = [v for v in group_items if v in incident_related_cox_table.columns]
    print(f"Items in group: {group_items}")

    drill_cols = (
        [f'{incident_cat}_event', 'start_days', 'OptionsNumber', f'{incident_cat}_stop_days']
        + group_items + demo_cols
    )
    drill_df = incident_related_cox_table[drill_cols].copy()

    cph_drill = CoxPHFitter()
    cph_drill.fit(
        drill_df,
        duration_col=f'{incident_cat}_stop_days',
        event_col=f'{incident_cat}_event',
        entry_col='start_days',
        cluster_col='OptionsNumber'
    )

    plot_cox_coefs(
        cph_drill,
        exclude_vars=demo_cols,
        title=f"Log Hazard Ratio — {top_group.replace('_', ' ')} Items ({incident_display})"
    )

    top_items = (
        cph_drill.summary.loc[group_items, 'coef']
        .abs()
        .nlargest(min(top_n, len(group_items)))
        .index.tolist()
    )
    print(f"\nTop {len(top_items)} most impactful CANS scores:")
    for item in top_items:
        print(f"  {item}: coef = {cph_drill.summary.loc[item, 'coef']:.4f}, "
              f"p = {cph_drill.summary.loc[item, 'p']:.4e}")

    return cph_drill, drill_df, top_items


def plot_partial_effects(cph_drill, top_items, incident_display):
    """Survival curves for each item in top_items at score values 0–3, one figure per item."""
    for item in top_items:
        fig, ax = plt.subplots(figsize=(8, 5))
        cph_drill.plot_partial_effects_on_outcome(
            covariates=item,
            values=[0, 1, 2, 3],
            ax=ax,
            plot_baseline=False
        )
        ax.set_title(f"Partial Effects: {item}\n({incident_display})")
        ax.set_xlabel("Days")
        ax.set_ylabel("Survival Probability (no incident)")
        ax.set_xlim(0, 365 * 3)
        ax.legend(title=item, labels=[f"{item} = {v}" for v in [0, 1, 2, 3]])
        plt.tight_layout()
        plt.show()


def add_delta_features(col_list, table, k=1):
    """
    For each column in col_list, add a delta column = current value minus the k-th lag.
    First assessments (no prior data) get delta = 0.
    New columns are named delta_{k}_{col_name} where spaces are replaced with underscores.
    """
    table = table.copy().sort_values(by=['OptionsNumber', 'DateCompleted'])
    for col in col_list:
        col_name = col.replace(' ', '_')
        lag_col = f'lag_{k}_{col_name}'
        if lag_col not in table.columns:
            table[lag_col] = table.groupby('OptionsNumber')[col].shift(k)
        table[f'delta_{k}_{col_name}'] = (table[col] - table[lag_col]).fillna(0)
    return table


def build_level_vs_change_cox_df(incident_related_cox_table, incident_cat,
                                  feature_cols, demo_cols=None, k=1):
    """
    Build a Cox df with both level (current score) and delta (change from k-th prior
    assessment) features so we can test which signal drives incident risk.

    Delta is filled with 0 for first assessments (no prior value to diff against).

    Returns
    -------
    cox_df : pd.DataFrame
        Ready to pass to fit_cox_model.
    feature_pairs : list of (level_col, delta_col) tuples
        Pairs linking each original feature to its delta counterpart.
    """
    if demo_cols is None:
        demo_cols = DEMO_COLS

    table = add_delta_features(feature_cols, incident_related_cox_table, k=k)

    delta_cols = [f'delta_{k}_{col.replace(" ", "_")}' for col in feature_cols]

    cox_cols = (
        [f'{incident_cat}_event', 'start_days', 'OptionsNumber', f'{incident_cat}_stop_days']
        + feature_cols + delta_cols + demo_cols
    )
    cox_df = table[cox_cols].copy()

    print(f"Shape: {cox_df.shape}")
    print(f"Events: {cox_df[f'{incident_cat}_event'].sum()}")

    feature_pairs = list(zip(feature_cols, delta_cols))
    return cox_df, feature_pairs


def plot_level_vs_change_coefs(cph, feature_pairs, title=None):
    """
    Side-by-side horizontal bar chart comparing level vs. delta (change) coefficients.

    For each feature group, two bars are drawn: one for the current level and one for
    the change from the prior assessment. Significant effects (p < 0.05) are colored,
    non-significant effects are grey.

    Parameters
    ----------
    cph : fitted CoxPHFitter
    feature_pairs : list of (level_col, delta_col) tuples
        As returned by build_level_vs_change_cox_df.
    title : str, optional
    """
    

    summary = cph.summary

    level_cols = [p[0] for p in feature_pairs if p[0] in summary.index]
    delta_cols = [p[1] for p in feature_pairs if p[1] in summary.index]

    level_summary = summary.loc[level_cols]
    delta_summary = summary.loc[delta_cols]

    # Align by position
    n = min(len(level_summary), len(delta_summary))
    level_summary = level_summary.iloc[:n]
    delta_summary = delta_summary.iloc[:n]

    labels = [c.replace('_', ' ') for c in level_cols[:n]]
    y = np.arange(n)
    bar_h = 0.35

    def _bar_color(coef, p):
        if p >= 0.05:
            return '#d3d3d3'
        return '#f4a6a6' if coef > 0 else '#a8ddb5'

    level_colors = [_bar_color(r['coef'], r['p']) for _, r in level_summary.iterrows()]
    delta_colors = [_bar_color(r['coef'], r['p']) for _, r in delta_summary.iterrows()]

    fig, ax = plt.subplots(figsize=(10, 0.6 * n + 2))

    ax.barh(y + bar_h / 2, level_summary['coef'], bar_h, color=level_colors,
            label='Level (current score)', zorder=2)
    ax.barh(y - bar_h / 2, delta_summary['coef'], bar_h, color=delta_colors,
            label='Delta (change from prior)', zorder=2, hatch='//', edgecolor='black', linewidth=0.5)

    # Error bars
    for i, (_, row) in enumerate(level_summary.iterrows()):
        ax.errorbar(row['coef'], y[i] + bar_h / 2,
                    xerr=[[row['coef'] - row['coef lower 95%']],
                          [row['coef upper 95%'] - row['coef']]],
                    fmt='none', ecolor='black', capsize=2, zorder=3)
    for i, (_, row) in enumerate(delta_summary.iterrows()):
        ax.errorbar(row['coef'], y[i] - bar_h / 2,
                    xerr=[[row['coef'] - row['coef lower 95%']],
                          [row['coef upper 95%'] - row['coef']]],
                    fmt='none', ecolor='black', capsize=2, zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.axvline(0, color='black', linestyle='--')
    ax.set_xlabel('Log Hazard Ratio')
    ax.set_title(title or 'Level vs. Change: Log Hazard Ratios')
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.show()


def _resolve_profile(feature_cols, scores):
    """
    Build a column-keyed score dict from a user-supplied {item_name: value} dict,
    using case-insensitive substring matching against feature_cols.

    Returns (resolved_dict, unmatched_keys).
    """
    resolved = {}
    unmatched = []
    for key, val in scores.items():
        candidates = [c for c in feature_cols if key.lower() in c.lower()]
        if not candidates:
            unmatched.append(key)
            continue
        exact = [c for c in candidates if c.lower() == key.lower()]
        col = exact[0] if exact else candidates[0]
        if not exact and len(candidates) > 1:
            print(f"Ambiguous match for '{key}': {candidates}. Using '{col}'.")
        resolved[col] = float(val)
    return resolved, unmatched


def plot_profile_risk(cph_drill, drill_df, incident_cat, profiles, incident_display, time_horizon=90):
    """
    Plot cumulative incident risk curves for low, medium, and high risk profiles.

    Each profile is a dict of CANS item name → score value. Keys are matched
    case-insensitively as substrings of column names. All other covariates are
    held at the median of drill_df.

    Parameters
    ----------
    cph_drill : fitted CoxPHFitter
        The drill-down Cox model (individual CANS items).
    drill_df : pd.DataFrame
        Dataframe used to fit cph_drill (provides median covariate baseline).
    incident_cat : str
        Incident category name (e.g. 'Abuse_CPS_Report').
    profiles : dict
        Mapping of risk tier label → score dict, e.g.::

            {
                'Low':    {'Depression': 0, 'Anxiety': 0},
                'Medium': {'Depression': 1, 'Anxiety': 1},
                'High':   {'Depression': 3, 'Anxiety': 2},
            }

        Any number of tiers is accepted; canonical names 'Low', 'Medium', 'High'
        receive green/orange/red coloring automatically.
    incident_display : str
        Display name for the incident type used in plot title.
    time_horizon : int
        Day to mark on the plot with a vertical dashed line (default: 90).
    """
    _TIER_COLORS = {'low': '#2ca02c', 'medium': '#ff7f0e', 'high': '#d62728'}
    _FALLBACK_COLORS = ['steelblue', 'purple', 'brown', 'teal', 'olive']

    exclude = {f'{incident_cat}_event', 'start_days', 'OptionsNumber', f'{incident_cat}_stop_days'}
    feature_cols = [c for c in drill_df.columns if c not in exclude]
    duration_col = f'{incident_cat}_stop_days'
    t_max = int(drill_df[duration_col].max())
    times = list(range(0, t_max + 1))

    baseline = drill_df[feature_cols].median().to_frame().T.reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    all_max_risks = []
    fallback_idx = 0

    for tier_label, scores in profiles.items():
        color = _TIER_COLORS.get(tier_label.lower())
        if color is None:
            color = _FALLBACK_COLORS[fallback_idx % len(_FALLBACK_COLORS)]
            fallback_idx += 1

        resolved, unmatched = _resolve_profile(feature_cols, scores)
        if unmatched:
            print(f"[{tier_label}] Warning: no column match found for: {unmatched}")

        profile = baseline.copy()
        for col, val in resolved.items():
            profile[col] = val

        x_max = min(time_horizon, t_max)
        sf = cph_drill.predict_survival_function(profile, times=times)
        cum_risk_pct = (1 - sf.values.flatten()) * 100
        visible_risk = cum_risk_pct[:x_max + 1]
        all_max_risks.append(max(visible_risk))

        score_label = ', '.join(f'{k}={v}' for k, v in scores.items())
        ax.plot(times, cum_risk_pct, color=color, lw=2,
                label=f'{tier_label}  ({score_label})')

    y_max = max(all_max_risks) if all_max_risks else 10
    ax.set_title(f'Cumulative Incident Risk by Profile — {incident_display}', fontsize=12)
    ax.set_xlabel('Days', fontsize=11)
    ax.set_ylabel('Cumulative Incident Probability (%)', fontsize=11)
    x_max = min(time_horizon, t_max)
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max * 1.15 + 0.2)
    ax.legend(fontsize=9, loc='upper left')
    plt.tight_layout()
    plt.show()


def plot_90day_probability(cph_drill, drill_df, incident_cat, top_items, incident_display):
    """Bar chart of predicted 90-day incident probability for each item in top_items, one figure per item."""
    baseline = drill_df.drop(columns=[
        f'{incident_cat}_event', 'start_days', 'OptionsNumber', f'{incident_cat}_stop_days'
    ]).median().to_frame().T

    score_values = [0, 1, 2, 3]

    for item in top_items:
        probs_90 = []
        for v in score_values:
            row = baseline.copy()
            row[item] = v
            s = cph_drill.predict_survival_function(row, times=[90]).values[0][0]
            probs_90.append(1 - s)

        fig, ax = plt.subplots(figsize=(6, 4))
        bar_colors = ['#f4a6a6' if p == max(probs_90) else '#aec6e8' for p in probs_90]
        bars = ax.bar([str(v) for v in score_values], probs_90,
                      color=bar_colors, edgecolor='black', linewidth=0.7)

        for bar, prob in zip(bars, probs_90):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(probs_90) * 0.01,
                f'{prob:.3f}',
                ha='center', va='bottom', fontsize=10
            )

        ax.set_xlabel(item)
        ax.set_ylabel('Predicted Probability')
        ax.set_title(f'90-Day {incident_display} Probability\nby {item} Score')
        ax.set_ylim(0, max(probs_90) * 1.2)
        plt.tight_layout()
        plt.show()
