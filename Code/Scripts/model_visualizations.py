import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

def plot_cox_combined(
    cph_model,
    data,
    time_horizon,
    sort_by="coef",
    figsize_scale=0.5,
    fade_nonsignificant=False
):
    """
    Side-by-side plots:
    Left = log hazard coefficients with CI
    Right = change in incident probability (0 → 2)

    Parameters:
    - cph_model: fitted CoxPHFitter
    - data: dataframe used for fitting
    - time_horizon: time point for probability calc
    """

    summary = cph_model.summary.copy()

    # ---------- Compute probability shifts ----------
    baseline = data.median().to_frame().T
    effects = []

    for var in summary.index:
        if var not in baseline.columns:
            effects.append(None)
            continue

        base = baseline.copy()
        mod = baseline.copy()

        base[var] = 0
        mod[var] = 2

        s_base = cph_model.predict_survival_function(base, times=[time_horizon]).values[0][0]
        s_mod = cph_model.predict_survival_function(mod, times=[time_horizon]).values[0][0]

        effects.append((1 - s_mod) - (1 - s_base))

    summary["delta_prob"] = effects
    summary = summary.dropna(subset=["delta_prob"])

    # ---------- Sorting ----------
    if sort_by == "abs_coef":
        summary["abs_coef"] = summary["coef"].abs()
        summary = summary.sort_values(by="abs_coef")
    else:
        summary = summary.sort_values(by="coef")

    # ---------- Colors ----------
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
        colors = [
            neg_color if coef < 0 else pos_color
            for coef in summary["coef"]
        ]

    # ---------- Figure ----------
    fig, axes = plt.subplots(
        ncols=2,
        figsize=(16, figsize_scale * len(summary)),
        sharey=True
    )

    # ---------- LEFT: Coefs + CI ----------
    ax = axes[0]
    group_names = [name.replace('_', ' ') for name in summary.index.tolist()]

    ax.barh(group_names, summary["coef"], color=colors, zorder=2)

    lower_err = summary["coef"] - summary["coef lower 95%"]
    upper_err = summary["coef upper 95%"] - summary["coef"]

    group_names = [name.replace('_', ' ') for name in summary.index.tolist()]
    ax.errorbar(
        summary["coef"],
        group_names,
        xerr=[lower_err, upper_err],
        fmt='none',
        ecolor='black',
        capsize=3,
        zorder=3
    )

    ax.axvline(0, color='black', linestyle='--')
    ax.set_title("Log Hazard Ratio (coef)")
    ax.set_xlabel("Effect Size")

    # ---------- RIGHT: Probability shift ----------
    ax2 = axes[1]

    prob_colors = [
        neg_color if v < 0 else pos_color
        for v in summary["delta_prob"]
    ]

    ax2.barh(group_names, summary["delta_prob"], color=prob_colors)

    ax2.axvline(0, color='black', linestyle='--')
    ax2.set_title(f"Δ Incident Probability (0 → 2)\n t = {time_horizon}")
    ax2.set_xlabel("Change in Probability")

    # ---------- Layout ----------
    plt.tight_layout()
    plt.show()




def interval_risk_model(hazard_table_plt, features, incident_col, test_size=0.4, random_state=42,
                        C=1.0, max_iter=1000):
    """
    Fits a regularized logistic regression on interval-level data.
    Train/test split is done by individual (OptionsNumber) to control for
    repeated observations.

    Parameters
    ----------
    hazard_table_plt : pd.DataFrame
        Must contain incident_col, OptionsNumber, and the feature columns.
    features : list of str
        Column names to use as predictors.
    incident_col : str
        Name of the binary event column (e.g. 'Abuse_CPS_Report_event').
    test_size : float
        Fraction of unique OptionsNumber to use for test set.
    random_state : int
        Random seed for train/test split.
    C : float
        Inverse regularization strength (smaller = more regularization).
    max_iter : int
        Maximum solver iterations.

    Returns
    -------
    result : LogisticRegression
        Fitted model.
    test_df : pd.DataFrame
        Test set with predicted interval-level risk in 'pred_risk_interval'.
    """
    pp_df = hazard_table_plt.copy()
    pp_df['interval_event'] = pp_df[incident_col]

    # Train/test split by individual to avoid leakage from repeated observations
    unique_options = pp_df['OptionsNumber'].unique()
    train_opts, test_opts = train_test_split(unique_options, test_size=test_size, random_state=random_state)
    train_df = pp_df[pp_df['OptionsNumber'].isin(train_opts)]
    test_df  = pp_df[pp_df['OptionsNumber'].isin(test_opts)].copy()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[features].astype(float))
    X_test  = scaler.transform(test_df[features].astype(float))
    y_train = train_df['interval_event'].astype(float)
    y_test  = test_df['interval_event'].astype(float)

    result = LogisticRegression(C=C, max_iter=max_iter, solver='lbfgs')
    result.fit(X_train, y_train)

    test_df['pred_risk_interval'] = result.predict_proba(X_test)[:, 1]

    auc_interval = roc_auc_score(y_test, test_df['pred_risk_interval'])
    print(f"Interval-level AUC: {auc_interval:.3f}")

    # Histogram of interval-level predicted risk
    plt.figure(figsize=(8, 6))
    plt.hist(
        test_df.loc[test_df['interval_event'] == 0, 'pred_risk_interval'],
        bins=30, density=True, alpha=0.5, color='skyblue', label='No Incident'
    )
    plt.hist(
        test_df.loc[test_df['interval_event'] == 1, 'pred_risk_interval'],
        bins=30, density=True, alpha=0.5, color='salmon', label='Had Incident'
    )
    plt.xlabel('Predicted Interval Risk')
    plt.ylabel('Density')
    plt.title('Distribution of Interval-Level Predicted Risk')
    plt.legend()
    plt.show()

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, test_df['pred_risk_interval'])
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='black', lw=2, label=f'ROC curve (AUC = {auc_interval:.3f})')
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Interval-Level Predictions')
    plt.legend(loc='lower right')
    plt.show()

    return result, test_df



def plot_cross_incident_heatmap(models_dict, feature_cols, display_names=None,
                                 show_significance=True, figsize=None):
    """
    Heatmap of log hazard ratios across CANS group features × incident types.

    Shows which CANS domains are broadly predictive across all incident types vs.
    incident-specific. Cells are colored on a diverging red/green scale (red =
    increased risk, green = protective). Significance stars are overlaid when
    show_significance=True.

    Parameters
    ----------
    models_dict : dict
        Mapping of incident_cat (str) → fitted CoxPHFitter. Models must share
        the same feature_cols.
    feature_cols : list of str
        CANS group feature names to display (rows). Must be present in every model.
    display_names : dict, optional
        Maps incident_cat keys to prettier display strings for column labels.
    show_significance : bool
        If True, overlay '★' on cells with p < 0.05.
    figsize : tuple, optional
        Figure size. Defaults to (len(models_dict) * 1.6, len(feature_cols) * 0.7 + 1).

    Returns
    -------
    coef_df : pd.DataFrame
        (feature_cols × incident_cats) matrix of log hazard ratio coefficients.
    pval_df : pd.DataFrame
        Corresponding p-value matrix.
    """
    display_names = display_names or {}
    incident_cats = list(models_dict.keys())

    coef_data, pval_data = {}, {}
    for cat, cph in models_dict.items():
        available = [c for c in feature_cols if c in cph.summary.index]
        coef_data[cat] = cph.summary.loc[available, 'coef']
        pval_data[cat] = cph.summary.loc[available, 'p']

    coef_df = pd.DataFrame(coef_data, index=feature_cols).fillna(0)
    pval_df = pd.DataFrame(pval_data, index=feature_cols).fillna(1)

    col_labels = [display_names.get(c, c.replace('_', ' ')) for c in coef_df.columns]
    row_labels = [r.replace('_', ' ').replace('/', '/\n') for r in coef_df.index]

    abs_max = coef_df.abs().max().max()
    norm = mcolors.TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)

    if figsize is None:
        figsize = (max(8, len(incident_cats) * 1.6), max(5, len(feature_cols) * 0.7 + 1.5))

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(coef_df.values, cmap='RdYlGn_r', norm=norm, aspect='auto')

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=35, ha='right', fontsize=9)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)

    for r in range(coef_df.shape[0]):
        for c in range(coef_df.shape[1]):
            val = coef_df.iloc[r, c]
            pval = pval_df.iloc[r, c]
            star = ' ★' if (show_significance and pval < 0.05) else ''
            text_color = 'white' if abs(norm(val) - 0.5) > 0.3 else 'black'
            ax.text(c, r, f'{val:.2f}{star}', ha='center', va='center',
                    fontsize=7.5, color=text_color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label('Log Hazard Ratio', fontsize=9)

    ax.set_title(
        'CANS Group Log Hazard Ratios Across Incident Types\n'
        '(★ = p < 0.05, red = increased risk, green = protective)',
        fontsize=11
    )
    plt.tight_layout()
    plt.show()
    return coef_df, pval_df


def plot_concordance_index_bar(concordance_dict, display_names=None, figsize=(9, 4)):
    """
    Bar chart of concordance index (Harrell's C) across incident types.

    Parameters
    ----------
    concordance_dict : dict
        Mapping of incident_cat → concordance_index (float).
    display_names : dict, optional
        Maps incident_cat keys to prettier display strings.
    figsize : tuple
    """
    display_names = display_names or {}
    labels = [display_names.get(k, k.replace('_', ' ')) for k in concordance_dict]
    values = list(concordance_dict.values())
    colors = ['#5b9bd5' if v >= 0.6 else '#aec6e8' for v in values]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(labels, values, color=colors, edgecolor='black', linewidth=0.6)
    ax.axvline(0.5, color='red', linestyle='--', linewidth=1.2, label='Random (0.5)')
    ax.axvline(0.6, color='orange', linestyle=':', linewidth=1.2, label='Acceptable (0.6)')

    for bar, val in zip(bars, values):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', fontsize=9)

    ax.set_xlim(0.4, min(max(values) * 1.1, 1.0))
    ax.set_xlabel("Concordance Index (Harrell's C)", fontsize=10)
    ax.set_title("Cox Model Discrimination by Incident Type", fontsize=11)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_top_partial_effects(cph, top_n=6, values=[0, 1, 2, 3], max_days=1095):
    """
    Plot partial effects of the top N covariates from a fitted CoxPHFitter model.
    
    Parameters
    ----------
    cph : lifelines.CoxPHFitter
        Fitted Cox proportional hazards model.
    top_n : int
        Number of top covariates to plot (by absolute coefficient magnitude).
    values : list
        List of covariate values to plot (e.g., [0,1,2,3]).
    max_days : int
        Maximum x-axis (time) in days to display.
    """
    # Get top N covariates by absolute coefficient
    model_summary = cph.summary.reset_index()
    model_summary['abs_coef'] = model_summary['coef'].abs()
    top_covs = model_summary.sort_values(by='abs_coef', ascending=False).head(top_n)['covariate'].tolist()
    
    # Create subplots
    rows = (top_n + 2) // 3  # 3 columns layout
    fig, axes = plt.subplots(rows, 3, figsize=(18, 6*rows))
    axes = axes.flatten()
    
    # Plot partial effects
    for i, cov in enumerate(top_covs):
        cph.plot_partial_effects_on_outcome(
            covariates=cov,
            values=values,
            ax=axes[i],
            plot_baseline=True
        )
        axes[i].set_title(f'Partial Effect: {cov}')
        axes[i].set_xlim(0, max_days)
    
    # Hide empty subplots if top_n < total axes
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.show()