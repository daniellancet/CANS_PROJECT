"""
CANS Incidents Streamlit Dashboard
Tabs: Group Coef Plot | Risk Curves | Drill Down | Profile Builder
"""

import io
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from lifelines import CoxPHFitter


def _drive_service():
    creds = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=["https://www.googleapis.com/auth/drive.readonly"],
    )
    return build("drive", "v3", credentials=creds)


def _download_csv(file_id: str) -> pd.DataFrame:
    service = _drive_service()
    request = service.files().get_media(fileId=file_id)
    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    buf.seek(0)
    return pd.read_csv(buf)

INCIDENT_TYPES = [
    "Abuse/CPS Report",
    "Suicide Related Incidents",
    "Behavioral_Incidents",
    "AWOL/ Child_Absense",
    "Health/Medical_Incidents",
    "Police_Involvement",
]

ALT_TO_GROUP = {
    "alt_Caregiver_Support_Needs_score":         "Caregiver_Support_Needs",
    "alt_Internalizing___Self-Harm_score":        "Internalizing_/_Self-Harm",
    "alt_Developmental_&_Sexual_Concerns_score":  "Developmental_&_Sexual_Concerns",
    "alt_Externalizing_Behavior_score":           "Externalizing_Behavior",
    "alt_Family_&_Social_Functioning_score":      "Family_&_Social_Functioning",
    "alt_Community_&_Strengths_score":            "Community_&_Strengths",
    "alt_Substance_Use_&_Delinquency_score":      "Substance_Use_&_Delinquency",
    "alt_School_Functioning_score":               "School_Functioning",
}
GROUP_TO_ALT = {v: k for k, v in ALT_TO_GROUP.items()}

DEMO_COLS = ["AgeWhenAssessed", "Black", "Latino", "Asian", "Other", "Native"]

NEG_COLOR = "#a8ddb5"
POS_COLOR = "#f4a6a6"
GRAY_COLOR = "#d3d3d3"
TIER_COLORS = {"low": "#2ca02c", "medium": "#ff7f0e", "high": "#d62728"}
FALLBACK_COLORS = ["steelblue", "purple", "brown", "teal", "olive", "crimson"]


def label(col: str) -> str:
    return col.replace("alt_", "").replace("_score", "").replace("_", " ").strip()



@st.cache_data(show_spinner="Loading data…")
def load_data():
    hazard = _download_csv(st.secrets["drive"]["hazard_file_id"])
    cat_map = _download_csv(st.secrets["drive"]["cat_map_file_id"])
    first = hazard[hazard["DateCompleted"] == hazard["origin_assessment"]].copy()
    return hazard, first, cat_map


def _exclude_left_censored(df: pd.DataFrame, incident_type: str) -> tuple[pd.DataFrame, int]:
    """Remove youth who had the incident before their origin assessment."""
    date_col = f"first_incident_date_{incident_type}"
    if date_col not in df.columns:
        return df, 0
    left_censored = df[df[date_col] < df["origin_assessment"]]
    exclude_ids = left_censored["OptionsNumber"].unique()
    return df[~df["OptionsNumber"].isin(exclude_ids)].copy(), len(exclude_ids)


@st.cache_resource(show_spinner="Fitting group model…")
def fit_group_model(incident_type: str):
    _, first, _ = load_data()
    first, n_excluded = _exclude_left_censored(first, incident_type)
    covariates = list(first.filter(like="alt_").columns)
    dur = f"T_{incident_type}"
    evt = f"status_{incident_type}"
    tbl = first[[dur, evt] + covariates].dropna()
    cph = CoxPHFitter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cph.fit(tbl, duration_col=dur, event_col=evt)
    return cph, tbl, covariates, n_excluded


@st.cache_resource(show_spinner="Fitting drill-down model…")
def fit_drill_model(incident_type: str, group_name: str):
    _, first, cat_map = load_data()
    first, _ = _exclude_left_censored(first, incident_type)
    group_items = cat_map.loc[cat_map["group"] == group_name, "variable"].tolist()
    group_items = [v for v in group_items if v in first.columns]
    available_demo = [c for c in DEMO_COLS if c in first.columns]
    dur = f"T_{incident_type}"
    evt = f"status_{incident_type}"
    drill_df = first[[dur, evt] + group_items + available_demo].dropna()

    if drill_df[evt].sum() < 5 or len(group_items) == 0:
        return None, drill_df, group_items

    cph = CoxPHFitter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cph.fit(drill_df, duration_col=dur, event_col=evt)
    return cph, drill_df, group_items



def _bar_colors(summary, fade=True):
    colors = []
    for coef, p in zip(summary["coef"], summary["p"]):
        if fade and p >= 0.05:
            colors.append(GRAY_COLOR)
        elif coef < 0:
            colors.append(NEG_COLOR)
        else:
            colors.append(POS_COLOR)
    return colors


def plot_cox_coefs(cph, exclude_vars=None, title="Log Hazard Ratio"):
    summary = cph.summary.copy()
    if exclude_vars:
        summary = summary.drop(index=[v for v in exclude_vars if v in summary.index])
    summary = summary.sort_values("coef")
    names = [label(n) for n in summary.index]
    colors = _bar_colors(summary)

    fig, ax = plt.subplots(figsize=(9, max(3, 0.55 * len(summary))))
    ax.barh(names, summary["coef"], color=colors, zorder=2)
    lower_err = summary["coef"] - summary["coef lower 95%"]
    upper_err = summary["coef upper 95%"] - summary["coef"]
    ax.errorbar(summary["coef"], names, xerr=[lower_err, upper_err],
                fmt="none", ecolor="black", capsize=3, zorder=3)
    ax.scatter(summary["coef"], names, color="black", zorder=4, s=20)
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Log Hazard Ratio (coef)")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=POS_COLOR, label="Increased risk (p < 0.05)"),
        Patch(facecolor=NEG_COLOR, label="Protective (p < 0.05)"),
        Patch(facecolor=GRAY_COLOR, label="Non-significant"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="lower right")
    plt.tight_layout()
    return fig


def plot_risk_curves(cph, variables, values=(0, 3, 6, 9, 12), xlim=(0, 90)):
    figs = []
    for var in variables:
        try:
            ax = cph.plot_partial_effects_on_outcome(covariates=var, values=list(values), cmap="coolwarm")
        except Exception:
            continue

        for line in ax.lines:
            line.set_ydata(1 - line.get_ydata())

        ax.set_title(f"Risk Curve — {label(var)}", fontsize=10)
        ax.set_ylabel("P(Incident before time T)")
        ax.set_xlabel("Time (days)")
        ax.set_xlim(*xlim)

        y_vals = []
        for line in ax.lines:
            x, y = line.get_xdata(), line.get_ydata()
            mask = (x >= xlim[0]) & (x <= xlim[1])
            y_vals.extend(y[mask][~np.isnan(y[mask])])
        if y_vals:
            y_min, y_max = min(y_vals), max(y_vals)
            pad = max((y_max - y_min) * 0.1, 0.005)
            ax.set_ylim(max(0, y_min - pad), min(1, y_max + pad))

        handles, old_labels = ax.get_legend_handles_labels()
        ax.legend(handles, [label(l) for l in old_labels], fontsize=8)
        fig = ax.get_figure()
        plt.tight_layout()
        figs.append(fig)
    return figs


def plot_90day_bars(cph_drill, items, drill_df, incident_type, dur_col, evt_col):
    figs = []
    score_values = [0, 1, 2, 3]
    feature_cols = [c for c in drill_df.columns if c not in {dur_col, evt_col}]
    baseline = drill_df[feature_cols].median().to_frame().T

    for item in items:
        if item not in cph_drill.summary.index:
            continue
        p_val = cph_drill.summary.loc[item, "p"]
        if p_val >= 0.05:
            continue

        probs = []
        for v in score_values:
            row = baseline.copy()
            row[item] = v
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                s = cph_drill.predict_survival_function(row, times=[90]).values[0][0]
            probs.append(1 - s)

        fig, ax = plt.subplots(figsize=(5, 3.5))
        bar_colors = [POS_COLOR if p == max(probs) else "#aec6e8" for p in probs]
        bars = ax.bar([str(v) for v in score_values], probs, color=bar_colors,
                      edgecolor="black", linewidth=0.6)
        for bar, prob in zip(bars, probs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(probs) * 0.015,
                    f"{prob:.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_xlabel(f"{item} Score")
        ax.set_ylabel("90-Day Risk")
        ax.set_title(f"90-Day {incident_type.replace('_', ' ')}\n{item}  (p={p_val:.4f})",
                     fontsize=9)
        ax.set_ylim(0, max(probs) * 1.25)
        plt.tight_layout()
        figs.append((item, fig))
    return figs


def plot_baseline_hazard(cph):
    bh = cph.baseline_hazard_["baseline hazard"].head(365)
    bin_size = 30
    binned = bh.groupby(bh.index // bin_size).sum()
    day_labels = binned.index * bin_size
    cond_prob = 1 - np.exp(-binned.values)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(day_labels, cond_prob, color="steelblue")
    ax.scatter(day_labels, cond_prob, color="white", edgecolors="steelblue",
               linewidths=1.5, s=60, zorder=5)
    ax.set_xlabel("Month")
    ax.set_ylabel("Monthly Incident Probability (Incident-Free to Date)")
    ax.set_title("Baseline Hazard — First Year")
    ax.set_xticks(day_labels)
    ax.set_xticklabels([i + 1 for i in range(len(day_labels))])
    ax.set_xlim(0, 365)
    plt.tight_layout()
    return fig


def plot_hazard_by_score(cph, cox_table, cans):
    covar = cph.summary.loc[cans, "coef"].idxmax()
    coef = cph.params_[covar]
    bh = cph.baseline_hazard_["baseline hazard"].head(365)
    bin_size = 30
    binned_base = bh.groupby(bh.index // bin_size).sum()
    day_labels = binned_base.index * bin_size

    scores = cox_table[covar]
    vals = sorted({
        int(round(scores.quantile(0.25))),
        int(round(scores.quantile(0.75))),
        int(round(scores.quantile(0.90))),
    })
    clean_name = covar.replace("alt_", "").replace("_score", "").replace("_", " ")

    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ["#2ca02c", "#ff7f0e", "#d62728"]
    for val, color in zip(vals, colors):
        cond_prob = 1 - np.exp(-binned_base.values * np.exp(coef * val))
        ax.plot(day_labels, cond_prob, color=color, label=f"Score = {val}")
        ax.scatter(day_labels, cond_prob, color="white", edgecolors=color,
                   linewidths=1.5, s=60, zorder=5)

    ax.set_xlabel("Month")
    ax.set_ylabel("Monthly Incident Probability (Incident-Free to Date)")
    ax.set_title(f"Hazard Function by {clean_name} Score (Top Predictor)")
    ax.set_xticks(day_labels)
    ax.set_xticklabels([i + 1 for i in range(len(day_labels))])
    ax.set_xlim(0, 365)
    ax.legend()
    plt.tight_layout()
    return fig


def plot_profile_curves(cph_drill, profiles, drill_df, dur_col, evt_col, incident_type):
    feature_cols = [c for c in drill_df.columns if c not in {dur_col, evt_col}]
    baseline = drill_df[feature_cols].median().to_frame().T.reset_index(drop=True)
    t_max = int(drill_df[dur_col].max())
    times = list(range(0, min(t_max + 1, 91)))

    fig, ax = plt.subplots(figsize=(9, 5))
    all_max = []
    fallback_idx = 0
    summary_rows = []

    for name, scores in profiles.items():
        color = TIER_COLORS.get(name.lower())
        if color is None:
            color = FALLBACK_COLORS[fallback_idx % len(FALLBACK_COLORS)]
            fallback_idx += 1

        profile = baseline.copy()
        for col, val in scores.items():
            if col in profile.columns:
                profile[col] = val

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sf = cph_drill.predict_survival_function(profile, times=times)

        cum_risk_pct = (1 - sf.values.flatten()) * 100
        all_max.append(max(cum_risk_pct))
        risk_90 = cum_risk_pct[min(90, len(cum_risk_pct) - 1)]

        score_str = ", ".join(f"{k}={v}" for k, v in scores.items())
        ax.plot(times, cum_risk_pct, color=color, lw=2, label=f"{name}  ({score_str})")
        summary_rows.append({"Profile": name, "90-Day Risk (%)": f"{risk_90:.2f}%"})

    ax.axvline(90, color="grey", linestyle="--", alpha=0.6, label="90-day mark")
    ax.set_title(f"Cumulative Incident Risk by Profile — {incident_type.replace('_', ' ')}", fontsize=12)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Probability of Incident Before Time T (%)")
    ax.set_xlim(0, 90)
    y_max = max(all_max) if all_max else 10
    ax.set_ylim(0, y_max * 1.2 + 0.5)
    ax.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    return fig, pd.DataFrame(summary_rows)


st.set_page_config(
    page_title="CANS Incident Risk Dashboard",
    page_icon="📊",
    layout="wide",
)
st.title("CANS Incident Risk Dashboard")


with st.sidebar:
    st.header("Settings")
    incident_type = st.selectbox("Incident Type", INCIDENT_TYPES,
                                 format_func=lambda x: x.replace("_", " "))
    st.markdown("---")
    st.caption("Models use first CANS assessment per youth (Cox PH, 90-day horizon).")

# Load data
hazard_table, just_first, cat_map = load_data()

dur_col = f"T_{incident_type}"
evt_col = f"status_{incident_type}"

cph_group, group_tbl, covariates, n_left_censored = fit_group_model(incident_type)

# Sidebar stats — derived from fitted model (after left-censoring exclusion)
n_total = len(cph_group.event_observed)
n_events = int(cph_group.event_observed.sum())
with st.sidebar:
    st.metric("Youth (first assessment)", n_total)
    st.metric("Observed events", n_events)
    st.metric("Censoring rate", f"{(1 - n_events / n_total) * 100:.1f}%")


tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Group Coefficients",
    "Risk Curves",
    "Drill Down",
    "Profile Builder",
    "Hazard Functions",
])


with tab1:
    st.subheader(f"CANS Domain Log Hazard Ratios — {incident_type.replace('_', ' ')}")
    st.caption("Green = protective (p < 0.05) · Red = increased risk (p < 0.05) · Gray = non-significant")

    fig_coef = plot_cox_coefs(
        cph_group,
        title=f"Log Hazard Ratio — {incident_type.replace('_', ' ')}",
    )
    st.pyplot(fig_coef)
    plt.close(fig_coef)

    st.markdown("#### Model Summary")
    summary_display = cph_group.summary[["coef", "exp(coef)", "se(coef)",
                                         "coef lower 95%", "coef upper 95%", "p"]].copy()
    summary_display.index = [label(i) for i in summary_display.index]
    summary_display = summary_display.round(4)

    def highlight_p(val):
        if isinstance(val, float) and val < 0.05:
            return "background-color: #fff3cd; font-weight: bold"
        return ""

    st.dataframe(
        summary_display.style.map(highlight_p, subset=["p"]),
        use_container_width=True,
    )
    st.caption(f"Concordance index: {cph_group.concordance_index_:.4f}")



with tab2:
    st.subheader(f"Group-Level Risk Curves — {incident_type.replace('_', ' ')}")
    st.caption("Showing significant CANS domains (p < 0.05). Y-axis = cumulative incident probability.")

    sig_vars = cph_group.summary[cph_group.summary["p"] < 0.05].index.tolist()
    sig_vars = [v for v in sig_vars if v in covariates]

    if not sig_vars:
        st.info("No CANS domains reach p < 0.05 for this incident type.")
    else:
        figs_risk = plot_risk_curves(cph_group, sig_vars, values=(0, 3, 6, 9, 12), xlim=(0, 90))
        cols = st.columns(min(2, len(figs_risk)))
        for i, fig_r in enumerate(figs_risk):
            with cols[i % len(cols)]:
                st.pyplot(fig_r)
                plt.close(fig_r)

    st.markdown("---")
    st.caption("All CANS domain risk curves (regardless of significance):")
    show_all = st.checkbox("Show all domains")
    if show_all:
        figs_all = plot_risk_curves(cph_group, covariates, values=(0, 3, 6, 9, 12), xlim=(0, 90))
        cols2 = st.columns(2)
        for i, fig_r in enumerate(figs_all):
            with cols2[i % 2]:
                st.pyplot(fig_r)
                plt.close(fig_r)



with tab3:
    st.subheader(f"Drill Down — {incident_type.replace('_', ' ')}")

    # Default to highest-coef group
    top_alt = cph_group.summary.loc[covariates, "coef"].idxmax()
    default_group = ALT_TO_GROUP[top_alt]
    all_groups = list(ALT_TO_GROUP.values())

    selected_group = st.selectbox(
        "CANS Group to drill into",
        all_groups,
        index=all_groups.index(default_group),
        format_func=lambda x: x.replace("_", " "),
    )

    cph_drill, drill_df, group_items = fit_drill_model(incident_type, selected_group)

    if cph_drill is None:
        st.warning(f"Insufficient events ({drill_df[evt_col].sum():.0f}) to fit a model for "
                   f"**{selected_group.replace('_', ' ')}** in **{incident_type}**.")
    else:
        demo_in_drill = [c for c in DEMO_COLS if c in drill_df.columns]

        # --- Sub-panel A: Item coef plot
        st.markdown("### Item-Level Coefficient Plot")
        fig_drill_coef = plot_cox_coefs(
            cph_drill,
            exclude_vars=demo_in_drill,
            title=f"{selected_group.replace('_', ' ')} — Item Log Hazard Ratios ({incident_type.replace('_', ' ')})",
        )
        st.pyplot(fig_drill_coef)
        plt.close(fig_drill_coef)

        # --- Sub-panel B: Risk curves for top 3 items
        st.markdown("### Risk Curves — Top 3 Items by |coef|")
        item_coefs_in_model = [i for i in group_items if i in cph_drill.summary.index]
        top3 = (cph_drill.summary.loc[item_coefs_in_model, "coef"]
                .abs().nlargest(3).index.tolist())

        figs_drill_risk = plot_risk_curves(cph_drill, top3, values=(0, 1, 2, 3), xlim=(0, 90))
        if figs_drill_risk:
            cols3 = st.columns(min(3, len(figs_drill_risk)))
            for i, fig_r in enumerate(figs_drill_risk):
                with cols3[i % len(cols3)]:
                    st.pyplot(fig_r)
                    plt.close(fig_r)
        else:
            st.info("No risk curves could be generated for the top items.")

        # --- Sub-panel C: 90-day bar charts
        st.markdown("### 90-Day Incident Probability by Item Score")
        bar_figs = plot_90day_bars(
            cph_drill, item_coefs_in_model, drill_df, incident_type, dur_col, evt_col
        )
        if not bar_figs:
            st.info("No items reach p < 0.05 significance for 90-day bar charts.")
        else:
            bar_cols = st.columns(min(3, len(bar_figs)))
            for i, (item_name, fig_b) in enumerate(bar_figs):
                with bar_cols[i % len(bar_cols)]:
                    st.pyplot(fig_b)
                    plt.close(fig_b)


# ===========================================================================
# Tab 4 — Profile Builder
# ===========================================================================
with tab4:
    st.subheader(f"Profile Risk Builder — {incident_type.replace('_', ' ')}")
    st.caption("Set CANS scores, add a named profile, then compare cumulative risk curves.")

    # Get the drill model for the top-risk group (for profile predictions)
    top_alt_pb = cph_group.summary.loc[covariates, "coef"].idxmax()
    top_group_pb = ALT_TO_GROUP[top_alt_pb]
    cph_pb, drill_df_pb, group_items_pb = fit_drill_model(incident_type, top_group_pb)

    if cph_pb is None:
        st.warning("Cannot build profiles: insufficient events for the top-risk group model.")
    else:
        dur_pb = f"T_{incident_type}"
        evt_pb = f"status_{incident_type}"
        feature_cols_pb = [c for c in drill_df_pb.columns if c not in {dur_pb, evt_pb}]

        st.info(f"Profiles are scored over **{top_group_pb.replace('_', ' ')}** items "
                f"(top-risk group for this incident type). All other features held at median.")

        # Session state for stored profiles
        if "profiles" not in st.session_state:
            st.session_state.profiles = {}

        col_sliders, col_plot = st.columns([1, 2])

        with col_sliders:
            st.markdown("#### Set CANS Scores")
            current_scores = {}

            # Group sliders by CANS domain
            for grp_name, items in cat_map.groupby("group")["variable"].apply(list).items():
                items_in_model = [i for i in items if i in feature_cols_pb]
                if not items_in_model:
                    continue
                with st.expander(grp_name.replace("_", " "), expanded=(grp_name == top_group_pb)):
                    for item in items_in_model:
                        val = st.slider(
                            item.replace("_", " "),
                            min_value=0, max_value=3, value=1,
                            key=f"slider_{incident_type}_{item}",
                        )
                        current_scores[item] = val

            st.markdown("---")
            profile_name = st.text_input("Profile name", value="Profile 1",
                                         key=f"pname_{incident_type}")
            add_col, clear_col = st.columns(2)
            with add_col:
                if st.button("Add Profile", type="primary"):
                    if len(st.session_state.profiles) >= 5:
                        st.warning("Maximum 5 profiles. Remove one first.")
                    else:
                        st.session_state.profiles[profile_name] = dict(current_scores)
                        st.success(f"Added '{profile_name}'")
            with clear_col:
                if st.button("Clear All"):
                    st.session_state.profiles = {}

            if st.session_state.profiles:
                st.markdown("**Saved profiles:**")
                for pname in list(st.session_state.profiles.keys()):
                    remove_col, label_col = st.columns([1, 4])
                    with remove_col:
                        if st.button("✕", key=f"rm_{pname}"):
                            del st.session_state.profiles[pname]
                            st.rerun()
                    with label_col:
                        st.caption(pname)

        with col_plot:
            if not st.session_state.profiles:
                st.info("Add at least one profile to see risk curves.")
            else:
                fig_profiles, summary_tbl = plot_profile_curves(
                    cph_pb,
                    st.session_state.profiles,
                    drill_df_pb,
                    dur_pb, evt_pb,
                    incident_type,
                )
                st.pyplot(fig_profiles)
                plt.close(fig_profiles)

                st.markdown("#### 90-Day Risk Summary")
                st.dataframe(summary_tbl, use_container_width=True, hide_index=True)


with tab5:
    st.subheader(f"Hazard Functions — {incident_type.replace('_', ' ')}")

    st.markdown("#### Baseline Hazard")
    st.caption(
        "Monthly conditional incident probability for a youth with median CANS scores, "
        "assuming they have been incident-free to date."
    )
    fig_bh = plot_baseline_hazard(cph_group)
    st.pyplot(fig_bh)
    plt.close(fig_bh)

    st.markdown("---")
    st.markdown("#### Hazard by Top Predictor Score")
    top_covar = cph_group.summary.loc[covariates, "coef"].idxmax()
    st.caption(
        f"Top predictor: **{label(top_covar)}**. "
        "Curves show how shifting that domain score (at the 25th, 75th, and 90th percentile "
        "of observed values) scales the baseline hazard up or down proportionally."
    )
    fig_hs = plot_hazard_by_score(cph_group, group_tbl, covariates)
    st.pyplot(fig_hs)
    plt.close(fig_hs)
