#!/usr/bin/env python3
"""Tables 1-2 and Supplementary Tables S1-S23, rebuilt from the deposited result tables.

Each builder returns the table as displayed in the manuscript: rows, columns,
rounding and units match the printed version. Every table is written as CSV
under ``output/tables/`` and all of them are collected in ``output/tables/tables.md``.
Percentages have one decimal place unless the manuscript prints otherwise.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import pandas as pd

from repro.paths import RESULTS_DIR, TABLES_DIR, VALIDATION_DIR, require

GROUPS = ("G1", "G2")
COMMUNITY = {"G1": "Visible", "G2": "Hidden", "G2-G1": "Hidden - visible"}


@dataclass
class Table:
    key: str
    title: str
    sources: tuple[str, ...]
    frame: pd.DataFrame
    note: str = ""


# Formatting ---------------------------------------------------------------

def pct(x: float, digits: int = 1) -> str:
    return f"{100 * x:.{digits}f}"


def ci(low: float, high: float, digits: int = 1, scale: float = 100) -> str:
    return f"[{scale * low:.{digits}f}, {scale * high:.{digits}f}]"


def n(x: float) -> str:
    return f"{int(round(x)):,}"


def count_share(num: float, den: float) -> str:
    return f"{n(num)}/{n(den)} ({pct(num / den)})"


def floor(x: float) -> str:
    return str(int(math.floor(x)))


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(require(RESULTS_DIR / name), keep_default_na=False, na_values=[""])


def by(frame: pd.DataFrame, **match: object) -> pd.Series:
    """Return the single row matching every column=value pair."""
    mask = pd.Series(True, index=frame.index)
    for column, value in match.items():
        mask &= frame[column].eq(value)
    rows = frame.loc[mask]
    if len(rows) != 1:
        raise ValueError(f"expected one row for {match}, found {len(rows)}")
    return rows.iloc[0]


# Main text ----------------------------------------------------------------

def table_1() -> Table:
    design = read("design_activity_summary.csv").set_index("group")
    flow = read("study_sample_flow.csv").set_index(["group", "stage"])["n"]
    cells = read("rq3_interaction_cells.csv").query("variant == 'primary'").set_index("cell")
    rows = {
        "Participants": lambda g: n(design.loc[g, "humans_total"]),
        "AI accounts": lambda g: n(design.loc[g, "ai_agents_total"]),
        "Root posts (participants / AI accounts)":
            lambda g: f"{n(design.loc[g, 'human_roots'])} / {n(design.loc[g, 'ai_roots'])}",
        "Comments (participants / AI accounts)":
            lambda g: f"{n(design.loc[g, 'human_comments'])} / {n(design.loc[g, 'ai_comments'])}",
        "Reactions (human and AI accounts)": lambda g: n(design.loc[g, "reactions"]),
        "AI targets among analysed participant replies":
            lambda g: pct(cells.loc[f"{g}_AI", "events"]
                          / (cells.loc[f"{g}_AI", "events"] + cells.loc[f"{g}_H", "events"])) + "%",
        "Median active participants per day": lambda g: n(design.loc[g, "median_active_human"]),
        "Median active AI accounts per day": lambda g: n(design.loc[g, "median_active_ai"]),
        "Human reply events entering choice-set construction": lambda g: n(design.loc[g, "human_reply_events"]),
        "Human replies analysed": lambda g: n(design.loc[g, "analysed_human_replies"]),
        "Eligible same-thread candidates": lambda g: n(flow[(g, "choice_candidates")]),
    }
    frame = pd.DataFrame(
        [{"": label, "Visible labels": f("G1"), "Hidden labels": f("G2")} for label, f in rows.items()]
    )
    return Table("table1", "Table 1. Community composition and activity across 21 experiment days",
                 ("design_activity_summary.csv", "study_sample_flow.csv", "rq3_interaction_cells.csv"), frame)


def table_2() -> Table:
    crit = read("rq3_sol_criticism_estimands.csv").set_index("estimand")
    crit_cells = read("rq3_sol_criticism_cells.csv").set_index("cell")
    tox = read("rq3_detoxify_toxicity.csv")
    down = read("rq3_interaction_cells.csv").query("variant == 'primary'").set_index("cell")
    down_est = read("rq3_interaction_estimands.csv").query("variant == 'primary'").set_index("estimand")
    order = [("G1", "AI"), ("G1", "H"), ("G2", "AI"), ("G2", "H")]
    columns = ["Visible / AI", "Visible / participant", "Hidden / AI", "Hidden / participant"]

    def row(label: str, f: Callable[[str, str], str]) -> dict[str, str]:
        return {"Outcome": label, **{c: f(g, t) for c, (g, t) in zip(columns, order)}}

    def toxic(g: str, t: str) -> str:
        r = by(tox, group=g, dyad="H->AI" if t == "AI" else "H->H")
        return count_share(float(r["toxic_n"]), float(r["n_texts"]))

    panel_a = pd.DataFrame([
        row("Criticism: modelled probability, % [95% CrI]",
            lambda g, t: f"{pct(crit.loc[f'{g}_{t}', 'estimate'])} "
                         f"{ci(crit.loc[f'{g}_{t}', 'ci_low'], crit.loc[f'{g}_{t}', 'ci_high'])}"),
        row("Criticism: positive / eligible replies (%)",
            lambda g, t: count_share(crit_cells.loc[f"{g}_{t}", "positive"], crit_cells.loc[f"{g}_{t}", "observations"])),
        row("Toxicity: positive / eligible replies (%)", toxic),
        row("Other-participant downvotes: positive / eligible reply events (%)",
            lambda g, t: count_share(down.loc[f"{g}_{t}", "positives"], down.loc[f"{g}_{t}", "events"])),
    ])
    c = crit.loc["interaction_visible_minus_hidden_AI_gap"]
    d = down_est.loc["interaction_AI_gap_minus_H_gap"]
    gap = "Gap difference, pp [95% CrI]"
    panel_b = pd.DataFrame([
        {"Outcome": "Criticism, primary", gap: f"{pct(c['estimate'])} {ci(c['ci_low'], c['ci_high'])}"},
        {"Outcome": "Downvotes, primary", gap: f"{pct(d['estimate'])} {ci(d['ci_low'], d['ci_high'])}"},
    ])
    frame = pd.concat([panel_a.assign(Panel="a. Outcomes by community and target authorship"),
                       panel_b.assign(Panel="b. Visible-minus-hidden differences in the AI-participant gap")],
                      ignore_index=True).fillna("")
    frame = frame[["Panel", *[col for col in frame.columns if col != "Panel"]]]
    return Table("table2", "Table 2. Human reception of AI- and participant-authored contributions",
                 ("rq3_sol_criticism_estimands.csv", "rq3_sol_criticism_cells.csv", "rq3_detoxify_toxicity.csv",
                  "rq3_interaction_cells.csv", "rq3_interaction_estimands.csv"),
                 frame, "Panel b: (visible AI - visible participant) - (hidden AI - hidden participant), "
                        "in percentage points.")


# Supplementary: RQ1 -------------------------------------------------------

def table_s1() -> Table:
    est = read("rq1_model_estimands.csv").query("model == 'M1'")
    spec = [
        ("Observed community sets", "observed_risk_set_ai_choice_probability", ("G1", "G2")),
        ("Pooled sets (primary)", "common_opportunity_ai_choice_probability", ("G1", "G2")),
        ("Pooled sets (primary)", "G2_minus_G1_common_opportunity_ai_choice_probability", ("G2-G1",)),
        ("Authorship term only", "direct_G2xAI_common_opportunity_probability", ("G1", "G2", "G2-G1")),
        ("Mixed candidates", "mixed_risk_set_common_opportunity_ai_choice_probability", ("G1", "G2", "G2-G1")),
        ("Equal participant weight", "equal_participant_mixed_risk_set_ai_choice_probability", ("G1", "G2", "G2-G1")),
    ]
    rows = []
    for label, estimand, groups in spec:
        for g in groups:
            r = by(est, estimand=estimand, group=g)
            rows.append({"Reference / projection": label, "Community": COMMUNITY[g],
                         "Mean": pct(r["posterior_mean"]), "95% CrI": ci(r["ci_low"], r["ci_high"])})
    return Table("tableS1", "Table S1. RQ1 standardized AI-target selection probabilities and contrasts",
                 ("rq1_model_estimands.csv",), pd.DataFrame(rows),
                 "Community rows are percentages; hidden-minus-visible rows are percentage points.")


def table_s2() -> Table:
    fwd = read("rq1_forward_validation_comparisons.csv")
    frame = pd.DataFrame({
        "Fold": fwd["fold"].astype(str),
        "Held-out replies": fwd["test_choice_sets"].map(n),
        "Delta ELPD": fwd["elpd_difference"].map(lambda x: f"{x:.1f}"),
        "s.e.": fwd["elpd_difference_se"].map(lambda x: f"{x:.1f}"),
        "Clusters": fwd["uncertainty_clusters"].map(n),
    })
    return Table("tableS2", "Table S2. RQ1 rolling-origin predictive comparison",
                 ("rq1_forward_validation_comparisons.csv",), frame, "M1 minus M0 in summed held-out ELPD.")


def _diagnostics(frame: pd.DataFrame, labels: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame({
        "Fit": list(labels),
        "R-hat max": frame["max_rhat"].map(lambda x: f"{x:.4f}").tolist(),
        "Bulk ESS": frame["min_ess_bulk"].map(floor).tolist(),
        "Tail ESS": frame["min_ess_tail"].map(floor).tolist(),
        "Divergences": frame["divergences"].map(n).tolist(),
    })


def table_s3() -> Table:
    diag = read("rq1_model_diagnostics.csv")
    labels = []
    for _, r in diag.iterrows():
        if r["role"] == "rq1_main":
            labels.append("M1 primary")
        elif r["role"] == "rq1_doubled_prior_scales":
            labels.append("M1 doubled priors")
        else:
            labels.append(f"{r['model']} fold {int(float(r['fold']))}")
    panel_a = _diagnostics(diag, labels)
    prior = read("rq1_prior_scale_sensitivity.csv")
    panel_b = pd.DataFrame([{
        "Community": COMMUNITY[r["group"]],
        "Primary mean": pct(r["posterior_mean_primary"]),
        "Primary 95% CrI": ci(r["ci_low_primary"], r["ci_high_primary"]),
        "Doubled mean": pct(r["posterior_mean_doubled_prior"]),
        "Doubled 95% CrI": ci(r["ci_low_doubled_prior"], r["ci_high_doubled_prior"]),
    } for _, r in prior.iterrows()])
    frame = pd.concat([panel_a.assign(Panel="A. Sampler diagnostics"),
                       panel_b.assign(Panel="B. Prior-scale sensitivity")], ignore_index=True).fillna("")
    frame = frame[["Panel", *[c for c in frame.columns if c != "Panel"]]]
    return Table("tableS3", "Table S3. RQ1 sampler diagnostics and doubled-prior sensitivity",
                 ("rq1_model_diagnostics.csv", "rq1_prior_scale_sensitivity.csv"), frame,
                 "ESS values are minima across monitored parameters, rounded down.")


def table_s4() -> Table:
    est = read("rq1b_primary_estimands.csv").query("standardization_scope == 'equal_participant_observed_opportunities'")
    spec = [
        ("Participant target", "direct_thread_entry_probability", "G1", "H"),
        ("AI target", "direct_thread_entry_probability", "G1", "AI"),
        ("Participant target", "direct_thread_entry_probability", "G2", "H"),
        ("AI target", "direct_thread_entry_probability", "G2", "AI"),
        ("AI minus participant", "AI_minus_H_direct_thread_entry_probability", "G1", "AI-H"),
        ("AI minus participant", "AI_minus_H_direct_thread_entry_probability", "G2", "AI-H"),
        ("Participant target contrast", "G2_minus_G1_direct_thread_entry_probability", "G2-G1", "H"),
        ("AI target contrast", "G2_minus_G1_direct_thread_entry_probability", "G2-G1", "AI"),
        ("Difference in AI-participant gap", "G2_minus_G1_change_in_AI_minus_H_gap", "G2-G1", "AI-H"),
    ]
    rows = []
    for label, estimand, g, t in spec:
        r = by(est, estimand=estimand, group=g, author_type=t)
        rows.append({"Panel": "A. Primary equal-participant estimates", "Quantity / analysis": label,
                     "Community": COMMUNITY[g], "Mean / count": pct(r["posterior_mean"]),
                     "95% CrI": ci(r["ci_low"], r["ci_high"])})
    design = read("rq1b_design_sensitivity_estimands.csv").set_index("role")
    prior = by(read("rq1b_prior_sensitivity.csv"), estimand="G2_minus_G1_change_in_AI_minus_H_gap",
               standardization_scope="equal_participant_observed_opportunities")
    sens = [("Primary", design.loc["main", "posterior_mean"], design.loc["main", "ci_low"], design.loc["main", "ci_high"]),
            ("Recorded next page view", *design.loc["closed", ["posterior_mean", "ci_low", "ci_high"]]),
            ("Both author types in feed visit", *design.loc["both_author", ["posterior_mean", "ci_low", "ci_high"]]),
            ("Participant-aggregated model", *design.loc["participant_aggregate", ["posterior_mean", "ci_low", "ci_high"]]),
            ("Doubled priors", prior["doubled_posterior_mean"], prior["doubled_ci_low"], prior["doubled_ci_high"])]
    for label, mean, low, high in sens:
        rows.append({"Panel": "B. Sensitivity of the difference in gaps", "Quantity / analysis": label,
                     "Community": COMMUNITY["G2-G1"], "Mean / count": pct(mean), "95% CrI": ci(low, high)})
    cov = read("rq1b_coverage.csv").set_index("group")
    audit = read("rq1b_endpoint_source_audit.csv").set_index("group")
    for g in GROUPS:
        for label, column in (("Participants expected", "expected_participants"),
                              ("Participants in endpoint", "participants_in_endpoint"),
                              ("Browser sessions in endpoint", "endpoint_sessions"),
                              ("Rendered-card opportunities", "rendered_cards"),
                              ("Direct entries", "direct_opens")):
            rows.append({"Panel": "C. Coverage and independent endpoint matching", "Quantity / analysis": label,
                         "Community": COMMUNITY[g], "Mean / count": n(cov.loc[g, column]), "95% CrI": "--"})
    for g in GROUPS:
        rows.append({"Panel": "C. Coverage and independent endpoint matching", "Quantity / analysis": "Entries matched to source",
                     "Community": COMMUNITY[g],
                     "Mean / count": f"{n(audit.loc[g, 'exact_vendor_matches'])}/{n(audit.loc[g, 'direct_opens'])}",
                     "95% CrI": "--"})
    diag = read("rq1b_model_diagnostics.csv").iloc[0]
    deletion = read("rq1b_participant_deletion_summary.csv").iloc[0]
    note = (f"Probabilities in percent, contrasts in percentage points. Max R-hat {diag['max_rhat']:.4f}; "
            f"min bulk/tail ESS {floor(diag['min_ess_bulk'])}/{floor(diag['min_ess_tail'])}; "
            f"{n(diag['divergences'])} divergences. Single-participant deletions: "
            f"{n(deletion['positive_differences'])} of {n(deletion['deletions'])} differences positive, "
            f"range {pct(deletion['min_difference_in_differences'])}–{pct(deletion['max_difference_in_differences'])} points.")
    return Table("tableS4", "Table S4. Direct thread-entry estimates, sensitivities and coverage",
                 ("rq1b_primary_estimands.csv", "rq1b_design_sensitivity_estimands.csv", "rq1b_prior_sensitivity.csv",
                  "rq1b_coverage.csv", "rq1b_endpoint_source_audit.csv", "rq1b_model_diagnostics.csv",
                  "rq1b_participant_deletion_summary.csv"), pd.DataFrame(rows), note)


def table_s5() -> Table:
    gates = read("browsing_gate_status.csv")
    link = {g: by(gates, metric="pre_action_linkage_rate", group=g) for g in GROUPS}
    closure = {g: by(gates, metric="interval_closure_rate", group=g) for g in GROUPS}
    gap = by(gates, metric="arm_pre_action_linkage_gap", group="all")
    frame = pd.DataFrame([
        {"Criterion": "Pre-action linkage",
         **{f"{COMMUNITY[g]} labels": f"{pct(float(link[g]['value']))}% "
                                      f"({n(float(link[g]['numerator']))}/{n(float(link[g]['denominator']))})" for g in GROUPS},
         "Requirement": "At least 90% in each community"},
        {"Criterion": "Exactly closed appearance intervals",
         **{f"{COMMUNITY[g]} labels": f"{pct(float(closure[g]['value']))}%" for g in GROUPS},
         "Requirement": "At least 90% in each community"},
        {"Criterion": "Between-community linkage difference",
         "Visible labels": f"{pct(float(gap['value']))} percentage points", "Hidden labels": "",
         "Requirement": "At most 5 percentage points"},
    ])
    return Table("tableS5", "Table S5. Browser-telemetry coverage and quality criteria",
                 ("browsing_gate_status.csv",), frame)


def table_s6() -> Table:
    flow = read("study_sample_flow.csv").set_index(["group", "stage"])["n"]
    rq2 = read("rq2_projection_denominators.csv").set_index("group")
    cov = read("rq1b_coverage.csv").set_index("group")
    spec = [
        ("Human and AI contributions", lambda g: flow[(g, "all_contributions")]),
        ("Annotated human comments", lambda g: flow[(g, "annotated_human_comments")]),
        ("Comments without a parent record", lambda g: flow[(g, "comments_missing_parent_record")]),
        ("Linked human replies entering choice construction", lambda g: flow[(g, "linked_human_reply_events")]),
        ("Excluded: fewer than two eligible alternatives",
         lambda g: flow[(g, "choice_excluded_fewer_than_two_eligible_candidates")]),
        ("Excluded: selected target absent after filtering",
         lambda g: flow[(g, "choice_excluded_chosen_target_absent_after_filter")]),
        ("Analysed reply-target choices", lambda g: flow[(g, "choice_replies")]),
        ("Candidate contribution rows", lambda g: flow[(g, "choice_candidates")]),
        ("Direct-parent criticism: self-replies excluded", lambda g: flow[(g, "criticism_excluded_self_reply")]),
        ("Direct-parent criticism: parent unavailable", lambda g: flow[(g, "criticism_excluded_parent_unavailable")]),
        ("Eligible direct-parent criticism replies", lambda g: flow[(g, "criticism_replies")]),
        ("RQ2 reply-by-AI-account projection opportunities", lambda g: rq2.loc[g, "projection_opportunities"]),
        ("Browser root-post opportunities", lambda g: cov.loc[g, "rendered_cards"]),
        ("Direct thread entries", lambda g: cov.loc[g, "direct_opens"]),
        ("Participants in the direct-entry cohort", lambda g: cov.loc[g, "participants_in_endpoint"]),
    ]
    rows = []
    for label, f in spec:
        v, h = float(f("G1")), float(f("G2"))
        rows.append({"Stage or unit": label, "Visible": n(v), "Hidden": n(h), "Total": n(v + h)})
    return Table("tableS6", "Table S6. Endpoint-specific sample flow",
                 ("study_sample_flow.csv", "rq2_projection_denominators.csv", "rq1b_coverage.csv"), pd.DataFrame(rows))


# Supplementary: RQ3 and RQ2 -----------------------------------------------

DYADS = (("H->H", "Participant -> Participant"), ("H->AI", "Participant -> AI"),
         ("AI->H", "AI -> Participant"), ("AI->AI", "AI -> AI"))


def table_s7() -> Table:
    tox = read("rq3_detoxify_toxicity.csv")
    rows = []
    for g in GROUPS:
        for dyad, label in DYADS:
            r = by(tox, group=g, dyad=dyad)
            rows.append({"Community": COMMUNITY[g], "Reply author -> target": label,
                         "Replies": str(int(r["n_texts"])), "Toxic": str(int(r["toxic_n"])),
                         "Percent": pct(float(r["toxic_share"]))})
    test = by(tox, group="G2_vs_G1", dyad="H->AI")
    return Table("tableS7", "Table S7. Toxicity by community and reply dyad", ("rq3_detoxify_toxicity.csv",),
                 pd.DataFrame(rows),
                 f"Detoxify score > 0.5. Human replies to AI targets, between communities: "
                 f"two-proportion z test p = {float(test['p_value']):.3f}.")


def table_s8() -> Table:
    mix = read("rq3_reaction_target_mix_valid.csv").set_index(["group", "actor_type", "target_author_type", "reaction_type"])["reactions"]
    rows = []
    for actor, target in (("H", "AI"), ("H", "H"), ("AI", "AI"), ("AI", "H")):
        row = {"Actor": "Participant" if actor == "H" else "AI", "Target": "Participant" if target == "H" else "AI"}
        for g in GROUPS:
            for kind, label in (("upvote", "Upvotes"), ("downvote", "Downvotes")):
                row[f"{COMMUNITY[g]} {label}"] = n(mix[(g, actor, target, kind)])
        rows.append(row)
    return Table("tableS8", "Table S8. Timestamp-eligible reactions by actor and target authorship",
                 ("rq3_reaction_target_mix_valid.csv",), pd.DataFrame(rows))


def table_s9() -> Table:
    primary = read("rq2_primary_estimands.csv")
    doubled = read("rq2_doubled_prior_estimands.csv")
    rows = []
    for direction, label in (("human_to_ai", "Participant to AI"), ("ai_to_human", "AI to participant")):
        for g in ("G1", "G2", "G2-G1"):
            p = by(primary, direction=direction, group=g)
            d = by(doubled, direction=direction, group=g)
            rows.append({"Panel": "A. Primary and doubled-prior contact contrasts", "1": label, "2": COMMUNITY[g],
                         "3": pct(p["posterior_median"]), "4": ci(p["ci_low"], p["ci_high"]),
                         "5": pct(d["posterior_median"]), "6": ci(d["ci_low"], d["ci_high"])})
    head = lambda panel, *cells: {"Panel": panel, **{str(i + 1): c for i, c in enumerate(cells)}}
    rows.append(head("B. Model and projection denominators", "Community", "Model replies", "Candidates",
                     "Projection replies", "Opportunities"))
    for _, r in read("rq2_projection_denominators.csv").iterrows():
        rows.append(head("B. Model and projection denominators", {"ALL": "All"}.get(r["group"], COMMUNITY.get(r["group"])),
                         n(r["modelled_replies"]), n(r["candidate_contributions"]),
                         n(r["projection_replies"]), n(r["projection_opportunities"])))
    rows.append(head("C. Rolling-origin predictive comparison", "Fold", "Held-out replies", "Delta ELPD", "s.e.", "Clusters"))
    for _, r in read("rq2_admission_forward_comparisons.csv").iterrows():
        rows.append(head("C. Rolling-origin predictive comparison", str(r["fold"]), n(r["test_choice_sets"]),
                         f"{r['elpd_difference']:.1f}", f"{r['elpd_difference_se']:.1f}", n(r["uncertainty_clusters"])))
    rows.append(head("D. Sampler diagnostics", "Fit", "R-hat max", "Bulk ESS", "Tail ESS", "Divergences"))
    fits = [("RQ2 primary", read("rq2_model_diagnostics.csv").iloc[0]),
            ("RQ2 doubled priors", read("rq2_sensitivity_model_diagnostics.csv").iloc[0])]
    fits += [(f"{r['model']} fold {r['fold']}", r) for _, r in read("rq2_admission_forward_diagnostics.csv").iterrows()]
    for label, r in fits:
        rows.append(head("D. Sampler diagnostics", label, f"{r['max_rhat']:.4f}", floor(r["min_ess_bulk"]),
                         floor(r["min_ess_tail"]), n(r["divergences"])))
    rows.append(head("E. Exact deletion refits", "Deleted unit", "Refits", "R-hat max", "Min. bulk ESS",
                     "Min. tail ESS", "Class changes"))
    refits = read("rq2_admission_exact_refits.csv")
    for unit, part in refits.groupby("influence_unit", sort=False):
        rows.append(head("E. Exact deletion refits", unit.capitalize(), n(len(part)), f"{part['max_rhat'].max():.4f}",
                         floor(part["min_ess_bulk"].min()), floor(part["min_ess_tail"].min()),
                         n(part["qualitative_class_changes"].sum())))
    frame = pd.DataFrame(rows).fillna("")
    frame.columns = ["Panel", "", "", "", "", "", ""]
    return Table("tableS9", "Table S9. Directional-contact estimates, support and validation",
                 ("rq2_primary_estimands.csv", "rq2_doubled_prior_estimands.csv", "rq2_projection_denominators.csv",
                  "rq2_admission_forward_comparisons.csv", "rq2_model_diagnostics.csv",
                  "rq2_sensitivity_model_diagnostics.csv", "rq2_admission_forward_diagnostics.csv",
                  "rq2_admission_exact_refits.csv"), frame,
                 "Panel A: columns are prior direction, community, primary median, 95% CrI, doubled-prior median, "
                 f"95% CrI, in percentage points. Panel E: {len(refits)} refits, "
                 f"{n(refits['divergences'].sum())} divergences in total.")


def table_s10() -> Table:
    feeds = read("study_rss_feeds.csv").rename(columns={"category": "Category", "name": "Source", "feed_url": "Feed URL"})
    feeds["Category"] = feeds["Category"].str.replace("_", " ")
    return Table("tableS10", "Table S10. Configured RSS feeds shared by the two communities",
                 ("study_rss_feeds.csv",), feeds, f"{len(feeds)} feed entries.")


def table_s11() -> Table:
    persona = read("study_ai_persona_summary.csv").set_index(["group", "feature", "level"])["value"]

    def v(g: str, feature: str, level: str) -> str:
        return n(persona[(g, feature, level)])

    rows = [
        ("Male / female personas", lambda g: f"{v(g, 'gender', 'male')} / {v(g, 'gender', 'female')}"),
        ("High-school / bachelor's education",
         lambda g: f"{v(g, 'education_level', 'high school')} / {v(g, 'education_level', 'bachelor')}"),
        *[(f"Toxicity disposition: {level}", lambda g, level=level: v(g, "toxicity", level))
          for level in ("absolutely no", "no", "moderately", "extremely")],
        ("Age range", lambda g: f"{v(g, 'age', 'min')}–{v(g, 'age', 'max')}"),
    ]
    frame = pd.DataFrame([{"Attribute": label, "Visible (n=40)": f("G1"), "Hidden (n=32)": f("G2")} for label, f in rows])
    return Table("tableS11", "Table S11. Selected AI persona attributes by community",
                 ("study_ai_persona_summary.csv",), frame)


def table_s12() -> Table:
    metrics = pd.read_csv(require(VALIDATION_DIR / "llm_majority_metrics.csv")).set_index("task")

    def share(x: float) -> str:
        text = pct(x)
        return ("100" if text == "100.0" else text) + "%"

    spec = [
        ("Validation items", lambda r: n(r["n_items"])),
        ("Human Krippendorff alpha", lambda r: f"{r['krippendorff_alpha']:.3f}"),
        ("LLM-majority Cohen kappa", lambda r: f"{r['kappa_llm_majority']:.3f}"),
        ("Accuracy", lambda r: share(r["accuracy"])),
        ("Positive-class precision", lambda r: share(r["precision_yes"])),
        ("Positive-class recall", lambda r: share(r["recall_yes"])),
        ("Positive-class F1", lambda r: f"{r['f1_yes']:.3f}"),
        ("False positives / false negatives", lambda r: f"{n(r['fp'])} / {n(r['fn'])}"),
    ]
    frame = pd.DataFrame([{"Measure": label, "Criticism": f(metrics.loc["criticism"]),
                           "AI-identity references": f(metrics.loc["ai_identity"])} for label, f in spec])
    return Table("tableS12", "Table S12. Human agreement and language-model performance against majority human labels",
                 ("07_annotations/human_validation/llm_majority_metrics.csv",), frame)


def table_s13() -> Table:
    summary = read("rq3_group_summary.csv")
    spec = [
        ("Any participant", "Strict after reply, 24 h", "rq1_analysis_set", "round_start", "target_received_human_downvote_24h"),
        ("Any participant", "Includes reply hour, 24 h", "rq1_analysis_set", "same_round_inclusive", "target_received_human_downvote_24h"),
        ("Replier", "Includes reply hour, 24 h", "rq1_analysis_set", "same_round_inclusive", "target_replier_downvote_24h"),
        ("Other participant", "Includes reply hour, 24 h", "rq1_analysis_set", "same_round_inclusive", "target_other_downvote_24h"),
        ("Any participant", "Strict after reply, complete 24 h", "fully_observed_window", "round_start", "target_received_human_downvote_24h"),
        ("Other participant", "Includes reply hour, complete 24 h", "fully_observed_window", "same_round_inclusive", "target_other_downvote_24h"),
    ]
    rows = []
    for endpoint, timing, subset, placement, metric in spec:
        row = {"Endpoint": endpoint, "Timing and cohort": timing}
        for g in GROUPS:
            r = by(summary, subset=subset, reaction_placement=placement, metric=metric, group=g, dyad="H->AI")
            row[COMMUNITY[g]] = f"{n(r['numerator'])}/{n(r['denominator_n'])}"
        rows.append(row)
    complete_h = {g: by(summary, subset="fully_observed_window", reaction_placement="round_start",
                        metric="target_other_downvote_24h", group=g, dyad="H->H")["denominator_n"] for g in GROUPS}
    return Table("tableS13", "Table S13. Downvote outcomes among reply events selecting AI-authored contributions",
                 ("rq3_group_summary.csv",), pd.DataFrame(rows),
                 f"Complete 24-hour follow-up also retained {n(complete_h['G1'])} and {n(complete_h['G2'])} "
                 "participant-target reply events in the visible- and hidden-label communities.")


def table_s14() -> Table:
    s = read("rq1_candidate_support.csv").set_index("group")
    total = lambda c: n(s[c].sum())
    rows = [("Reply events", "reply_events"), ("Mixed-author-type sets", "mixed_sets"),
            ("Human-only sets", "human_only_sets"), ("AI-only sets", "ai_only_sets"),
            ("Candidate contributions, distinct", "candidate_contributions"),
            ("Selected target contributions, distinct", "selected_target_contributions"),
            ("Root threads", "root_threads"), ("Accounts in the full population", "all_accounts"),
            ("Accounts appearing among candidates", "candidate_accounts"),
            ("Accounts selected at least once", "selected_accounts")]
    frame = [{"Measure": label, "Visible": n(s.loc["G1", c]), "Hidden": n(s.loc["G2", c]), "Total": total(c)}
             for label, c in rows]
    iqr = lambda g: f"{n(s.loc[g, 'candidate_n_median'])} ({n(s.loc[g, 'candidate_n_q25'])}–{n(s.loc[g, 'candidate_n_q75'])})"
    rng = lambda g: f"{n(s.loc[g, 'candidate_n_min'])}–{n(s.loc[g, 'candidate_n_max'])}"
    frame.append({"Measure": "Candidate-count median (IQR)", "Visible": iqr("G1"), "Hidden": iqr("G2"), "Total": ""})
    frame.append({"Measure": "Candidate-count range", "Visible": rng("G1"), "Hidden": rng("G2"),
                  "Total": f"{n(s['candidate_n_min'].min())}–{n(s['candidate_n_max'].max())}"})
    return Table("tableS14", "Table S14. Support in the retained same-thread candidate sets",
                 ("rq1_candidate_support.csv",), pd.DataFrame(frame),
                 "The pooled median (IQR) printed in the manuscript requires reply-level candidate counts, "
                 "which the deposit does not include; the Total cell is left empty.")


BROWSE = {"llm": "Language-model selection", "tree_dfs": "depth-first order"}


def table_s16() -> Table:
    settings = read("study_ai_settings.csv")
    wide = settings.pivot(index="feature", columns="group", values="value")
    shared = wide.loc[wide["G1"] == wide["G2"], "G1"]
    s = shared.get
    rows = [
        ("Feed ranking", s("feed_ranking")),
        ("Configured action likelihoods",
         f"Read {s('action_read')}; comment {s('action_comment')}; post {s('action_post')}; "
         f"share link {s('action_share_link')}; share image {s('action_share_image')}"),
        ("Maximum replies per round", s("max_replies_per_round")),
        ("Reply cooldown", f"{s('reply_cooldown_rounds')} round"),
        ("Thread browsing", f"{BROWSE.get(s('thread_browse_mode'), s('thread_browse_mode'))}; "
                            f"{BROWSE.get(s('thread_browse_order'), s('thread_browse_order'))}"),
        ("Maximum traversed nodes / chunk size",
         f"{s('thread_browse_max_nodes')} / {s('thread_browse_chunk_size')} contributions"),
        ("Candidate shortlist / maximum LLM steps", f"{s('thread_browse_top_k')} / {s('thread_browse_max_llm_steps')}"),
        ("Candidate snippet length", f"{s('thread_browse_snippet_chars')} characters"),
        ("Memory retrieval", f"Up to {s('memory_search_k')} items"),
        ("Total memory character budget", n(float(s("memory_total_max_chars")))),
        ("Social-card decay / corruption rate",
         f"{s('memory_social_decay_lambda')} / {s('memory_social_corruption_rate')}"),
        ("Social-card resummarization", f"Every {s('memory_social_resummarize_every_events')} events"),
        ("Memory reset on client start", "Disabled" if s("memory_reset_on_start") == "False" else "Enabled"),
    ]
    differing = wide.loc[wide["G1"] != wide["G2"]]
    note = "Settings that differ between communities: " + "; ".join(
        f"{feature} (visible {r['G1']}, hidden {r['G2']})" for feature, r in differing.iterrows())
    return Table("tableS16", "Table S16. Shared configured AI browsing and memory settings",
                 ("study_ai_settings.csv",), pd.DataFrame(rows, columns=["Feature", "Setting in both communities"]), note)


CELLS = (("G1_AI", "Visible / AI"), ("G1_H", "Visible / human"), ("G2_AI", "Hidden / AI"), ("G2_H", "Hidden / human"))


def table_s17() -> Table:
    checks = read("rq3_sol_criticism_predictive_checks.csv")
    metrics = (("criticism_replies", "Criticism replies"), ("threads_with_criticism", "Threads with criticism"),
               ("within_thread_pairs", "Within-thread pairs"))
    rows = []
    for cell, label in CELLS:
        for metric, metric_label in metrics:
            prior = by(checks, cell=cell, metric=metric, phase="prior")
            post = by(checks, cell=cell, metric=metric, phase="posterior")
            interval = lambda r: f"{r['predicted_mean']:.1f} [{r['predicted_low']:.1f}–{r['predicted_high']:.1f}]"
            rows.append({"Community / parent": label, "Statistic": metric_label, "Observed": n(prior["observed"]),
                         "Prior mean [95% interval]": interval(prior),
                         "Posterior mean [95% interval]": interval(post)})
    return Table("tableS17", "Table S17. Prior and posterior predictive checks for the criticism model",
                 ("rq3_sol_criticism_predictive_checks.csv",), pd.DataFrame(rows))


def table_s18() -> Table:
    params = read("rq3_sol_criticism_parameter_summary.csv").set_index("parameter")
    labels = [("alpha", "Intercept", 1), ("beta_visible", "Visible community", 1), ("beta_ai", "AI parent", 1),
              ("beta_interaction", "Visible x AI", 1), ("sigma_participant", "Participant SD", 1),
              ("sigma_context", "Thread SD", 1), ("sigma_target", "Addressed-account SD", 1),
              ("cell_probability[G1_AI]", "Visible / AI (%)", 100), ("cell_probability[G1_H]", "Visible / human (%)", 100),
              ("cell_probability[G2_AI]", "Hidden / AI (%)", 100), ("cell_probability[G2_H]", "Hidden / human (%)", 100),
              ("probability_interaction", "Probability interaction (pp)", 100)]
    rows = []
    for key, label, scale in labels:
        r = params.loc[key]
        rows.append({"Parameter": label, "Mean": f"{scale * r['mean']:.2f}",
                     "95% CrI": f"{scale * r['ci_low']:.2f} to {scale * r['ci_high']:.2f}",
                     "MCSE": f"{scale * r['mcse_mean']:.3f}", "Bulk ESS": n(r["ess_bulk"]).replace(",", ""),
                     "Tail ESS": n(r["ess_tail"]).replace(",", ""), "R-hat": f"{r['r_hat']:.3f}"})
    sens = read("rq3_sol_criticism_sensitivity_estimands.csv").query(
        "estimand == 'interaction_visible_minus_hidden_AI_gap'").set_index("variant")
    for variant, label in (("primary", "Primary"), ("equal_participant", "Equal-participant standardization"),
                           ("doubled_prior", "Doubled prior standard deviations")):
        r = sens.loc[variant]
        rows.append({"Parameter": f"Sensitivity: {label}", "Mean": pct(r["estimate"]),
                     "95% CrI": f"{pct(r['ci_low'])}–{pct(r['ci_high'])}"})
    return Table("tableS18", "Table S18. Criticism-model coefficients, varying-effect scales, standardized probabilities "
                 "and sensitivity analyses", ("rq3_sol_criticism_parameter_summary.csv",
                                              "rq3_sol_criticism_sensitivity_estimands.csv"),
                 pd.DataFrame(rows).fillna(""),
                 "Coefficients and SDs on the log-odds scale. Sensitivity rows: visible-minus-hidden difference "
                 "in the AI-minus-participant criticism gap, percentage points.")


def table_s19() -> Table:
    overlap = read("rq3_annotation_overlap.csv").set_index(["group", "target_type"])
    upper = []
    for cell, label in CELLS:
        g, t = cell.split("_")
        r = overlap.loc[(g, t)]
        upper.append({"Community / parent": label, "N": r["denominator"], "C": r["criticism"], "T": r["toxicity"],
                      "I": r["identity"], "C and T": r["criticism_toxicity"], "C and I": r["criticism_identity"],
                      "T and I": r["toxicity_identity"], "All three": r["all_three"]})
    votes = read("rq3_annotation_parent_downvotes.csv")
    lower = []
    for cell, label in CELLS:
        g, t = cell.split("_")
        for category in ("criticism", "toxicity", "identity"):
            r = by(votes, group=g, target_type=t, category=category)
            lower.append({"Community / parent": label, "Category": category.capitalize(), "N": r["denominator"],
                          "Positive replies": r["category_positive"], "Replier vote": r["replier_downvote"],
                          "Other vote": r["other_downvote"], "Both": r["both_voter_types"]})
    frame = pd.concat([pd.DataFrame(upper).assign(Panel="Upper: reply content overlap"),
                       pd.DataFrame(lower).assign(Panel="Lower: downvotes on the selected parent")],
                      ignore_index=True)
    frame = frame[["Panel", *[c for c in frame.columns if c != "Panel"]]].astype(object).where(frame.notna(), "")
    for column in frame.columns[1:]:
        frame[column] = frame[column].map(lambda x: str(int(x)) if isinstance(x, float) else str(x))
    return Table("tableS19", "Table S19. Overlap between reply content and downvotes on the contribution being answered",
                 ("rq3_annotation_overlap.csv", "rq3_annotation_parent_downvotes.csv"), frame,
                 "C: direct-parent criticism; T: toxicity > 0.5; I: AI-identity reference.")


def table_s20() -> Table:
    a = read("study_activity_contact.csv").set_index("group")
    fmt = lambda x: f"{x:g}"
    spec = [
        ("Human contributions per participant, median", lambda g: fmt(a.loc[g, "human_median"])),
        ("Share from five most active participants (%)", lambda g: pct(a.loc[g, "top5_human_share"])),
        ("Cross-type direct replies", lambda g: n(a.loc[g, "cross_type_replies"])),
        ("Distinct human-AI pairs", lambda g: n(a.loc[g, "pairs"])),
        ("Pairs with replies on at least two dates", lambda g: n(a.loc[g, "pairs_multiple_days"])),
        ("Pairs with replies in at least two threads", lambda g: n(a.loc[g, "pairs_multiple_threads"])),
        ("Pairs with replies in both directions", lambda g: n(a.loc[g, "pairs_both_directions"])),
        ("Human contributions per participant, range",
         lambda g: f"{n(a.loc[g, 'human_min'])}–{n(a.loc[g, 'human_max'])}"),
    ]
    frame = pd.DataFrame([{"Measure": label, "Visible": f("G1"), "Hidden": f("G2")} for label, f in spec])
    return Table("tableS20", "Table S20. Participation and direct contact across selected analytical dates",
                 ("study_activity_contact.csv",), frame)


def table_s21() -> Table:
    attribution = read("study_ai_identity_attribution.csv")
    context = read("study_ai_identity_context.csv").set_index("group")
    counts = attribution.groupby("target")["comments"].sum()
    rows = [{"Measure": "AI-identity reference comments", "Visible": n(context.loc["G1", "comments"]),
             "Hidden": n(context.loc["G2", "comments"])},
            {"Measure": "Proportion of human comments",
             **{COMMUNITY[g]: f"{n(context.loc[g, 'comments'])}/{n(context.loc[g, 'denominator'])} "
                              f"({pct(context.loc[g, 'comments'] / context.loc[g, 'denominator'])})" for g in GROUPS}},
            {"Measure": "Participants", "Visible": n(context.loc["G1", "speakers"]), "Hidden": n(context.loc["G2", "speakers"])},
            {"Measure": "Threads", "Visible": n(context.loc["G1", "threads"]), "Hidden": n(context.loc["G2", "threads"])}]
    for target in ("DIRECT_PARENT", "GENERAL", "OTHER_PREVIOUS", "UNRESOLVED"):
        part = attribution.loc[attribution["target"].eq(target)].set_index("group")["comments"]
        rows.append({"Measure": f"Coded target: {target.lower().replace('_', ' ')}",
                     "Visible": n(part.get("G1", 0)), "Hidden": n(part.get("G2", 0))})
    return Table("tableS21", "Table S21. AI-identity references: counts behind the contextual examples",
                 ("study_ai_identity_context.csv", "study_ai_identity_attribution.csv"), pd.DataFrame(rows),
                 f"{n(counts.sum())} reference comments in total. The quoted examples in Table S21 come from "
                 "`02_platform_activity/posts.csv`; their selection is editorial and is not scripted.")


def table_s23() -> Table:
    t = read("rq3_toxicity_threshold_summary.csv")
    rows = []
    for g in GROUPS:
        for target, label in (("AI", "AI"), ("H", "Participant")):
            r = by(t, group=g, target_author_type=target)
            cell = lambda k: f"{n(r[f'n_gt_{k}'])} ({float(r[f'pct_gt_{k}']):.1f}%)"
            rows.append({"Community": COMMUNITY[g], "Target": label, "n": n(r["n"]),
                         "Median [Q1, Q3]": f"{r['median']:.5f} [{r['q1']:.5f}, {r['q3']:.5f}]",
                         ">0.4": cell("0_4"), ">0.5": cell("0_5"), ">0.6": cell("0_6")})
    ratios = [float(by(t, group=g, target_author_type="AI")["pct_gt_0_5"])
              / float(by(t, group=g, target_author_type="H")["pct_gt_0_5"]) for g in GROUPS]
    return Table("tableS23", "Table S23. Distribution of toxicity scores and sensitivity to the classification threshold",
                 ("rq3_toxicity_threshold_summary.csv",), pd.DataFrame(rows),
                 f"AI/participant toxicity ratio at 0.5: {min(ratios):.1f} to {max(ratios):.1f}. The footnote's "
                 "category shares among toxic replies need reply-level Detoxify scores, which are not deposited.")


BUILDERS: tuple[Callable[[], Table], ...] = (
    table_1, table_2, table_s1, table_s2, table_s3, table_s4, table_s5, table_s6, table_s7, table_s8, table_s9,
    table_s10, table_s11, table_s12, table_s13, table_s14, table_s16, table_s17, table_s18, table_s19, table_s20,
    table_s21, table_s23,
)


def to_markdown(table: Table) -> str:
    frame = table.frame.astype(str)
    header = "| " + " | ".join(frame.columns) + " |"
    rule = "|" + "|".join(["---"] * len(frame.columns)) + "|"
    body = ["| " + " | ".join(v.replace("|", "\\|") for v in row) + " |" for row in frame.itertuples(index=False)]
    parts = [f"## {table.title}", "", header, rule, *body, ""]
    if table.note:
        parts += [table.note, ""]
    parts += ["Source: " + ", ".join(f"`{s}`" for s in table.sources), ""]
    return "\n".join(parts)


def build_all() -> list[Table]:
    require(RESULTS_DIR)
    return [builder() for builder in BUILDERS]


def write(tables: Sequence[Table], out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for table in tables:
        path = out / f"{table.key}.csv"
        table.frame.to_csv(path, index=False)
        paths.append(path)
    summary = out / "tables.md"
    summary.write_text("# Manuscript tables\n\nRebuilt from the Zenodo result tables by `python -m repro.tables`. "
                       "Tables S15 and S22 contain definitions and settings text only and are not rebuilt.\n\n"
                       + "\n".join(to_markdown(t) for t in tables), encoding="utf-8")
    return paths + [summary]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=TABLES_DIR, help=f"output directory (default: {TABLES_DIR})")
    args = parser.parse_args(argv)
    for path in write(build_all(), args.out):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
