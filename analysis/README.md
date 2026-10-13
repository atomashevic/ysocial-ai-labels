# Model-fitting code

This folder holds the code that prepared the analysis tables and fitted the Bayesian models reported in the paper. It is copied unchanged from the study's analysis repository at commit `40c425c`, so the script hashes recorded in the run manifests still match. The figures and tables in `../repro/` are built from the posterior summaries this code wrote, which are deposited in the Zenodo record under `08_analysis_ready/results_tables/`.

Run everything from this folder, with the environment in `requirements-fit.txt`:

```bash
cd analysis
pip install -r requirements-fit.txt
python -m scripts.publication_analysis --help
```

Fits used 4 chains with 1,000 warm-up and 1,000 retained draws, target acceptance 0.95 and seed 20260715 (`analysis.yaml`). The full RQ1 stage, which includes M0, M1, the doubled-prior refit, forward-validation folds and deletion refits, took about 16 hours on 4 cores.

## Where each reported model lives

| Paper | Model | Code | Command |
|---|---|---|---|
| RQ1, Fig. 2, Tables S1–S3 | Same-thread reply-target choice (M0, M1) | `scripts/publication_analysis/model.py` (`_build_model`, `fit_model`, `standardized_estimands`); data in `data.py:prepare_choice_data`, terms in `terms.py:build_model_frame` | `prepare-choice`, `fit-choice`, `validate-choice` |
| Direct thread entry, Fig. 3, Tables S4–S5 | Bernoulli-logit entry model | `scripts/publication_analysis/rq1b.py` (`build_direct_thread_entry_panel`, `build_pymc_model`, `build_participant_aggregate_model`) | `python -m scripts.publication_analysis.rq1b` |
| RQ2, Fig. 4, Table S9, Figs. S1–S2 | Directional prior-contact model | `scripts/publication_analysis/rq2.py`, `pipeline.py:run_fit_rq2`; supporting diagnostics in `scripts/diagnostics/rq2_sensitivity.py`, `rq2_contact.py` | `prepare-rq2`, `fit-rq2`, `validate-rq2` |
| RQ3 downvotes, Table 2, Tables S13, S19 | Windowed outcomes and bootstrap contrasts | `scripts/publication_analysis/rq3.py` | `prepare-rq3`, `estimate-rq3`, `validate-rq3` |
| RQ3 downvotes, Table 2b | Exploratory Bayesian interaction | `scripts/diagnostics/rq3_interaction.py` | `python -m scripts.diagnostics.rq3_interaction --panel PANEL --run-id ID` |
| RQ3 criticism, Table 2, Tables S17–S18 | Criticism model and sensitivities | `scripts/diagnostics/annotation_models.py` (`build_model`), panels in `annotation_panels.py` and `sol_criticism.py`, sensitivities in `criticism_sensitivities.py` | `python -m scripts.diagnostics.annotation_models fit --analysis criticism` |
| Sample size, Supplementary Note (retrospective sensitivity to reply-choice differences) | Fixed-scale Laplace refits of the reply-target choice model on simulated choices; 200 datasets per imposed difference. The refits include the topic-varying AI slope, with variance scales from an earlier M1 fit that contained it | `scripts/diagnostics/reply_observed_design.py`; settings in the `reply_observed_design_sensitivity` block of `analysis.yaml` | `python scripts/diagnostics/reply_observed_design.py --config analysis.yaml --root . --output DIR` |
| Annotation | Language-model labelling prompts and runners (local Ollama endpoint) | `scripts/diagnostics/annotate_interlocutors.py`, `annotate_recommendations.py`, `classify_requests.py` | — |

## Inputs

`analysis.yaml` points to the study's processed tables, which are not public in that form. The de-identified versions are in the Zenodo record under `08_analysis_ready/canonical/` (`snapshot_contents.csv`, `snapshot_events.csv`, `snapshot_user_reputation.csv`). Running the preparation code on those tables reproduces the published RQ1 choice sets row for row (2,143 replies, 17,489 candidates, identical attrition), the RQ2 history-support table, and the RQ3 reply panel (2,547 rows, identical cell counts and round timestamps).

Some inputs are not in record version 1.0.0:

- **Thread topics for RQ2.** The RQ2 model includes a topic-varying slope for AI authorship (45 topic levels), read from a legacy panel. Without it, every thread is untagged and the topic effect drops out. The reported RQ1 M1 fit has no topic term, so the record's tables reproduce its input exactly (same 17,489 candidate rows and covariates).
- **Criticism labels for RQ3.** The criticism model uses thread-context labels from a later labelling run. The record contains the earlier labelling (`07_annotations/llm_annotations/interlocutor_measures.csv`) and the human validation sample only.
- **Downvote panel for the exploratory interaction model.** The model reads a reply-event panel written by `estimate-rq3`; it is not deposited.
- **Browser telemetry for direct thread entry.** The record's `05_browser_telemetry/` files use different names and columns from the private session tables the code reads.

The run-orchestration code also checks file hashes and paths of the original runs (`runs/...`). Those checks fail outside the original repository and need to be pointed at new run directories.
