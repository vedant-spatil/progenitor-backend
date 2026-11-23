"""Phase 6 evaluation: TrialGPT cohorts, frozen trial snapshot, three systems, metrics, report.

Cost controls (owner decision 2026-11-18):
- one LLM response cache shared by all systems (parse and judge results are reused, not re-bought)
- each trial is parsed once and reused across patients
- a fixed candidate pool per patient from the TrialGPT qrels (no live search)
- free-text patients (short prompts)
- a frozen snapshot + the cache make re-runs free (`--replay`)
"""
