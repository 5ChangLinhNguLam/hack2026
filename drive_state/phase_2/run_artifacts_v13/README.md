# Retained v13 run artifacts

This directory keeps model-run evidence copied with the phase-2 pipeline.

- `evaluation_report.json` and `T01-Sample.csv` through `T06-Sample.csv` are
  the original six-sample practice evaluation artifacts.
- `redacted_10_trips/verification_report.json` and `T01d.csv` through
  `T10d.csv` are the complete 18,000-frame verification run produced on a T4
  using `hack2026.tripkit.TripReplayer` and this v13 model bundle.

Paths inside the archived JSON identify the machine on which the run was made
and are retained for provenance. Use `drive_state.phase_2.cli.replay` to create
fresh reports with paths from the current machine.
