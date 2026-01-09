# MATLAB folder

This folder contains the MATLAB analysis pipeline. The key design choice is that **all paths and model lists are configured in one place**:

- `config_paths.m` (create by copying `config_paths_template.m`)

## Quick start

1) Create config:

- Copy `config_paths_template.m` → `config_paths.m`
- Edit at least:
  - `cfg.models_root`
  - `cfg.models_seg` (for Step 2 segmentation generation)
  - any of the `cfg.models_*` tables you need for your paper runs

2) Run:

```matlab
cd matlab
setup
run_all
```

## Entry scripts (run these)

### Step 2 — Build alignment files for Praat

- `main_compute_seg_files_from_alignment.m`

Converts the alignment export (produced by your FastSpeech2 inference) into:

- `TEST%05d_seg.csv` (TAB-separated)

These `*_seg.csv` files are required by the Praat script in `praat/`.

Configuration:
- `cfg.models_seg` : rows `{key, model_output_dir, model_type}`
- `cfg.alignment_csv_filename` : the alignment export filename inside each model folder
- `cfg.stats_phon_corpus_path` : optional `.mat` file to compute z-scores (if missing, z-scores are 0)

### Step 4 — Bias by layer (main analyses)

- `main_correlate_embeddings_with_acoustics_by_layer.m`
  - Computes correlation/regression between intermediate embeddings (by layer) and **continuous** acoustic parameters.
- `main_predict_phonetic_categories_by_layer.m`
  - Predicts **categorical** phonetic events (pauses, liaisons) from intermediate embeddings (by layer).

### Step 6 — Controllability evaluation (optional)

- `main_evaluate_controllability_by_layer.m`
- `main_compute_categorical_bias_effect_by_layer.m`

## Helper scripts

Most other `.m` files in this folder are helpers (loading `.mat` intermediate states, reading acoustic CSVs, plotting, regressions, etc.). They are called by the entry scripts above.

## Notes

- Some scripts expect the per-utterance files produced in Steps 1–3 (see repo `README.md`).
- If you see any remaining absolute paths in your local fork, remove them and rely on `setup` + `config_paths.m`.
