# Causal-Control-TTS — Reproducibility Code

This repository provides the analysis pipeline used in the paper, across **three stages**:

- **Python (external, FastSpeech2 fork)**: synthesize audio, export *intermediate encoder/decoder embeddings* by layer, and export *character durations / alignments*.
- **MATLAB (this repo)**: build `*_seg.csv` alignment files, analyze intermediate spaces, and compute bias-by-layer metrics.
- **Praat (this repo)**: compute per-character acoustic features from synthesized audio using the `*_seg.csv` alignments.

## Repository structure

- `python/` — reference modules and patch instructions for FastSpeech2 (intermediate states + embedding bias).
  - `results_step_1/` — **example output of Step 1 (Python)** for a single utterance, provided as a reference to validate file formats, naming conventions, and data shapes.
- `matlab/` — all MATLAB scripts (entry scripts + helpers). **All model paths are configured in one file:** `matlab/config_paths.m`.
- `praat/` — Praat scripts + list files to compute acoustic parameters.

## End-to-end reproduction procedure

### Step 0 — Naming conventions

For each model condition you analyze, create a **model output folder**. This folder will ultimately contain (per utterance):

- `TEST00001_syn.wav` … synthesized waveform
- `TEST00001_syn_enc_emb_by_layer.mat` with variable `enc_output_by_layer_mat` (shape `[D, T_enc, L]`)
- `TEST00001_syn_dec_emb_by_layer.mat` with variable `dec_output_by_layer_mat` (shape `[D, T_dec, L]`)
- `TEST00001_seg.csv` (from MATLAB, Step 2)
- `TEST00001_acoustic_params.csv` (from Praat, Step 3)

---

### Step 1 — (Python) synthesize audio baseline & export intermediate states + durations

This repo does **not** vendor FastSpeech2 or Tacotron2. Instead, patches to [FastSpeech2 repository](https://github.com/ming024/FastSpeech2) are described as code examples in the folder `python/`:
- capture encoder/decoder hidden states by layer in `transformer/Models.py`
- forward them through `model/fastspeech2.py`
- export the per-utterance `.mat` files expected by MATLAB
- export a single alignment/duration CSV per model folder (used in Step 2)

See **`python/README.md`**.

Output of Step 1 (per model output folder):
- `TEST%05d_syn.wav`
- `TEST%05d_syn_enc_emb_by_layer.mat`
- `TEST%05d_syn_dec_emb_by_layer.mat`
- `<alignment csv>` (filename set by `cfg.alignment_csv_filename` in Step 2)

An **example of the expected output of Step 1** (single utterance, fully compliant with the pipeline) is provided in:
- `python/results_step_1/`

This example can be used to:
- inspect the expected `.mat` variable names and shapes,
- verify alignment/duration export format,
- run **Step 2 (MATLAB)** and **Step 3 (Praat)** without first running FastSpeech2.

---

### Step 2 — (MATLAB) generate `*_seg.csv` alignment files

Run the MATLAB script that converts the alignment export into per-utterance segmentation files:

1. Copy and edit the single config:
   - `matlab/config_paths_template.m` → `matlab/config_paths.m`
   - Set `cfg.models_seg` to your model output folders
   - Set `cfg.alignment_csv_filename` to the filename you exported in Step 1

2. Run:
```matlab
cd matlab
setup
main_compute_seg_files_from_alignment
```

This writes `TEST%05d_seg.csv` into each model output folder.

Details: **`matlab/README.md`**

---

### Step 3 — (Praat) compute per-character acoustic features

Use the Praat script to create `TEST%05d_acoustic_params.csv` in each model output folder.

```bash
praat --run praat/calculate_acoustic_params_syn_cli.praat \
  /abs/path/to/MODELS_ROOT \
  praat/lists/list_folder_praat_script.txt \
  1000 1
```

Details: **`praat/README.md`**

---

### Step 4 — (MATLAB) analyze intermediate spaces and acoustic correlates

Entry scripts (bias by layer):

- **Continuous acoustic feature bias** (correlations): `matlab/main_correlate_embeddings_with_acoustics_by_layer.m`
- **Categorical bias** (pauses & liaisons): `matlab/main_predict_phonetic_categories_by_layer.m`

Run all main analyses in order:
```matlab
cd matlab
setup
run_all
```

---

### Step 5 — (Python / FastSpeech2) add EmbeddingBias modules for control

Add `EmbeddingBias` / `EmbeddingBiasCategorical` to your TTS model to control acoustic features by layer and by feature.

See **`python/README.md`**.

---

### Step 6 — (Optional MATLAB) evaluate controllability

- `matlab/main_evaluate_controllability_by_layer.m`
- `matlab/main_compute_categorical_bias_effect_by_layer.m`

## Outputs

By default, MATLAB writes figures/results under `results/` (configurable in `matlab/config_paths.m` via `cfg.results_root`).

## Citation

If you use this code, please cite the corresponding paper.
