# Python folder (FastSpeech2 patch guidance)

This repo **does not** vendor a full TTS codebase. We used a fork of FastSpeech2:

- https://github.com/ming024/FastSpeech2

This folder provides:
- a reference implementation of the **EmbeddingBias** modules (Step 5)
- **snippets** showing where we modified FastSpeech2 to export intermediate states (Step 1)

## Step 1 — Export intermediate states + durations

### 1.1 Capture encoder/decoder states by layer

In your FastSpeech2 fork:

- `transformer/Models.py`
  - add logic to accumulate `enc_output_by_layer` / `dec_output_by_layer`
  - gate it with a config flag (e.g. `save_embeddings_by_layer`) and `not self.training`

See `FastSpeech2/transformer/Models.py`.

### 1.2 Forward them through the model output

- `model/fastspeech2.py`
  - forward `enc_output_by_layer` / `dec_output_by_layer` and return them during inference

See `FastSpeech2/model/fastspeech2.py`.

### 1.3 Export `.mat` files (per utterance)

For each utterance `TEST%05d`, write:

- `TEST%05d_syn_enc_emb_by_layer.mat` containing variable `enc_output_by_layer_mat`
- `TEST%05d_syn_dec_emb_by_layer.mat` containing variable `dec_output_by_layer_mat`

MATLAB expects shapes:

- `[D, T_enc, L]` for encoder
- `[D, T_dec, L]` for decoder

If your tensors are `[L, B, T, D]`, a typical conversion for a single utterance is:

- select batch index `b=0`
- permute to `[D, T, L]`

### 1.4 Export an alignment/duration file for Step 2

Step 2 (MATLAB) needs an alignment export to build `TEST%05d_seg.csv`.

In our internal pipeline, we exported a single file per model folder, with `|` delimiter and 5 string columns, and used:
- column 4: `duration_mat` (space-separated numbers)
- column 2: `log_duration_mat` (space-separated numbers)
- column 5: `align` string (space-separated characters/phon labels)

Because this format is codebase-specific, **adapt `matlab/main_compute_seg_files_from_alignment.m` if your export differs**.

The expected filename is set in `matlab/config_paths.m`:
- `cfg.alignment_csv_filename`

## Step 5 — Add EmbeddingBias modules (control)

We provide a minimal, codebase-agnostic implementation:

- `embedding_bias.py`
  - `EmbeddingBias`
  - `EmbeddingBiasCategorical`

These modules inject a trainable bias into intermediate embeddings (optionally at specific layers and for specific token patterns).

See `FastSpeech2/model/modules.py` for the original context in our experiments.

### Integration points

You can apply the bias:
- to encoder outputs by layer
- to decoder outputs by layer
- only at selected layers (e.g., layer 7)
- only on token spans matching a pattern (categorical version)

Because tokenization differs across TTS systems, you must implement the pattern-to-index mapping in your codebase.

## What to cite / what to copy

- **Copy**: `embedding_bias.py` into your TTS repository.
- **Use as guidance**: the snippet files in `FastSpeech2/` (they show the hook points, but should be adapted to your framework and export format).
