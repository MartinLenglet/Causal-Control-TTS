# Praat folder

This folder contains Praat scripts to compute **per-character acoustic features** for synthesized speech.

The main script used by the paper is:

- `calculate_acoustic_params_syn_cli.praat` (portable wrapper)
  - wraps the original `original/calculate_acoustic_params_syn.praat` but avoids hard-coded paths

## Inputs

For each model output folder (per utterance):

- `TEST%05d_syn.wav`
- `TEST%05d_seg.csv`

The `*_seg.csv` files are produced by **MATLAB Step 2** (`matlab/main_compute_seg_files_from_alignment.m`).

The `*_seg.csv` format must be TAB-separated with header:

```
character    start    end    GTduration    ZScoreGT    ZScoreAlign
```

## Outputs

For each utterance, written into the same model output folder:

- `TEST%05d_acoustic_params.csv`

This file is `|`-separated and contains the acoustic parameters required by the MATLAB analyses.

## Batch run

### macOS / Linux

```bash
praat --run praat/calculate_acoustic_params_syn_cli.praat \
  /abs/path/to/MODELS_ROOT \
  praat/lists/list_folder_praat_script.txt \
  1000 1
```

Arguments:
1) `MODELS_ROOT` : the parent folder that contains the model subfolders listed in the list file
2) list file : one model subfolder per line (relative to `MODELS_ROOT`)
3) number of utterances (e.g., 1000)
4) `is_male` : `1` for male pitch settings, `0` for female

### Windows (PowerShell)

```powershell
& 'C:\Program Files\Praat.exe' --run praat\calculate_acoustic_params_syn_cli.praat `
  'C:\path\to\MODELS_ROOT' `
  'praat\lists\list_folder_praat_script.txt' `
  1000 1
```

## List files

- `lists/list_folder_praat_script.txt` — the default model subfolder list (you can create your own)

## Original scripts

- `original/calculate_acoustic_params_syn.praat` — original script as used in our experiments (kept for reference)
