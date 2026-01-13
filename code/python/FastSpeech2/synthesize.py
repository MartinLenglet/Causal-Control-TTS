"""FastSpeech2 inference export helpers (Step 1.3 / Step 1.4).

This repository only vendors *snippets* of a FastSpeech2 fork (see `code/python/README.md`).
Use this script as an **example** of how to export the artifacts needed by the MATLAB
pipeline from the output of `FastSpeech2.forward()` implemented in
`code/python/FastSpeech2/model/fastspeech2.py`.

It demonstrates how to export, per utterance:

Step 1.3
  - `*_syn_enc_emb_by_layer.mat`  (variable: `enc_output_by_layer_mat`, shape [D, T_enc, L])
  - `*_syn_dec_emb_by_layer.mat`  (variable: `dec_output_by_layer_mat`, shape [D, T_dec, L])

Step 1.4
  - `*_duration.mat`              (variable: `duration_mat`, shape [T_enc])
  - `*_log_duration.mat`          (variable: `log_duration_mat`, shape [T_enc])

Optional extras in this example:
  - `*_mel_postnet.mat`           (variable: `mel_postnet_mat`, shape [n_mel, T_mel])

Forward output tuple index map (see `model/fastspeech2.py` return statement):
  - output[1]  : postnet mel            [B, T_mel, n_mel]
  - output[4]  : log_d_predictions      [B, T_src]
  - output[5]  : d_rounded (durations)  [B, T_src]
  - output[8]  : src_lens               [B]
  - output[9]  : mel_lens               [B]
  - output[10] : by-layer list
      - [0] enc_output_by_layer         [L, B, T_src, D]
      - [1] dec_output_by_layer         [L, B, T_mel, D]

Note: this script intentionally does *not* implement full text preprocessing / vocoding.
Plug it into your inference loop right after you call the model.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple
from text.symbols import out_symbols

import numpy as np
import torch
from scipy.io import savemat

def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def _to_numpy(x: Any) -> np.ndarray:
    """Detach Torch tensors (CPU) to numpy; pass numpy arrays through."""
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def _float_list_to_str(x: np.ndarray) -> str:
    x = np.asarray(x).reshape(-1)
    return " ".join(f"{float(v):.6f}" for v in x.tolist())

def _int_list_to_str(x: np.ndarray) -> str:
    x = np.asarray(x).reshape(-1)
    return " ".join(str(int(v)) for v in x.tolist())

def _phon_outputs_to_tokens(phon_outputs: Any, src_len: int, batch_index: int = 0) -> list[str]:
    """
    Convert phon_outputs [B, C, T_src] into a list of tokens length src_len.
    Uses out_symbols for id->token mapping.
    """
    po = _to_numpy(phon_outputs)  # -> np array
    # po shape: [B, C, T]
    ids = np.argmax(po[batch_index, :, :src_len], axis=0)  # [T]
    tokens = [out_symbols[int(i)] for i in ids.tolist()]
    return tokens

def append_alignment_row(
    alignment_path: str | Path,
    log_d: np.ndarray,
    dur: np.ndarray,
    align_tokens: list[str],
    col1: str = "",
    col3: str = "",
    delimiter: str = "|",
) -> Path:
    """
    Write one row in the exact 5-column format the MATLAB script expects:
      col2: log_duration_mat (space-separated)
      col4: duration_mat     (space-separated)
      col5: align tokens     (space-separated)
    """
    alignment_path = Path(alignment_path)
    alignment_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure lengths match (MATLAB indexes log_duration_mat(i_char) for each token)
    n = min(len(align_tokens), len(log_d), len(dur))
    align_str = " ".join(align_tokens[:n])

    row = delimiter.join([
        col1,
        _float_list_to_str(log_d[:n]),
        col3,
        _int_list_to_str(dur[:n]),
        align_str,
    ])

    with alignment_path.open("a", encoding="utf-8") as f:
        f.write(row + "\n")

    return alignment_path

def export_from_forward(
    forward_out: Sequence[Any],
    out_dir: str | Path,
    basename: str,
    batch_index: int = 0,
    export_by_layer: bool = True,
    export_durations: bool = True,
    export_postnet_mel: bool = True,
    alignment_csv_path: str | Path | None = None,
    export_alignment_csv: bool = False,
) -> Dict[str, Path]:
    """Export `.mat` files from a FastSpeech2 forward output.

    Args:
        forward_out: tuple returned by `FastSpeech2.forward()`.
        out_dir: directory where files will be written.
        basename: utterance base name (e.g., "TEST00001").
        batch_index: which item in the batch to export.
        export_by_layer: writes encoder/decoder embeddings by layer (Step 1.3).
        export_durations: writes durations and log-durations (Step 1.4).
        export_postnet_mel: writes postnet mel as `.mat` (handy for debugging).

    Returns:
        Dict mapping logical artifact names to output paths.
    """
    out_dir = Path(out_dir)
    _ensure_dir(out_dir)

    created: Dict[str, Path] = {}

    # Lengths
    src_lens = _to_numpy(forward_out[8]).astype(int)
    mel_lens = _to_numpy(forward_out[9]).astype(int)
    src_len = int(src_lens[batch_index])
    mel_len = int(mel_lens[batch_index])

    # ---- Step 1.4: durations
    if export_durations:
        log_d = _to_numpy(forward_out[4])[batch_index, :src_len].astype(np.float32)
        dur = _to_numpy(forward_out[5])[batch_index, :src_len].astype(np.float32)

        p = out_dir / f"{basename}_log_duration.mat"
        savemat(str(p), {"log_duration_mat": log_d})
        created["log_duration_mat"] = p

        p = out_dir / f"{basename}_duration.mat"
        savemat(str(p), {"duration_mat": dur})
        created["duration_mat"] = p

        # Step 1.4 alignment CSV row (one file per model folder)
        if export_alignment_csv:
            if alignment_csv_path is None:
                raise ValueError("export_alignment_csv=True requires alignment_csv_path.")
            phon_outputs = forward_out[13]
            tokens = _phon_outputs_to_tokens(phon_outputs, src_len=src_len, batch_index=batch_index)

            p = append_alignment_row(
                alignment_csv_path=alignment_csv_path,
                log_d=log_d,
                dur=dur,
                align_tokens=tokens,
            )
            created["alignment_csv"] = p

    # ---- Step 1.3: embeddings by layer
    if export_by_layer:
        by_layer = forward_out[10]
        enc_by_layer = by_layer[0]  # [L, B, T_src, D]
        dec_by_layer = by_layer[1]  # [L, B, T_mel, D]

        enc = _to_numpy(enc_by_layer)[:, batch_index, :src_len, :]  # [L, T, D]
        dec = _to_numpy(dec_by_layer)[:, batch_index, :mel_len, :]  # [L, T, D]

        # MATLAB expects [D, T, L]
        enc_matlab = np.transpose(enc, (2, 1, 0)).astype(np.float32, copy=False)
        dec_matlab = np.transpose(dec, (2, 1, 0)).astype(np.float32, copy=False)

        p = out_dir / f"{basename}_syn_enc_emb_by_layer.mat"
        savemat(str(p), {"enc_output_by_layer_mat": enc_matlab})
        created["enc_output_by_layer_mat"] = p

        p = out_dir / f"{basename}_syn_dec_emb_by_layer.mat"
        savemat(str(p), {"dec_output_by_layer_mat": dec_matlab})
        created["dec_output_by_layer_mat"] = p

    # Optional: postnet mel
    if export_postnet_mel:
        postnet_mel = _to_numpy(forward_out[1])[batch_index, :mel_len, :]  # [T, n_mel]
        mel_matlab = np.transpose(postnet_mel, (1, 0)).astype(np.float32, copy=False)  # [n_mel, T]
        p = out_dir / f"{basename}_mel_postnet.mat"
        savemat(str(p), {"mel_postnet_mat": mel_matlab})
        created["mel_postnet_mat"] = p

    return created


def _cli_load_pt(path: Path) -> Tuple[Sequence[Any], str]:
    """Load a torch-saved forward output for quick testing.

    Expected formats:
      - a tuple/list returned by the model
      - or a dict with keys: {"forward_out": <tuple/list>, "basename": <str>}
    """
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, (tuple, list)):
        return obj, path.stem
    if isinstance(obj, dict) and "forward_out" in obj:
        return obj["forward_out"], str(obj.get("basename", path.stem))
    raise ValueError(
        f"Unsupported .pt contents. Got type={type(obj)}; expected tuple/list or dict with 'forward_out'."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export MATLAB artifacts (Step 1.3/1.4) from a FastSpeech2 forward output. "
            "This is an example helper; integrate `export_from_forward` into your own inference loop."
        )
    )
    parser.add_argument("--forward-pt", type=str, help="Path to a torch-saved forward output (.pt)")
    parser.add_argument("--out-dir", type=str, required=True, help="Directory to write exports")
    parser.add_argument("--basename", type=str, default=None, help="Override utterance basename")
    parser.add_argument("--batch-index", type=int, default=0, help="Which batch item to export")
    parser.add_argument("--no-by-layer", action="store_true", help="Disable Step 1.3 exports")
    parser.add_argument("--no-durations", action="store_true", help="Disable Step 1.4 exports")
    parser.add_argument("--no-mel", action="store_true", help="Disable postnet mel export")
    args = parser.parse_args()

    if not args.forward_pt:
        raise SystemExit(
            "This example CLI requires --forward-pt. In real usage, import this module and call "
            "export_from_forward(model(...), ...)."
        )

    forward_out, inferred_basename = _cli_load_pt(Path(args.forward_pt))
    basename = args.basename or inferred_basename

    created = export_from_forward(
        forward_out,
        out_dir=args.out_dir,
        basename=basename,
        batch_index=args.batch_index,
        export_by_layer=not args.no_by_layer,
        export_durations=not args.no_durations,
        export_postnet_mel=not args.no_mel,
    )

    for k, p in created.items():
        print(f"{k}: {p}")


if __name__ == "__main__":
    main()
