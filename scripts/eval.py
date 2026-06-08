#!/usr/bin/env python3
"""Minimal evaluation skeleton for the DPA4 reproduction.

Loads a trained/frozen model and a test system, then reports energy and force
errors (MAE / RMSE). This is a *placeholder* — fill in the TODOs once a real
checkpoint and test set are available.

Usage (example):
    python scripts/eval.py \
        --model frozen_model.pth \
        --system /path/to/test/system

The deepmd-kit API used here:
    - deepmd.infer.DeepPot         : load a frozen model and run inference
    - deepmd.utils.data.DeepmdData : load a system in the DeePMD-kit data format
"""

from __future__ import annotations

import argparse

import numpy as np

# These imports require deepmd-kit to be installed/mounted (see repo README).
from deepmd.infer import DeepPot
from deepmd.utils.data import DeepmdData


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a DPA4 model on a test system.")
    parser.add_argument(
        "--model",
        required=True,
        help="Path to the frozen model (e.g. frozen_model.pth) or checkpoint.",
    )
    parser.add_argument(
        "--system",
        required=True,
        help="Path to a test system in DeePMD-kit data format.",
    )
    parser.add_argument(
        "--ntests",
        type=int,
        default=-1,
        help="Number of test frames to use (-1 = all).",
    )
    return parser.parse_args()


def rmse(pred: np.ndarray, label: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - label) ** 2)))


def mae(pred: np.ndarray, label: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - label)))


def main() -> None:
    args = parse_args()

    # 1. Load the model.
    dp = DeepPot(args.model)
    type_map = dp.get_type_map()

    # 2. Load the test system.
    #    TODO: confirm type_map alignment between the model and the data.
    data = DeepmdData(args.system, type_map=type_map)
    test = data.get_test(ntests=args.ntests)

    coords = test["coord"]              # (nframes, natoms * 3)
    natoms = data.get_natoms()
    nframes = coords.shape[0]
    coords = coords.reshape(nframes, natoms, 3)
    atom_types = test["type"][0]        # (natoms,)
    cells = test.get("box")             # (nframes, 9) or None for nopbc

    # 3. Run inference.
    e_pred, f_pred, _v_pred = dp.eval(coords, cells, atom_types)

    # 4. Compare against labels.
    #    TODO: handle energy bias / per-atom normalization consistently with
    #    how the model was trained (see loss.f_use_norm in configs/dpa4.json).
    e_label = test["energy"].reshape(nframes)
    f_label = test["force"].reshape(nframes, natoms, 3)

    print(f"frames           : {nframes}")
    print(f"atoms / frame    : {natoms}")
    print(f"energy MAE  (eV)        : {mae(e_pred.reshape(-1), e_label):.6e}")
    print(f"energy RMSE (eV)        : {rmse(e_pred.reshape(-1), e_label):.6e}")
    print(f"energy MAE/atom (eV)    : {mae(e_pred.reshape(-1), e_label) / natoms:.6e}")
    print(f"force  MAE  (eV/A)      : {mae(f_pred, f_label):.6e}")
    print(f"force  RMSE (eV/A)      : {rmse(f_pred, f_label):.6e}")

    # TODO: optionally report virial errors, dump predictions to disk, and
    # aggregate across multiple systems.


if __name__ == "__main__":
    main()
