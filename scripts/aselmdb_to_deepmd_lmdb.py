#!/usr/bin/env python3
"""Convert an ASE/fairchem ``.aselmdb`` dataset (e.g. MPtrj) into a deepmd-kit
native LMDB dataset that ``dp --pt train`` can read directly.

Why this exists
---------------
deepmd-kit's LMDB loader (``deepmd.dpmodel.utils.lmdb_data``) only recognizes a
path ending in ``.lmdb`` or containing ``data.mdb``, and its frame schema is a
msgpack layout with DeePMD key conventions (``coords/cells/energies/forces/
atom_types/virials`` encoded as ``{type, shape, data}`` arrays). The MPtrj
``.aselmdb`` files are the *ASE* schema instead: integer string keys whose
values are ``zlib``-compressed ASE-db JSON rows (``numbers/positions/cell/
energy/forces/stress`` with ``{"__ndarray__": [...]}`` arrays). The two are not
interchangeable, so we transcode here using only ``lmdb + zlib + json + numpy +
msgpack`` (no ASE / pymatgen dependency).

Output schema (matches deepmd's reader exactly)
-----------------------------------------------
* ``__metadata__`` (msgpack): ``nframes``, ``frame_idx_fmt="012d"``,
  ``system_info={}``, ``frame_nlocs=[...]``, ``frame_system_ids=[...]``,
  ``type_map=[...]``.
* each frame key ``"%012d" % i`` -> msgpack dict with ``coords (N,3) f8``,
  ``cells (3,3) f8``, ``energies (1,) f8``, ``forces (N,3) f8``,
  ``atom_types (N,) i8`` (indices into ``TYPE_MAP``), and optionally
  ``virials (3,3) f8``.

Type map
--------
A fixed Z-ordered element list (H..Pu, Z=1..94) is used so atom_types are simply
``Z-1``. MPtrj's ~89 elements are a subset; the same list MUST appear as
``model.type_map`` in the training config (see ``configs/dpa4_mptrj.json``).

Virial convention
-----------------
deepmd ``virial`` carries energy units. We follow the dpdata/VASP convention
``virial = -stress * volume`` (stress assumed eV/A^3, as stored in these rows).
If you only want energy+force training, pass ``--no-virial`` and set the virial
loss prefactors to 0 in the config.

Usage
-----
    python scripts/aselmdb_to_deepmd_lmdb.py \
        --src /mnt/afs/share/dataset/periodicSystem/MPtrj/aselmdb/train \
        --dst /mnt/afs/share/dataset/periodicSystem/MPtrj/deepmd_lmdb/train.lmdb

    python scripts/aselmdb_to_deepmd_lmdb.py \
        --src /mnt/afs/share/dataset/periodicSystem/MPtrj/aselmdb/val \
        --dst /mnt/afs/share/dataset/periodicSystem/MPtrj/deepmd_lmdb/val.lmdb
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import zlib
from pathlib import Path

import lmdb
import msgpack
import numpy as np

# Z-ordered element symbols, Z = 1..94 (H..Pu). Superset of MPtrj's elements.
# Keep this identical to model.type_map in the training config.
TYPE_MAP = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu",
]
ASE_SKIP_KEYS = {b"metadata", b"nextid", b"deleted_ids"}


def _ase_array(val):
    """Decode an ASE-db value that may be a plain scalar/list or an
    ``{"__ndarray__": [shape, dtype, flat_data]}`` dict."""
    if isinstance(val, dict) and "__ndarray__" in val:
        shape, dtype, data = val["__ndarray__"]
        return np.asarray(data, dtype=dtype).reshape(shape)
    return np.asarray(val)


def _enc(arr: np.ndarray) -> dict:
    """Encode an ndarray into deepmd's msgpack array dict {type, shape, data}."""
    arr = np.ascontiguousarray(arr)
    return {"type": arr.dtype.str, "shape": list(arr.shape), "data": arr.tobytes()}


def _iter_ase_rows(path: str):
    """Yield decoded ASE-db rows (dicts) from one ``.aselmdb`` file."""
    env = lmdb.open(path, readonly=True, lock=False, subdir=False, max_readers=128)
    try:
        with env.begin() as txn:
            for key, raw in txn.cursor():
                if key in ASE_SKIP_KEYS:
                    continue
                yield json.loads(zlib.decompress(raw))
    finally:
        env.close()


def convert(src: str, dst: str, *, with_virial: bool = True,
            map_size: int = 8 * 1024**4) -> None:
    src_files = sorted(glob.glob(os.path.join(src, "*.aselmdb")))
    if not src_files:
        raise SystemExit(f"no .aselmdb files found under {src}")
    if os.path.exists(dst):
        shutil.rmtree(dst)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)

    z_to_idx = {z: z - 1 for z in range(1, len(TYPE_MAP) + 1)}
    fmt = "012d"
    frame_nlocs: list[int] = []
    frame_system_ids: list[int] = []
    frame_idx = 0
    n_skipped = 0

    out = lmdb.open(dst, map_size=map_size, subdir=True)
    try:
        for sys_id, fpath in enumerate(src_files):
            with out.begin(write=True) as txn:
                for row in _iter_ase_rows(fpath):
                    numbers = _ase_array(row["numbers"]).astype(np.int64).ravel()
                    pos = _ase_array(row["positions"]).astype(np.float64).reshape(-1, 3)
                    cell = _ase_array(row["cell"]).astype(np.float64).reshape(3, 3)
                    energy = float(row["energy"])
                    forces = _ase_array(row["forces"]).astype(np.float64).reshape(-1, 3)

                    if numbers.max() > len(TYPE_MAP) or numbers.min() < 1:
                        n_skipped += 1
                        continue
                    atypes = np.fromiter((z_to_idx[int(z)] for z in numbers),
                                         dtype=np.int64, count=len(numbers))

                    frame = {
                        "coords": _enc(pos),
                        "cells": _enc(cell),
                        "energies": _enc(np.asarray([energy], dtype=np.float64)),
                        "forces": _enc(forces),
                        "atom_types": _enc(atypes),
                    }
                    if with_virial and row.get("stress") is not None:
                        stress = _ase_array(row["stress"]).astype(np.float64).reshape(3, 3)
                        volume = float(abs(np.linalg.det(cell)))
                        virial = -stress * volume          # dpdata/VASP convention
                        frame["virials"] = _enc(virial)

                    txn.put(format(frame_idx, fmt).encode(),
                            msgpack.packb(frame, use_bin_type=True))
                    frame_nlocs.append(int(len(numbers)))
                    frame_system_ids.append(sys_id)
                    frame_idx += 1
            print(f"[{sys_id + 1}/{len(src_files)}] {os.path.basename(fpath)} "
                  f"-> cumulative {frame_idx} frames", flush=True)

        meta = {
            "nframes": frame_idx,
            "frame_idx_fmt": fmt,
            "system_info": {},
            "frame_nlocs": frame_nlocs,
            "frame_system_ids": frame_system_ids,
            "type_map": TYPE_MAP,
        }
        with out.begin(write=True) as txn:
            txn.put(b"__metadata__", msgpack.packb(meta, use_bin_type=True))
    finally:
        out.close()

    print(f"DONE: {frame_idx} frames written to {dst} "
          f"({n_skipped} skipped for out-of-range Z)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="dir of *.aselmdb files (e.g. .../aselmdb/train)")
    ap.add_argument("--dst", required=True, help="output deepmd LMDB dir (must end in .lmdb)")
    ap.add_argument("--no-virial", action="store_true", help="skip stress->virial conversion")
    args = ap.parse_args()
    if not args.dst.endswith(".lmdb"):
        raise SystemExit("--dst must end in .lmdb so deepmd's is_lmdb() recognizes it")
    convert(args.src, args.dst, with_virial=not args.no_virial)


if __name__ == "__main__":
    main()
