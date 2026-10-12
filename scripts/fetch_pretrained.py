"""Download and verify a pretrained encoder before training.

Training fetches weights on first use anyway; this script exists so the
download can happen ahead of time -- on a machine with internet access, or
before an unattended run that should not stall on the network.

    py -3.12 scripts/fetch_pretrained.py                     # dino_vits16
    py -3.12 scripts/fetch_pretrained.py --name dino_vitb16
    py -3.12 scripts/fetch_pretrained.py --list

The file lands in torch's hub cache (``torch.hub.get_dir()/checkpoints``).
To prepare an offline machine, copy that file to the same place there, or
point ``TORCH_HOME`` at a directory that contains it.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.pretrained import (  # noqa: E402
    PRETRAINED,
    checkpoint_path,
    describe_spec,
    fetch_state_dict,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--name", default="dino_vits16", choices=sorted(PRETRAINED))
    parser.add_argument("--list", action="store_true", help="list available encoders")
    args = parser.parse_args(argv)

    if args.list:
        for name in sorted(PRETRAINED):
            cached = "cached" if os.path.isfile(checkpoint_path(name)) else "not downloaded"
            print(f"{name:14s} {describe_spec(name)}  [{cached}]")
        return 0

    print(describe_spec(args.name))
    state = fetch_state_dict(args.name)          # downloads once, verifies size + hash
    print(f"OK  {len(state)} tensors  ->  {checkpoint_path(args.name)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
