import hashlib
import json
import os
import pathlib
from argparse import ArgumentParser

from tensorflow.keras.utils import Progbar


def parse_args():
    """Parse CLI argument."""
    parser = ArgumentParser()
    parser.add_argument("--input", "-i", help="Path to directory with .h5 files.")
    return parser.parse_args()


def main():
    """Generate md5 file has for each .h5 file in a given directory."""
    args = parse_args()

    weight_files = sorted([x for x in os.listdir(args.input) if x.endswith(".h5")])
    progbar = Progbar(target=len(weight_files))

    summary = {}

    for file in weight_files:
        path = os.path.join(args.input, file)
        file_hash = _md5_hash(path)
        summary.update({file: file_hash})
        progbar.add(1)

    print(json.dumps(summary, indent=4))


def _md5_hash(path):
    return hashlib.md5(pathlib.Path(path).read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
