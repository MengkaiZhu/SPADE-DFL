from __future__ import annotations

import argparse
import gzip
from pathlib import Path
import struct
from typing import Callable, Mapping
from urllib.request import urlopen

import numpy as np


_BASE = (
    "https://raw.githubusercontent.com/zalandoresearch/"
    "fashion-mnist/master/data/fashion"
)
OFFICIAL_FILES = {
    "x_train": f"{_BASE}/train-images-idx3-ubyte.gz",
    "y_train": f"{_BASE}/train-labels-idx1-ubyte.gz",
    "x_test": f"{_BASE}/t10k-images-idx3-ubyte.gz",
    "y_test": f"{_BASE}/t10k-labels-idx1-ubyte.gz",
}
LOCAL_FILENAMES = {
    name: url.rsplit("/", 1)[-1] for name, url in OFFICIAL_FILES.items()
}


def decode_gzipped_idx(payload: bytes) -> np.ndarray:
    raw = gzip.decompress(payload)
    if len(raw) < 4 or raw[:2] != b"\x00\x00":
        raise ValueError("invalid IDX header")
    dtype_code = raw[2]
    if dtype_code != 0x08:
        raise ValueError("Fashion-MNIST IDX payload must use uint8 values")
    ndim = raw[3]
    if ndim <= 0:
        raise ValueError("IDX array must have at least one dimension")
    header_size = 4 + 4 * ndim
    if len(raw) < header_size:
        raise ValueError("truncated IDX dimension header")
    shape = struct.unpack(f">{ndim}I", raw[4:header_size])
    values = np.frombuffer(raw, dtype=np.uint8, offset=header_size)
    expected_size = int(np.prod(shape, dtype=np.int64))
    if values.size != expected_size:
        raise ValueError(
            f"IDX payload size mismatch: expected {expected_size}, got {values.size}"
        )
    return values.reshape(shape).copy()


def write_npz_from_payloads(payloads: Mapping[str, bytes], output: Path) -> None:
    missing = set(OFFICIAL_FILES) - set(payloads)
    if missing:
        raise ValueError(f"missing Fashion-MNIST arrays: {sorted(missing)}")
    arrays = {
        name: decode_gzipped_idx(payloads[name]) for name in OFFICIAL_FILES
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(output)


def _fetch(url: str) -> bytes:
    with urlopen(url, timeout=120) as response:
        return response.read()


def download_official_fashion_mnist(
    output: Path,
    *,
    fetcher: Callable[[str], bytes] = _fetch,
) -> None:
    payloads = {name: fetcher(url) for name, url in OFFICIAL_FILES.items()}
    write_npz_from_payloads(payloads, output)


def convert_local_fashion_mnist(source_dir: Path, output: Path) -> None:
    source_dir = Path(source_dir)
    paths = {
        name: source_dir / filename for name, filename in LOCAL_FILENAMES.items()
    }
    missing = [path for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "missing Fashion-MNIST IDX file(s): "
            + ", ".join(path.name for path in missing)
        )
    write_npz_from_payloads(
        {name: path.read_bytes() for name, path in paths.items()}, output
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download official Fashion-MNIST IDX files and write one NPZ"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="directory containing the four official .gz IDX files; omit to download",
    )
    args = parser.parse_args()
    if args.source_dir is None:
        download_official_fashion_mnist(args.output)
    else:
        convert_local_fashion_mnist(args.source_dir, args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
