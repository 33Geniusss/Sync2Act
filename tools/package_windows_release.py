"""Create a maximum-compression ZIP for the packaged Windows application."""

import argparse
import os
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

PYARROW_DEVELOPMENT_SUFFIXES = {".h", ".hpp", ".lib", ".pxd", ".pxi", ".pyx"}
PYARROW_DEVELOPMENT_DIRECTORIES = {"include", "includes", "src", "tests"}
PYARROW_UNUSED_RUNTIME_FILES = {
    "_flight.cp313-win_amd64.pyd",
    "arrow_flight.dll",
    "arrow_python_flight.dll",
}
REQUIRED_RUNTIME_FILES = {
    "Sync2Act/Sync2Act.exe",
    "Sync2Act/_internal/pyarrow/lib.cp313-win_amd64.pyd",
    "Sync2Act/_internal/pyarrow/parquet.dll",
}
RUNTIME_DATA_DIRECTORIES = {".cache", "datasets", "models"}


def _is_runtime_file(path: Path, source: Path) -> bool:
    relative = path.relative_to(source)
    parts = relative.parts
    if parts and parts[0] in RUNTIME_DATA_DIRECTORIES:
        return False
    if len(parts) >= 3 and parts[:2] == ("_internal", "pyarrow"):
        if parts[2] in PYARROW_DEVELOPMENT_DIRECTORIES:
            return False
        if path.suffix.lower() in PYARROW_DEVELOPMENT_SUFFIXES:
            return False
        if path.name in PYARROW_UNUSED_RUNTIME_FILES:
            return False
    return True


def verify(target: Path) -> None:
    with ZipFile(target) as archive:
        names = set(archive.namelist())
        missing = REQUIRED_RUNTIME_FILES - names
        if missing:
            raise RuntimeError(f"Packaged archive is missing runtime files: {missing}")
        bad_entries = [
            name
            for name in names
            if name.endswith(tuple(PYARROW_UNUSED_RUNTIME_FILES))
        ]
        if bad_entries:
            raise RuntimeError(f"Packaged archive contains excluded files: {bad_entries}")


def package(source: Path, target: Path) -> Path:
    source = source.resolve()
    target = target.resolve()
    if not source.is_dir() or not (source / "Sync2Act.exe").is_file():
        raise FileNotFoundError(f"Invalid packaged application directory: {source}")

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with ZipFile(
        temporary,
        mode="w",
        compression=ZIP_DEFLATED,
        compresslevel=9,
        allowZip64=True,
    ) as archive:
        for path in sorted(source.rglob("*")):
            if path.is_file() and _is_runtime_file(path, source):
                archive.write(path, Path(source.name) / path.relative_to(source))
    os.replace(temporary, target)
    verify(target)
    return target


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("dist/Sync2Act"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("release/Sync2Act-Windows-x64-v1.0.0.zip"),
    )
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        verify(args.output)
        print(f"Verified {args.output}")
        return 0
    print(package(args.source, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
