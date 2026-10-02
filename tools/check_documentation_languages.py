"""Check that every published Chinese document has a tracked English counterpart."""

import json
import re
import subprocess
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    tracked = set(
        subprocess.check_output(["git", "ls-files", "-z"], cwd=root)
        .decode("utf-8")
        .strip("\0")
        .split("\0")
    )
    pairs = json.loads((root / "docs/language_pairs.json").read_text(encoding="utf-8"))
    chinese = re.compile(r"[\u4e00-\u9fff]")
    errors = []
    for name in sorted(tracked):
        path = root / name
        if path.suffix.lower() in {".md", ".tex", ".html"} and path.is_file():
            if chinese.search(path.read_text(encoding="utf-8")) and name not in pairs:
                errors.append(f"Chinese document has no registered English counterpart: {name}")
    for original, english in pairs.items():
        for name in (original, english):
            if name not in tracked or not (root / name).is_file():
                errors.append(f"Language-pair file is missing or untracked: {name}")
        path = root / english
        if path.is_file() and path.suffix != ".pdf":
            if chinese.search(path.read_text(encoding="utf-8")):
                errors.append(f"Untranslated Chinese in English counterpart: {english}")
    expected = {f"{i:02d}" for i in range(1, 85)}
    for name in ("SYNC2ACT_INTERVIEW_QA_ZH.md", "SYNC2ACT_INTERVIEW_QA_EN.md"):
        text = (root / name).read_text(encoding="utf-8")
        questions = re.findall(r"^\*\*Q(\d{2})\D", text, flags=re.M)
        if len(questions) != 84 or set(questions) != expected:
            errors.append(f"Expected exactly Q01-Q84 in {name}")
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Language audit passed: {len(pairs)} document pairs; Q01-Q84 in both languages.")


if __name__ == "__main__":
    main()
