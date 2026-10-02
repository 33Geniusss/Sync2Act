# Documentation languages

Every Chinese document published in this repository must have a corresponding English version. English is the main README language. Pairings are recorded in [language_pairs.json](language_pairs.json); the README links both versions.

When updating a Chinese document, update its English counterpart in the same commit. Keep section coverage, formulas, numeric results, and evidence boundaries aligned. A short English summary does not replace the full counterpart. Source code and machine-readable metric names do not need duplicate files; Chinese report templates have English translations in the same implementation.

## Generated reports

Both study runners write `report.html` (Chinese) and `report_en.html` (English). The temporal supplement also writes `section_en.html` and `ensemble_relative_mse_en.svg`. Both languages use the same numeric tables; unknown Chinese report text raises a translation error rather than silently producing a partial English report.

To regenerate reports from existing local evaluations:

```bash
python tools/evaluate_temporal_study.py --device cuda --decay 0.7 --report-only
python tools/build_english_report.py
```

The first command requires the original local study, checkpoints, datasets, and matching evaluation cache. The second translates the checked-in Chinese LaTeX source, preserves table cells and plot coordinates, and can run without experiment artifacts. It fails when source prose lacks a reviewed translation.

Compile both PDF sources from `output/pdf/`:

```bash
xelatex -interaction=nonstopmode -halt-on-error sync2act_mixed_modality_damage_report_v1_3_0.tex
xelatex -interaction=nonstopmode -halt-on-error sync2act_mixed_modality_damage_report_v1_3_0_en.tex
```

Use a TeX installation with XeLaTeX, CTeX, PGFPlots, and the packages declared by the sources. Repeat compilation if cross-references request it, render both PDFs, and inspect pages after layout changes. The published English source is a complete counterpart, including the temporal-ensemble plot and actual before/after MSE tables.

## Before publishing

```bash
python tools/check_documentation_languages.py
```

CI runs this check. It scans tracked Markdown/LaTeX/HTML documents for Chinese text, requires a registered English counterpart, verifies both files are tracked, and checks English counterparts for untranslated Chinese. PDF pairs are checked for tracked-file presence; numeric parity and visual layout need the separate report review above. Runtime datasets, checkpoints, caches, and large per-frame predictions remain ignored.
