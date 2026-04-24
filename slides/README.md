# Defense Slides — MSc Thesis

Slidev-based slides for the defense of *Implementation of Shrinkage Estimators for Cosmological Precision Matrices*.

- **English:** `slides.en.md`
- **Spanish:** `slides.es.md`
- **Figures:** loaded from `../paper/figures/` via relative paths.

## Install

```bash
cd slides
npm install
```

## Run (live)

```bash
npm run dev:en   # English
npm run dev:es   # Spanish
```

Open browser at the shown localhost URL. Use arrow keys / space to advance. Press `d` to toggle dark mode, `o` for overview, `p` for presenter mode.

## Export to PDF

```bash
npm run export:en
npm run export:es
```

Requires `playwright-chromium` (installed automatically by `slidev export`).

## Target length

~25 minutes, ~22 slides at ~1 min each. The three slides specifically addressing the juries' observations are marked with a `⭐` in the speaker notes.
