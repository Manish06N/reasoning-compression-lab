# Literature archive

Canonical merged PDF bundles that underpin the PhD papers, plus text extracts for search and LLM use.

## Layout

| Path | Pages | PDF size | Text extract |
|------|------:|---------:|-------------:|
| [paper1/ALL_PAPERS_MERGED.pdf](paper1/ALL_PAPERS_MERGED.pdf) | 1049 | ~116 MB | [paper1/ALL_PAPERS_MERGED.md](paper1/ALL_PAPERS_MERGED.md) (~3.5 MB) |
| [paper2/ALL_PAPERS_MERGED.pdf](paper2/ALL_PAPERS_MERGED.pdf) | 165 | ~30 MB | [paper2/ALL_PAPERS_MERGED.md](paper2/ALL_PAPERS_MERGED.md) (~535 KB) |

**Paper 1 reading guide:** [PAPER1_READING_MAP.md](PAPER1_READING_MAP.md)

> **Coverage warning (2026-08-14, still true as a corpus note):** the merged PDF bundle is a historical local snapshot, not a complete current novelty search. The 2026-08-17 manuscript Related Work table covers QRM, Lian, Lotfi, Kurtic, Erol, Hochlehnert, and G-Pass@k. Do not treat this folder as the live literature review.

## PDF vs Markdown — which to use?

| Use case | Use |
|----------|-----|
| Human reading, figures, tables | **PDF** |
| Grep, ripgrep, codebase search | **`.md` extract** |
| LLM in Cursor (single paper / section) | **`.md` extract** — load relevant pages only |
| LLM (full corpus at once) | **Neither** — 1049 pages is too large for one context window; use the reading map + targeted excerpts |

The `.md` files are **plain text extracted** from the PDFs (`pdftotext`). They lose layout, figures, and math formatting, but preserve searchable prose for agents.

Regenerate extracts after replacing a PDF:

```bash
pdftotext docs/literature/paper1/ALL_PAPERS_MERGED.pdf docs/literature/paper1/ALL_PAPERS_MERGED.md
pdftotext docs/literature/paper2/ALL_PAPERS_MERGED.pdf docs/literature/paper2/ALL_PAPERS_MERGED.md
```

## Git / GitHub notes

- **Paper 1 PDF (~116 MB)** exceeds GitHub’s **100 MB per-file limit**. It is **gitignored** — keep it locally (or use [Git LFS](https://git-lfs.com/) if you need it on GitHub).
- **Paper 1 `.md` extract (~3.5 MB)** and **Paper 2 PDF (~30 MB)** are safe to commit.
- Original source copies live under `~/Downloads/merged for paper 1/` and `~/Downloads/merged for paper 2/`.
