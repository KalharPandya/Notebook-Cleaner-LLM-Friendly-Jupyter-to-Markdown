# Notebook Cleaner — LLM-Friendly Jupyter to Markdown

**Convert Jupyter notebooks (`.ipynb`) to clean, token-efficient markdown** for AI assistants, code review, and documentation. Zero dependencies, 90–95% size reduction.

**Keywords:** Jupyter · ipynb · markdown · LLM-friendly · AI context window · notebook size reduction · token optimization · ChatGPT · Claude

---

## ⚠️ Important: Image Outputs Are Removed

**This tool removes all image content from notebook outputs.**

- **Removed:** Every `image/png`, `image/jpeg`, and `image/svg+xml` output (including plots, charts, screenshots, and figures) is **stripped** and replaced with the placeholder: `[Image output removed]`.
- **Why:** Base64-encoded images make notebooks huge and consume most of the size; removing them is what gives the large size reduction and makes output LLM-friendly.
- **Before running:** If you need to keep visual outputs, **back up your notebook** or use a copy. The original `.ipynb` file is never modified; only the new markdown file is written.

Use this tool when you want **code + text only** (e.g., for LLM context, code review, or lightweight docs), not when you need to preserve figures.

---

## Why Use This?

| Use case | Benefit |
|----------|---------|
| **LLM context** | Feed notebooks to ChatGPT, Claude, or other models without blowing the context window |
| **Token optimization** | Shrink notebooks to essential code and text for cheaper/faster AI usage |
| **Code review** | Share logic and outputs without multi‑MB image payloads |
| **Search & discovery** | Clean markdown is easy to search and index (e.g., LLM-friendly, notebook cleaner) |
| **Version control** | Smaller, diff-friendly markdown instead of large JSON + base64 |

**Preserved:** All code, all markdown, short text outputs, and truncated long/error outputs.  
**Removed:** Images, heavy metadata, long raw outputs (truncated), and large HTML/JS.

---

## Quick Start

```bash
# Single notebook → creates notebook_cleaned.md
python shrink_notebook.py my_notebook.ipynb

# All notebooks in a folder (recursive)
python shrink_notebook.py ./notebooks/

# Install and run from anywhere (uv or pip)
uv pip install -e .   # or: pip install -e .
shrink-notebook my_notebook.ipynb
```

---

## Installation

**Requirements:** Python 3.7+; no external dependencies (standard library only).

### Run directly

```bash
git clone <repo-url>
cd IPNYB_Compresser
python shrink_notebook.py sample.ipynb
```

### Install with uv (recommended)

```bash
# Install uv: https://github.com/astral-sh/uv
curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows: powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

uv pip install -e .
shrink-notebook sample.ipynb
python -m shrink_notebook sample.ipynb   # if Scripts not on PATH
```

### Install with pip

```bash
pip install -e .
shrink-notebook sample.ipynb
```

---

## Usage

| Command | Result |
|---------|--------|
| `python shrink_notebook.py file.ipynb` | Creates `file_cleaned.md` |
| `python shrink_notebook.py ./dir/` | Recursively processes all `.ipynb` in `./dir/` |
| `python shrink_notebook.py file.ipynb -o out.md` | Writes to `out.md` |
| `python shrink_notebook.py ./dir/ --skip-existing` | Skips notebooks that already have a `*_cleaned.md` |
| `python shrink_notebook.py ./dir/ --dry-run` | Shows what would be processed |
| `python shrink_notebook.py file.ipynb --verbose` | Shows per-file stats (images removed, outputs truncated) |

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `input_path` | File or directory (required) | — |
| `-o, --output` | Output path (single file only) | `{name}_cleaned.md` |
| `--max-text-length` | Truncate text outputs longer than this | 1000 |
| `--max-error-lines` | Max traceback lines to keep | 8 |
| `--skip-existing` | Skip if `_cleaned.md` already exists | off |
| `--dry-run` | List targets only | off |
| `--verbose` | Extra stats per file | off |

---

## What Gets Removed vs Preserved

### Removed (including images)

- **All image outputs** — PNG, JPEG, SVG (replaced with `[Image output removed]`). **This is the main size reduction.**
- **Metadata** — execution counts, cell IDs, Colab/widget/platform metadata
- **Long text** — truncated to first/last portions with a `[... truncated ...]` marker
- **Long tracebacks** — first and last lines only
- **Large HTML / JavaScript** — dropped or replaced with a short note

### Preserved

- All **code** and **markdown** cell content  
- Short **text outputs** (stdout/stderr)  
- Truncated **errors** and long outputs (beginning + end)  
- **Notebook order** and structure in the generated markdown  

---

## Example Output

```
Processing: sample.ipynb

Found 1 notebook(s)
------------------------------------------------------------
[OK] sample.ipynb -> sample_cleaned.md
  673.0 KB -> 47.0 KB (93.0% reduction)
  removed 2 images, truncated 4 outputs
------------------------------------------------------------

Summary:
  Processed: 1 file(s)
  Total size: 673.0 KB -> 47.0 KB (93.0% reduction)
  Images removed: 2
  Outputs truncated: 4
  Time: 0.3 seconds
```

---

## Output Format

The cleaned file is **markdown** (e.g. `notebook_cleaned.md`), not a runnable notebook:

```markdown
# Notebook: sample.ipynb
Generated from: path/to/sample.ipynb
Date: 2026-02-07

---

## Markdown Cell 1
[Your markdown, unchanged]

## Code Cell 1
```python
[Your code, unchanged]
```

**Output:**
```
[Text output or truncated output]
```
[Image output removed]

## Code Cell 2
...
```

Ideal for **LLM-friendly** ingestion, reading, or search.

---

## Troubleshooting

- **Unicode on Windows:** The script sets UTF-8 where possible. If needed: `python -X utf8 shrink_notebook.py file.ipynb` or `set PYTHONIOENCODING=utf-8`.
- **Invalid notebook:** Ensure the file is valid JSON with a `cells` array (e.g. open in Jupyter once).
- **Permission errors:** Need read access on inputs and write access for the output directory.

---

## Author

**Kalhar Pandya** — [kalharpandya38@gmail.com](mailto:kalharpandya38@gmail.com)

Built to make Jupyter notebooks **LLM-friendly** and easier to use with AI tools and code review.

---

## License

MIT — use and modify freely.
