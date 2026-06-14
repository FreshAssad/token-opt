# token-opt

**Deterministically compress and count the text, code, documents, and data you're about to send to any LLM — locally, offline, and honestly.**

One `pip install`, many modes. `token-opt` cleans and compresses your input *before* it reaches Claude, GPT, Gemini, or a local model, then tells you exactly how many tokens — and how much money — you saved.

```
file/text  →  token-opt (local, deterministic compress)  →  compressed output  →  you send it to ANY LLM with your own key
```

> **No universal multiplier.** Savings depend entirely on your content and file type. Some tools advertise headline numbers like "71.5×" or "40–60%" that collapse to single digits on real data. `token-opt` never markets a fixed multiplier — every command prints the **real measured before/after token counts** so you can see what you actually got.

---

## What it is — and isn't

- ✅ **Deterministic & local.** Compression is parsers, AST/format conversion, dedup, and regex. **No LLM is ever called in the compression path.**
- ✅ **Offline-first.** No network access unless you explicitly pass `--api` for an exact provider token count. (tiktoken fetches its vocabulary once on first use, then caches it.)
- ✅ **Lossless by default.** Default modes preserve all information. Lossy modes (extractive prose) are opt-in and clearly labelled.
- ✅ **Pipe-friendly.** Compressed payload → `stdout`. Diagnostics (before/after, cost) → `stderr`. Reads files, directories, or `stdin`.
- ❌ Not an AI tool, not a RAG/vector system, not a "suggestion" tool, not an agent proxy/MCP server, not GPU-dependent, no telemetry.

---

## Install

```bash
# end users (recommended): isolated global install
pipx install token-opt

# or with pip
pip install token-opt

# from source (dev)
git clone <repo> && cd token-opt
pip install -e ".[dev]"
```

Optional extras:

```bash
pip install "token-opt[doc]"       # PDF/DOCX/PPTX/XLSX backends (HTML & Markdown work without it)
pip install "token-opt[code]"      # tree-sitter grammars for `compress code` (Python/JS/TS)
pip install "token-opt[pymupdf]"   # hi-fi PDF backend — AGPL-3.0, see note below
pip install "token-opt[gemini]"    # exact/near-exact local Gemini counting (heavy)
```

> `compress doc` on **HTML/Markdown** works out of the box. For **PDF/DOCX/PPTX/XLSX**, install `token-opt[doc]`; otherwise `token-opt` prints a clear one-line hint instead of failing.

---

## Quickstart

```bash
# Count tokens (exact for GPT, offline)
token-opt count report.md --model gpt-4o

# Estimate cost across providers
token-opt cost report.md --compare claude,gpt-4o,gemini --monthly-queries 1000

# Compress a messy PDF/HTML/DOCX to clean Markdown
token-opt compress doc paper.pdf > clean.md

# Compress JSON to TOON (or compact JSON if that's smaller)
token-opt compress data records.json > records.toon

# Let it auto-detect and route
cat anything.json | token-opt pipe -
```

Everything supports `--json` (machine-readable) and reading from `stdin` via `-`.

---

## Honest savings reference

These are realistic ranges, **not guarantees**. Your mileage depends on the content.

| Mode | Realistic savings | Lossy? | Notes |
|---|---|---|---|
| `compress doc` (PDF) | 30–70% | No | print-heavy / messy PDFs save more |
| `compress doc` (HTML) | up to ~90% | No | boilerplate-heavy pages |
| `compress doc` (DOCX) | 40–60% | No | strips metadata / formatting |
| `compress data` (JSON→TOON) | ~40% on uniform arrays; **as low as ~2% on nested** | No | falls back to JSON when TOON isn't smaller |
| `compress code` (AST) | 30–55%; more with `--skeleton` | optional | strips comments; default keeps code runnable |
| `compress email` / `transcript` | 40–70% on deep threads / long calls | mild | dedup quoted history; strip timestamps + filler |
| `compress generic` (prose) | user-set (e.g. ~2–3× at `--ratio 0.3`) | **YES** | extractive, opt-in, clearly labelled |

### Measured on the bundled corpus

Reproduce with `python benchmarks/run_bench.py`:

<!-- BENCH:START -->
| File | Type | Result | Before | After | Saved |
|---|---|---|---:|---:|---:|
| `config_nested.json` | data | toon | 152 | 121 | **20.4%** ¹ |
| `meeting.srt` | transcript | transcript | 208 | 85 | **59.1%** ¹ |
| `report.html` | doc | markdown | 531 | 159 | **70.1%** ¹ |
| `sample.py` | code | code | 382 | 181 | **52.6%** ¹ |
| `thread.eml` | email | email | 234 | 71 | **69.7%** ¹ |
| `users_uniform.json` | data | toon | 224 | 86 | **61.6%** ¹ |

¹ heuristic estimate (tiktoken vocab unavailable in this environment).
<!-- BENCH:END -->

---

## Token counting accuracy

| Provider | Offline accuracy | How |
|---|---|---|
| **OpenAI / GPT** | **Exact** | `tiktoken` (`o200k_base` / `cl100k_base`), bundled, no key |
| **Gemini** | **Near-exact** | local Vertex tokenizer (optional `[gemini]` extra); `--api` for exact |
| **Claude** | **Estimate only** | No public offline tokenizer. We use an `o200k_base` proxy, always printed as `~N (estimate)`. `--api` → exact via Anthropic's free `count_tokens`. |
| **Llama / local** | Approximate | `cl100k_base` proxy, labelled `(approx)` |

- Offline Claude counts are **always** labelled `(estimate)` — never presented as exact.
- `--api` uses the provider's endpoint for ground truth (needs `ANTHROPIC_API_KEY` / `GEMINI_API_KEY`).
- `--opus-correction` applies a ~1.3× nudge to offline Claude estimates (newer Opus-class models tokenize denser).
- **Fully air-gapped box with no warm tiktoken cache?** Counting degrades to a deterministic heuristic labelled `(approx)` instead of failing.

---

## Commands

```text
token-opt count   <file|dir|->         --model --api --opus-correction --json
token-opt cost    <file|->             --model --compare a,b,c --monthly-queries N --json
token-opt compress doc        <file|-> --keep-tables/--no-keep-tables --strip-bibliography
token-opt compress data       <file|-> --format toon|csv
token-opt compress code       <file|-> --skeleton                 # strip comments / signatures-only
token-opt compress email      <file|-> --keep-quotes              # dedup quoted history
token-opt compress transcript <file|-> --summary --ratio 0.3      # .srt/.vtt cleanup
token-opt compress generic    <file|-> --ratio 0.3                # extractive prose (LOSSY)
token-opt pipe    <file|->             # auto-detect → route → report
```

Colon aliases also work: `token-opt compress:doc paper.pdf`. Every command takes `--json`, `--quiet`, and reads `stdin` via `-`.

Still on the roadmap: reversible identifier maps for `compress code`, an optional GPU `--backend llmlingua`, remote `prices.json --update`, and an n8n community node.

---

## Where compression fails (read this)

- **Scanned / image-only PDFs** — there's no text to extract; OCR is out of scope for v1. `token-opt` warns and emits what it can.
- **Nested / non-uniform JSON** — TOON's win comes from *uniform arrays of objects*. On deeply nested data it can be larger than JSON, so the guard emits compact JSON instead (and says so on stderr).
- **Already-clean input** — if there's no boilerplate to remove, savings are small. That's honest, not a bug.
- **Code in unsupported languages** — without a tree-sitter grammar, `compress code` does only a safe whitespace cleanup (and tells you so on stderr).
- **Lossy prose mode** — `compress generic` and `transcript --summary` drop sentences via extractive TextRank. Opt-in and labelled; don't use them where you need every word.

---

## How it works

```
 file / stdin ─► type detector (extension + content sniff)
                   ├─ pdf/docx/html/pptx/xlsx → MarkItDown → strip headers/footers/page-nums
                   ├─ .json                    → TOON encoder (+ JSON fallback guard)
                   ├─ source code              → tree-sitter AST → strip comments / --skeleton
                   ├─ .eml / .mbox             → dedup quoted history + drop signatures
                   ├─ .srt / .vtt              → strip timestamps/filler, merge speakers
                   └─ other prose              → lossless whitespace cleanup
                         │
                         ▼
                 token counter (tiktoken / Vertex / Claude-estimate | --api ground truth)
                         ▼
                 cost table (bundled prices.json)
        compressed text → STDOUT   │   before/after tokens + $ → STDERR
```

Prices live in `tokenopt/prices.json` (a dated snapshot for rough estimates). Override with `TOKENOPT_PRICES=/path/to/prices.json` or a `prices.json` in your working directory. **`token-opt` never fetches prices over the network in v1.**

---

## Development

```bash
pip install -e ".[dev]"
pytest                       # offline test suite (Anthropic API is mocked)
python benchmarks/run_bench.py   # regenerate the measured-savings table
```

The suite includes a no-network test asserting that the non-`--api` path never opens a socket.

---

## License

**MIT** — see [LICENSE](LICENSE). All default dependencies are permissive (MIT/Apache/BSD).

> **AGPL note:** the optional `pymupdf` extra (`pymupdf4llm`) is **AGPL-3.0**. Installing it does not relicense `token-opt`, but redistributing a bundle that includes it brings AGPL obligations to that bundle. It is off by default on purpose.
