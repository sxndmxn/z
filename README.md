# z

`z` analyzes CSV or TSV data and uses a local language model to propose XML changes.
Data analysis and XML modification run as separate commands.

## Install

Install Rust and Cargo.
Run this command from the repository root:

```bash
cargo install --path . --locked
```

The executable is named `z`.
The examples use `command z` to bypass shell functions, including a zoxide navigation function.

## Analyze data

```bash
command z analyze --csv data.csv --output-dir ml_output
command z analyze --help
```

Use `--tsv` for tab-separated input.
The output includes `summary.txt`, `clusters.csv`, `anomalies.csv`, and `stats.json`.
The tool also writes `correlation.csv` when correlation results are available.

## Preview XML changes

XML modification requires a `llama-server` executable and a compatible GGUF model file.
GGUF is a model file format used by llama.cpp.

```bash
command z modify \
  --context-dir ml_output \
  --xml document.xml \
  --server /path/to/llama-server \
  --model /path/to/model.gguf \
  --dry-run
```

The preview starts the model server and reports proposed changes without saving the XML file.
Back up the XML file before you run the command without `--dry-run`.
Use `command z modify --help` for model and iteration options.
