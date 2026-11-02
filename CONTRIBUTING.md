# Contributing to Antigravity Project Memory

Thank you for your interest in contributing to **Antigravity Project Memory**! We welcome community contributions of all kinds: bug fixes, performance optimizations, new retrieval algorithms, and integrations with other developer environments.

---

## 🛠️ Development Setup

### 1. Fork & Clone
```bash
git clone https://github.com/<your-username>/antigravity-project-memory.git
cd antigravity-project-memory
```

### 2. Set Up a Virtual Environment
```bash
# Python 3.10+ is recommended
python -m venv .venv

# On Linux/macOS:
source .venv/bin/activate

# On Windows PowerShell:
.venv\Scripts\Activate.ps1
```

### 3. Install in Editable Mode with Dev Dependencies
```bash
pip install -e ".[dev]"
```

---

## 🧪 Running Tests

Always ensure the full test suite passes before submitting changes:

```bash
# Using standard unittest
python -m unittest discover -s tests -v

# Or using pytest
pytest -v
```

---

## 📐 Code Style & Architecture Guidelines

To keep the codebase fast, lightweight, and maintainable by the open-source community:

1. **Zero External Neural Dependencies:**
   - The core retrieval engine must remain **pure Python** (using standard library `re`, `math`, `collections`, `sqlite3`).
   - Do **not** introduce heavy neural dependencies (PyTorch, transformers, CUDA) into the core package.
2. **SOLID & Clean Code:**
   - Follow Single Responsibility: keep retrieval logic, parsing, and MCP server routing in separate modules.
   - Use meaningful variable and function names.
   - Provide type hints (`typing`) and docstrings on all public methods.
3. **Cross-Platform Compatibility:**
   - Any path operations must use `os.path` or `pathlib.Path` to support Windows drive paths, POSIX paths, and `file:///` URIs.
4. **Performance Budgets:**
   - Retrieval queries must remain sub-5ms on standard developer hardware.
   - Memory footprint must remain minimal (<50 MB).

---

## 🚀 Submitting a Pull Request (PR)

1. **Branch Naming:** Create a feature branch with a descriptive name:
   ```bash
   git checkout -b feat/custom-bm25-parameters
   # or
   git checkout -b fix/windows-uri-spaces
   ```
2. **Commit Messages:** Follow standard conventional commits:
   - `feat: add configurable IDF smoothing option`
   - `fix: resolve URI encoding for paths with spaces`
   - `docs: improve Claude Desktop configuration examples`
3. **Open a PR:** Push your branch and open a Pull Request against `main`. Fill out the PR template completely.

---

## 💬 Getting Help & Discussions

- **Bug Reports:** [Open an Issue](https://github.com/R-Mabasha/antigravity-project-memory/issues/new?template=bug_report.yml)
- **Feature Requests:** [Propose an Idea](https://github.com/R-Mabasha/antigravity-project-memory/issues/new?template=feature_request.yml)
