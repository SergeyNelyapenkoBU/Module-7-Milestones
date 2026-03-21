# Module 7 - Milestones

Project milestones for the BU Advanced ML & AI course (Module 7). The first milestone focuses on team formation, problem understanding, and data exploration for an NLP/deep learning task.

## Setup

### Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager:
  ```bash
  brew install uv
  ```
- Python 3.9–3.11 (required for `tensorflow-metal` compatibility)

### Install dependencies

```bash
uv sync
```

This creates a `.venv` virtual environment and installs all required packages (TensorFlow, scikit-learn, Hugging Face Datasets, spaCy, etc.).

### Activate the environment

```bash
source .venv/bin/activate
```

### Run the notebook

```bash
uv run jupyter notebook Milestone_01.ipynb
```

Or select the `.venv` Python interpreter in VS Code / JupyterLab.
