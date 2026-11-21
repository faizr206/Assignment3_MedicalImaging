# ENG2440 Medical Imaging & AI in Healthcare — Assignment 3

**Molecular Property Prediction with Graph Neural Networks** (BBBP dataset, Fall 2026).

An end-to-end pipeline: molecules as graphs → atom/bond one-hot encoding → a Morgan/ECFP
fingerprint + logistic-regression baseline and a bond-aware GINE graph neural network →
random vs Bemis–Murcko scaffold evaluation → error and similarity analysis.

## Repository contents

| file | what it is |
|---|---|
| `Assignment3_Faiz.ipynb` | the full assignment — one executed Jupyter notebook, Parts A–G, runs top to bottom |
| `requirements.txt` | pip environment (exact versions used to produce the committed outputs) |
| `environment.yml` | equivalent conda environment |
| `ai_use_declaration.txt` | declaration of generative-AI use, per Section 6 of the assignment |
| `data/` (gitignored) | PyTorch Geometric downloads the BBBP dataset here automatically — do not upload |

## Environment setup

Option 1 — conda:

```bash
conda env create -f environment.yml
conda activate eng2440-a3
```

Option 2 — pip (Python ≥ 3.11):

```bash
python -m venv .venv && source .venv/bin/activate   # or any existing environment
pip install -r requirements.txt
```

A **CPU-only PyTorch build is sufficient**: the GNN is trained on CPU on purpose
(the model is small, ≈0.85 s/epoch, and CPU execution makes the notebook
bit-reproducible run-to-run; GPU scatter-add kernels are nondeterministic).

## Run order

Single notebook, no supporting scripts — run all cells in order:

```bash
jupyter notebook Assignment3_Faiz.ipynb   # Kernel → Restart & Run All
```

Notes:

* **First execution needs internet**: the PyG `MoleculeNet` loader downloads BBBP
  (~2,050 compounds) into `data/MoleculeNet/` and caches it; later runs work offline.
* **Runtime**: ≈3 min end-to-end on a 12-core CPU node (188 s measured for
  all 74 cells; six GNN training runs of well under a minute each; the
  10–15 min earlier stated in this README was measured on a slower machine).
* `data/` is in `.gitignore` — the dataset must not be pushed to GitHub or Moodle.

## Reproducibility

* Global seed `SEED = 42`: data cleaning, deduplication, both 70:15:15 splits
  (stratified random + Bemis–Murcko scaffold), the fingerprint baseline and its threshold.
* GNN seeds `42, 7, 2023` on the **same fixed splits** (only initialisation/batch order vary).
* Thresholds are selected on validation subsets only; each test subset is scored exactly once (Part F).
* With the pinned versions above, re-running the notebook reproduces the committed outputs
  bit-for-bit on CPU.
