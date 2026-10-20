# NOMAD-Net

**NOMAD-Net — Unsupervised Neural Operational Modal Analysis**

NOMAD-Net is a research implementation of an unsupervised neural framework for operational modal analysis (OMA). It learns from multichannel vibration responses **without labelled modal examples or reference modal parameters in the training objective**, and estimates complex mode shapes and modal components from which natural frequencies and damping ratios can be extracted.

This repository currently provides a **four-channel numerical example with classical damping**, its simulated acceleration records, classical OMA comparison results, and a pretrained PyTorch checkpoint. It is not yet a complete release of every case study described in the associated research.

Developed within [IMIS-Lab, University of Granada](https://github.com/IMIS-Lab-UGR).

## Method at a glance

- Input: multichannel vibration measurements arranged as samples × sensors.
- Preprocessing: mean removal, band-pass filtering, downsampling and training-set-based normalization.
- Representation: a time-delay Hankel matrix passed through a shared convolutional encoder.
- Outputs: modal components and complex mode shapes, followed by extraction of modal frequencies and damping.
- Training objective: multicomponent losses based on delayed statistical separation, signal reconstruction, recurrence and modal-shape constraints.

**Unsupervised learning:** `Classical_OMA_results/OMA_CoV-SSI.csv` is read for **MAC-based comparison and visualisation**, not for computing the training loss, supplying target modes, or supervising weight updates.

## Repository contents

| Path | Description |
| --- | --- |
| `NOMAD_Net.py` | Train the model and save checkpoints and diagnostic plots. |
| `NOMAD_Net_Use_model.py` | Load the supplied checkpoint, reconstruct responses, and estimate modal properties. |
| `Input_signals/Simulated_records.csv` | Four-channel numerical acceleration records (g). |
| `Input_signals/Model and CoV-SSI results.pdf` | Structural model and reference results. |
| `Classical_OMA_results/` | CoV-SSI and SOBI comparison CSVs. |
| `NOMAD_Net_model/Model.pt` | Bundled PyTorch model checkpoint. |
| `NOMAD_Net_model/Model_Best_Weights.h5` | Saved PyTorch `state_dict` (despite the `.h5` filename, this is not an HDF5 file). |
| `requirements.txt` | Python packages needed for the scripts. |

## Getting started

Run the following from the repository root. A recent Python 3 environment is required; tested dependency versions have not yet been pinned.

```bash
git clone https://github.com/IMIS-Lab-UGR/NOMAD_Net.git
cd NOMAD_Net
python -m venv .venv
```

Activate the environment (Windows PowerShell: `.venv\Scripts\Activate.ps1`; Linux/macOS: `source .venv/bin/activate`) and then install dependencies:

```bash
python -m pip install -r requirements.txt
```

### Use the pretrained model

```bash
python NOMAD_Net_Use_model.py
```

This uses `Input_signals/Simulated_records.csv`, loads `NOMAD_Net_model/Model.pt`, and displays diagnostic plots and estimated natural frequencies (Hz) and damping ratios (%). The script is currently an example-oriented research workflow rather than a packaged command-line tool.

### Retrain from the numerical example

```bash
python NOMAD_Net.py
```

Training uses four modal components, 80 Hankel lags and an initial sampling frequency of 100 Hz (reduced to 50 Hz after the configured downsampling). It writes diagnostic files under `results/` and `training_plots/` and **overwrites** the model files in `NOMAD_Net_model/`. Back up the supplied checkpoint if you want to preserve it.

Training may be computationally expensive, especially without a GPU.

## Input data and reference formats

- `Simulated_records.csv` has four channels (`Ch.1`–`Ch.4`), in acceleration units of `g`, in the order documented in `Input_signals/Model and CoV-SSI results.pdf`.
- Classical OMA CSVs contain one identified mode per row: frequency, damping ratio, and four complex modal displacements in the remaining columns.
- The original example and current scripts assume these four channels and the associated preprocessing settings. Other cases require adapting the code and configuration.

## Reproducibility and limitations

The scripts are provided as research code. Automated tests, a locked environment and validated cross-platform executions are not yet included. The inference script currently loads its bundled checkpoint with `torch.load(..., weights_only=False)`, which must only be used with **trusted** checkpoint files.

## Citation and license

A formal software citation and publication DOI will be added once bibliographic details are confirmed. Please contact the project maintainers about reuse conditions: **no open-source license has yet been selected**.
