# Joule Ecosystem: Measuring and Optimizing Edge AI Energy Consumption

> **Internship Report & Practical Guide**  
> **Author:** Daniele Famà ([daniele.fama@telecom-paris.fr](mailto:daniele.fama@telecom-paris.fr))  
> **Institution:** MultiMedia Lab, Image, Data and Signal Department (IDS), LTCI, Télécom Paris, Institut Polytechnique de Paris  
> **Term:** Summer Internship 2026  

📄 **Read the full compiled report:** [**`report_jouleecosystem.pdf`**](report_jouleecosystem.pdf)

---

## Overview

Deploying Deep Neural Networks on resource-constrained edge hardware requires a delicate balance between predictive accuracy and energy efficiency. Most neural architecture search (NAS) methods rely on theoretical proxies (e.g., FLOPs, parameter counts) or device datasheets, which fail to capture memory hierarchies, runtime scheduling overheads, and hardware-specific power dynamics.

The **Joule Ecosystem** provides an end-to-end, physically grounded pipeline to measure, model, and minimize the energy consumption of neural network inference on embedded edge devices (such as the **Raspberry Pi 5**, **NVIDIA Jetson Nano**, and **NVIDIA Jetson AGX Orin**).

```text
+------------------------+      energy_lookup_table.csv      +------------------------+
|       JouleQuest       | --------------------------------> |       JouleGrad        |
|  Physical Measurement  |                                   | Differentiable API     |
+------------------------+                                   +------------------------+
                                                                         |
                                                                         | \lambda_E * E_hat
                                                                         v
+------------------------+        Materialized Models        +------------------------+
|  Physical Validation   | <-------------------------------- |        JouleNAS        |
|  (Real Hardware Test)  |                                   | Architecture Search    |
+------------------------+                                   +------------------------+
```

### The Three Pillars

1. **JouleQuest (Physical Measurement & Acquisition Layer):**
   - Open-source framework ([github.com/danielefam/joulequest](https://github.com/danielefam/joulequest)) automating high-precision power and energy measurement on edge devices using Texas Instruments **INA226EVM** sensors.
   - Implements automated calibration, sub-millisecond clock synchronization across hosts, and adaptive burst sizing to maintain reliable sample counts across wide dynamic ranges.
   - Produces empirical, configuration-specific lookup tables (`energy_lookup_table.csv`) for basic layers (`Linear`, `Conv2d`, `Attention`) and full networks at batch size 1.

2. **JouleGrad (Differentiable Estimation Layer):**
   - Standalone Python library ([github.com/danielefam/joulegrad](https://github.com/danielefam/joulegrad)) providing differentiable energy estimates directly from measurement tables.
   - Computes exact autograd gradients for layer widths and channels via multilinear interpolation.
   - Enforces rigorous out-of-range policies (`error`, `warn`, `clamp`, `extrapolate`) and eliminates synthetic assumptions (no unmeasured zero-cost anchors).

3. **JouleNAS (Environmentally-Aware Architecture Search):**
   - Integrates differentiable energy penalties directly into the loss function:
     $$\mathcal{L} = \mathcal{L}_{\text{CE}} + \lambda_E\,\widehat{E}$$
   - Uses hard Straight-Through Estimator (STE) channel masks to optimize network topology for specific target hardware while keeping model weights frozen.
   - Evaluated on **SimpleConv** (MNIST/synthetic bars), **CIFAR-10 ResNet-18**, and **Imagenette ResNet-18**.

4. **The Bridge (Materialization & Physical Validation):**
   - Connects channel masks to physical deployment by materializing smaller networks with adjusted channel dimensions, residual shortcuts ($1\times 1$ convolutions), and matching BatchNorm layers.
   - Hardware validation on the **Raspberry Pi 5** confirmed a **60.08% real-world energy reduction** (11.33 mJ vs 28.38 mJ baseline) with only a **0.04% loss in test accuracy** (92.27% vs 92.31%).

---

## Report Structure

The master document [`report_jouleecosystem.tex`](report_jouleecosystem.tex) is structured modularly inside the [`chapters/`](chapters/) directory:

| Chapter | Source File | Description |
| :--- | :--- | :--- |
| **Front Matter** | [`chapters/title.tex`](chapters/title.tex), [`abstract.tex`](chapters/abstract.tex) | Title page, institution affiliation, and executive abstract. |
| **Introduction** | [`chapters/introduction.tex`](chapters/introduction.tex) | High-level problem statement, project objectives, and guide for readers. |
| **Chapter 1: JouleQuest** | [`chapters/joulequest.tex`](chapters/joulequest.tex) | Hardware wiring (low/high power), INA226 acquisition, adaptive burst sizing, clock sync, CLI test harnesses, and table generation. |
| **Chapter 2: JouleGrad** | [`chapters/joulegrad.tex`](chapters/joulegrad.tex) | Mathematical formulation, multilinear interpolation, boundary policies, and migration from legacy estimators. |
| **Chapter 3: JouleNAS** | [`chapters/joulenas.tex`](chapters/joulenas.tex) | Channel masking formulation, Straight-Through Estimators, Pareto frontiers, temperature schedules, and experiments. |
| **Chapter 4: The Bridge** | [`chapters/bridge.tex`](chapters/bridge.tex) | Reconstructing materialized models from mask logs, handling residual shortcuts, and real-world physical validation. |
| **Chapter 5: Future Work** | [`chapters/future-work.tex`](chapters/future-work.tex) | Composition discrepancies (sum of parts vs full networks), cache effects, denser grids, and multi-board comparisons. |
| **Chapter 6: Conclusion** | [`chapters/conclusion.tex`](chapters/conclusion.tex) | Summary of achievements and takeaways for future edge AI research. |
| **Back Matter** | [`chapters/ref.tex`](chapters/ref.tex), [`backcover.tex`](chapters/backcover.tex) | Bibliography references, list of figures/tables, and back cover. |

---

## Building the Report

### Prerequisites

To compile the LaTeX source to PDF, you need:
- A TeX distribution (e.g., TeX Live / MacTeX / MikTeX) including:
  - `pdflatex`
  - `bibtex`
  - `latexmk`
- `inkscape` (required by the LaTeX `svg` package to convert SVG graphics during compilation).

On Ubuntu / Debian:
```bash
sudo apt update
sudo apt install texlive-latex-extra texlive-fonts-recommended texlive-fonts-extra latexmk inkscape
```

### Compiling via CLI

The simplest and recommended method is using `latexmk`, which respects [`.latexmkrc`](.latexmkrc) and automatically handles multi-pass compilation and bibliography generation:

```bash
latexmk -pdf -shell-escape report_jouleecosystem.tex
```

Alternatively, compile manually with `pdflatex` and `bibtex`:

```bash
pdflatex -shell-escape report_jouleecosystem.tex
bibtex report_jouleecosystem
pdflatex -shell-escape report_jouleecosystem.tex
pdflatex -shell-escape report_jouleecosystem.tex
```

To clean auxiliary build files:
```bash
latexmk -c
```

### Compiling in Visual Studio Code

This repository includes preconfigured settings in [`.vscode/settings.json`](.vscode/settings.json) for the **LaTeX Workshop** extension:
1. Open the project folder in VS Code.
2. Ensure the `LaTeX Workshop` extension is installed.
3. The build recipe `pdflatex -> bibtex -> pdflatex x2` with `-shell-escape` is enabled by default and will trigger automatically on save.

---

## Directory Organization

```text
report_daniele_fama/
├── .latexmkrc                   # latexmk configuration setting jobname
├── .vscode/                     # VS Code LaTeX Workshop build settings
├── chapters/                    # Report content organized by chapter
│   ├── title.tex                # Title page
│   ├── abstract.tex             # Abstract
│   ├── introduction.tex         # Introduction & project overview
│   ├── joulequest.tex           # Ch. 1: JouleQuest (Measurement)
│   ├── joulegrad.tex            # Ch. 2: JouleGrad (Estimation)
│   ├── joulenas.tex             # Ch. 3: JouleNAS (Search)
│   ├── bridge.tex               # Ch. 4: The Bridge & Physical Validation
│   ├── future-work.tex          # Ch. 5: Future research directions
│   ├── conclusion.tex           # Ch. 6: Final conclusions
│   ├── ref.tex                  # References inclusion
│   └── backcover.tex            # Back cover
├── references.bib               # BibTeX bibliography database
├── report_jouleecosystem.tex    # Main LaTeX root document
├── report_jouleecosystem.pdf    # Compiled PDF report
└── README.md                    # This document
```

---

## Related Repositories

- **JouleQuest**: Hardware measurement and automated benchtop acquisition layer — [github.com/danielefam/joulequest](https://github.com/danielefam/joulequest)
- **JouleGrad**: Differentiable Energy Estimator for PyTorch — [github.com/danielefam/joulegrad](https://github.com/danielefam/joulegrad)
- **JouleNAS**: Neural Architecture Search with hardware-measured energy penalties *(private repository)*
