# Trajectory Prediction Assignment

This project is a modified version of the paper ["TUTR: Trajectory Unified Transformer for Pedestrian Trajectory Prediction"](https://openaccess.thecvf.com/content/ICCV2023/papers/Shi_Trajectory_Unified_Transformer_for_Pedestrian_Trajectory_Prediction_ICCV_2023_paper.pdf).
The original code can be accessed [here](https://github.com/lssiair/TUTR?tab=readme-ov-file#tutr-trajectory-unified-transformer-for-pedestrian-trajectory-prediction).

## Setup

```bash
conda create -n tutr python=3.9
conda activate tutr
pip install -r requirements.txt
```

## Usage

To run the all 16 modified configurations on a local GPU, run the following:

```bash
./run_analysis.sh
```

To run all 16 modified configurations as job array on the UCF Newton GPU cluster, run the following:

```bash
sbatch run_analysis.sbatch
```

## Report

To build the LaTeX report, run `latexmk` from within the `./report` directory.
