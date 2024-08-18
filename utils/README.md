# Utils Directory

Utilities directory for the CUPCase dataset. 
This directory contains all the utility scripts used in the CUPCase Paper.

## Installation

To get started, you'll need to install the required dependencies. Run the following command:

```bash
pip install -r requirements.txt
```

## Dataset

Please ensure that the CUPCase dataset is placed in the /datasets folder within this directory.

## Usage

Utils are used manually based on the utilities you wish to perform, to use a util run a script like:

```bash
python preprocess_w_images.py
```

## Scripts

The scripts available in this directory are:

### Convert csv to Hugging Face dataset  

Converts a csv to an .arrow file.
To do this run:
```bash
convert_csv_to_hfds.py
```

### Ablation study generation

Generates the datasets (.arrow) file for the ablation study done in the paper.
To generate it, run:
```bash
generate_ablation_study.py
```

### Plot
this script generates and saves the F1-BERT-SCORE plot between the correct diagnosis and distractors\
to generate the plot run:
```bash
plots.py
```
the plots will be saved in the /output directory

### Postprocessing distractors

This utility is used to separate the top-5 distractors column into separate columns for ease of evaluation.\
to do this run:

```bash
postprocess_distractors.py
```

### General statistics

To generate the general statistics in the CUPCase paper for reproducing or your own dataset\
run:

```bash
generate_statistics.py
```