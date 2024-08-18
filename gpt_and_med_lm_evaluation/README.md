# Eval Directory

Eval directory for the CUPCase dataset. 
This directory contains all the tools and scripts necessary to recreate the evaluation in the CUPCase paper.

## Installation

To get started, you'll need to install the required dependencies. Run the following command:

```bash
pip install -r requirements.txt
```

## Dataset

Please ensure that the CUPCase dataset is placed in the /datasets folder within this directory.

## API keys

To use gpt-4o (or any OpenAI model) for evaluation, please create and place your API_KEY in a .env file.
```bash
OPENAI_API_KEY="YOUR_API_KEY_HERE"
```
## Usage

To perform evaluation, run the specific evaluation script you wish to use. For example:

```bash
python gpt_free_text_eval.py
```

## Scripts

The scripts available in this directory are:

## GPT-4o
### Multiple-choice evaluation 

Used for evaluation GPT-4o with the multiple-choice QA in CUPCase.
Run 
```bash
gpt_qa_eval.py
```

### Open-ended evaluation

Used for evaluation GPT-4o with the open-ended QA in CUPCase.\
Run
```bash
gpt_free_text_eval.py
```
### Boostrap sampling calculate mean, std

Used for calculating the mean and standard deviation of the 4 bootstrap sampling iterations of 250 samples each. \
Run
```bash
bootstrap_sampling_mean_std.py
```

## MedLM-Large

To evaluate CUPCase using MedLM-Large, follow the instructions in the .ipynb.
```bash
medlm_inference.ipynb
```
MedLM-Large is a closed source model, requires specific access from Google to use.
And is most easily accessible through Google Colab.