# Preprocess Directory

Preprocess directory for the CUPCase dataset. 
This directory contains all the tools and scripts necessary to recreate the preprocessing done in the CUPCase paper.

## Installation

To get started, you'll need to install the required dependencies. Run the following command:

```bash
pip install -r requirements.txt
```

## Dataset

Please ensure that the raw CUPCase dataset is placed in the /datasets folder within this directory.

## API keys

To use gpt-4o-mini (or any OpenAI model) for preprocessing, please create and place your API_KEY in a .env file.
```bash
OPENAI_API_KEY="YOUR_API_KEY_HERE"
```
## Usage

To perform the preprocessing, run the specific processing script you wish to use. For example:

```bash
python preprocess_w_images.py
```

## Scripts

The scripts available in this directory are:

### Embedding the ICD-10-CM codes' longdescriptions 

Uses the JINA AI embedding model to embed the long descriptions of the ICD-10-CM codes\
long descriptions, embedding works in batches, to change a batch size, change the "batch_size" parameter\
in the batch_encode function. To start preprocessing, make sure the ICD-10-CM csv is placed in the /datasets folder\
and run the following script:

```bash
embedding_batches.py
```

### Removing final diagnosis from case presentations

to remove the final diagnoses from the case presentations, make sure the csv is placed in /datasets folder\
and run the following script:
```bash
gpt_free_text_eval.py
```
To perform the second preprocessing step, which ensure the removal of the final diagnosis\
run the following script over the resulting csv from the previous step:\

```bash
second_preprocess_w_images.py
```

The resulting dataset (csv file) will be saved to the /output directory.
