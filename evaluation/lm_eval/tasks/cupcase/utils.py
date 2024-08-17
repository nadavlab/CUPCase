import os

import evaluate
import numpy as np


def bert_score(predictions, references):
    return predictions[0], references[0]


def agg_bert_score(items):
    bert_score = evaluate.load('bertscore', device='cuda:0')
    predictions, references = zip(*items)
    print(items)
    results = bert_score.compute(predictions=predictions, references=references,
                                model_type='microsoft/deberta-xlarge-mnli')
    bert_score_f1 = np.mean(results["f1"])
    return bert_score_f1

def process_docs(dataset):
    print('in process docs!')
    return dataset.shuffle(seed=int(os.environ["DATA_SEED"]))