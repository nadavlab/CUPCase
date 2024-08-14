import sacrebleu
from collections.abc import Iterable
import pandas as pd

def is_non_str_iterable(obj):
    return isinstance(obj, Iterable) and not isinstance(obj, str)

def _sacreformat(refs, preds):
    """Format refs and preds for sacrebleu corpus calculation. It is very particular"""
    # Sacrebleu expects (List[str], List[List[str])
    #   e.g. sacrebleu.corpus_bleu([pred_t], [[ref1_stream], [ref2_stream], ...])

    # Note [ref1_stream] is the first reference for each pred.
    # So lists are size N and (M, N) for N preds and M possible refs for each pred
    # This is a different order of dimensions that I would expect

    # We expect refs to be List[str] or List[List[str]], the outer list corresponding to preds
    # Must become List[List[str]] with the inner list corresponding to preds
    if not is_non_str_iterable(refs):
        refs = list(refs)
    if not is_non_str_iterable(refs[0]):
        refs = [[ref] for ref in refs]
    refs = list(zip(*refs))
    # Note the number of refs in each ref list much match the number of preds

    # We expect preds to be List[str] or List[List[str]]. Must become List[str]
    if not is_non_str_iterable(preds):
        preds = list(preds)
    if is_non_str_iterable(preds[0]):
        assert len(preds[0]) == 1, f"Pred must be a str, was {preds[0]}"
        preds = [pred[0] for pred in preds]

    return refs, preds


def bleu(items):
    """The Bilingual Evaluation Understudy Score, or BLEU for short, is a metric
    for evaluating a generated sentence to a reference sentence. It counts matching
    n-grams in the candidate translation to n-grams in the reference text, where
    1-gram or unigram would be each token and a bigram comparison would be each
    word pair. The comparison is made regardless of word order
    Source: https://machinelearningmastery.com/calculate-bleu-score-for-text-python/
    Paper: https://www.aclweb.org/anthology/P02-1040/

    Higher is better
    """
    refs = list(zip(*items))[0]
    preds = list(zip(*items))[1]
    refs, preds = _sacreformat(refs, preds)
    return sacrebleu.corpus_bleu(preds, refs).score


def main():
    df = pd.read_csv('free_text_medlm_large.csv')

    true_diagnoses = df['True diagnosis'].str.lower().str.replace('.', '', regex=False).tolist()
    generated_diagnoses = df['Generated diagnosis'].str.lower().str.replace('.', '', regex=False).tolist()

    bleu_scores = []
    for true, generated in zip(true_diagnoses, generated_diagnoses):
        score = bleu([(true, generated)])
        bleu_scores.append(score)

    df['BLEU Score'] = bleu_scores

    df.to_csv('medlm_with_bleu_scores.csv', index=False)

    print("BLEU scores have been calculated and added to the CSV file.")

if __name__ == "__main__":
    main()
