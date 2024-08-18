import pandas as pd
import numpy as np

gpt_qa = pd.read_csv('dataset/gpt4_multiple_choice_batched.csv')
gpt_free = pd.read_csv('dataset/gpt4_free_text_batched.csv')
medlm_qa = pd.read_csv('dataset/qa_medlm_large.csv')
medlm_free = pd.read_csv('dataset/free_text_medlm_large.csv')

def calc_qa(df):
    # convert true false to numbers
    acc = []
    for x in df['Correct']:
        if x:
            acc.append(1)
        else:
            acc.append(0)
    chunk_size = 250
    df['acc'] = acc
    chunks = [df[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
    means = [np.mean(chunk['acc']) for chunk in chunks]
    std = np.std(means)
    print("Mean: {:.5f}, Std: {:.5f}".format(np.mean(means), std))
calc_qa(gpt_qa)


def calc_freetext(df):
    # convert true false to numbers
    chunk_size = 250
    chunks = [df[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
    means = [np.mean(chunk['BERTScore F1']) for chunk in chunks]
    std = np.std(means)
    print("Mean: {:.5f}, Std: {:.5f}".format(np.mean(means), std))
calc_freetext(gpt_free)

