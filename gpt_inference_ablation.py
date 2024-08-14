import pandas as pd
import bert_score
import os
from dotenv import load_dotenv
from openai import OpenAI
import time
import random
import numpy as np
import torch

load_dotenv()

# Set up OpenAI API key
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Read the CSV file
ds = pd.read_csv('ablation_study_tokens.csv')

# Set the device to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


def process_batch(batch):
    results = []
    for _, row in batch.iterrows():
        case_presentation = row['80%']
        true_diagnosis = row['final diagnosis']
        prompt = f"Predict the diagnosis of this case presentation of a patient. Return the final diagnosis in one concise sentence without any further elaboration.\nFor example: <diagnosis name here>\nCase presentation: {case_presentation}\nDiagnosis:"

        while True:
            try:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0
                )
                generated_diagnosis = response.choices[0].message.content.strip()
                break
            except Exception as e:
                print(f"API error: {e}. Waiting for 20 seconds before retrying.")
                time.sleep(20)

        results.append({
            'Case presentation': case_presentation,
            'True diagnosis': true_diagnosis,
            'Generated diagnosis': generated_diagnosis
        })
        time.sleep(1)  # Sleep for 1 second between each API call

    return results


def compute_bert_score(predictions, references, batch_size=250):
    model_type = "microsoft/deberta-xlarge-mnli"
    _, _, F1 = bert_score.score(predictions, references, lang="en", model_type=model_type, device=device,
                                batch_size=batch_size)
    return F1


all_results = []
batch_f1_scores = []

for batch_num in range(4):
    print(f"Processing batch {batch_num + 1}/4")
    batch = ds.sample(n=250, random_state=batch_num)
    batch_results = process_batch(batch)
    all_results.extend(batch_results)

    # Calculate BERTScore F1 for this batch
    predictions = [result['Generated diagnosis'] for result in batch_results]
    references = [result['True diagnosis'] for result in batch_results]

    F1 = compute_bert_score(predictions, references)

    batch_mean_f1 = F1.mean().item()
    batch_f1_scores.append(batch_mean_f1)

    print(f"Completed batch {batch_num + 1}/4 with mean F1 score: {batch_mean_f1:.4f}")
    time.sleep(5)  # Sleep for 10 seconds between batches

# Convert results to DataFrame
results_df = pd.DataFrame(all_results)

# Calculate BERTScore F1 using DeBERTa model for all results
predictions = results_df['Generated diagnosis'].tolist()
references = results_df['True diagnosis'].tolist()
F1 = compute_bert_score(predictions, references)

# Add BERTScore F1 to the DataFrame
results_df['BERTScore F1'] = F1.tolist()

# Calculate mean and standard deviation of batch F1 scores
mean_f1 = np.mean(batch_f1_scores)
std_f1 = np.std(batch_f1_scores)

print(f"\nMean F1 score across all batches: {mean_f1:.4f}")
print(f"Standard deviation of F1 scores: {std_f1:.4f}")

# Save the DataFrame to a CSV file
results_df.to_csv('gpt_free_text_ablation_80.csv', index=False)

# Save batch F1 scores to a separate CSV file
batch_scores_df = pd.DataFrame({
    'Batch': range(1, 5),
    'Mean F1 Score': batch_f1_scores
})
batch_scores_df.to_csv('batch_f1_scores_80.csv', index=False)

print("\nResults saved to 'gpt_free_text_ablation_80.csv'")
print("Batch F1 scores saved to 'batch_f1_scores.csv'")