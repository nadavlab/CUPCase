import pandas as pd
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
        distractor2 = row['distractor2']
        distractor3 = row['distractor3']
        distractor4 = row['distractor4']
        options = [true_diagnosis, distractor2, distractor3, distractor4]
        random.shuffle(options)
        options_text = "\n".join([f"{i + 1}. {option}" for i, option in enumerate(options)])
        prompt = (
            f"Predict the diagnosis of this case presentation of a patient. Return only the correct index from the following list, for example: 3\n"
            f"{options_text}\nCase presentation: {case_presentation}")

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
                try:
                    predicted_index = int(generated_diagnosis[0]) - 1
                except Exception as e:
                    predicted_index = -1
                    print(e)
                break
            except Exception as e:
                print(f"API error: {e}. Waiting for 60 seconds before retrying.")
                time.sleep(60)

        results.append({
            'Case presentation': case_presentation,
            'True diagnosis': true_diagnosis,
            'Generated diagnosis': generated_diagnosis,
            'Correct index': options.index(true_diagnosis),
            'Predicted index': predicted_index,
            'Correct': options.index(true_diagnosis) == predicted_index
        })
        time.sleep(1)  # Sleep for 1 second between each API call

    return results


def compute_accuracy(results):
    correct = sum(1 for result in results if result['Correct'])
    return correct / len(results)


all_results = []
batch_accuracies = []

for batch_num in range(4):
    print(f"Processing batch {batch_num + 1}/4")
    batch = ds.sample(n=250, random_state=batch_num)
    batch_results = process_batch(batch)
    all_results.extend(batch_results)

    # Calculate accuracy for this batch
    batch_accuracy = compute_accuracy(batch_results)
    batch_accuracies.append(batch_accuracy)
    print(f"Completed batch {batch_num + 1}/4 with accuracy: {batch_accuracy:.4f}")

    time.sleep(5)  # Sleep for 5 seconds between batches

# Convert results to DataFrame
results_df = pd.DataFrame(all_results)

# Calculate mean and standard deviation of batch accuracies
mean_accuracy = np.mean(batch_accuracies)
std_accuracy = np.std(batch_accuracies)

print(f"\nMean accuracy across all batches: {mean_accuracy:.4f}")
print(f"Standard deviation of accuracies: {std_accuracy:.4f}")

# Save the DataFrame to a CSV file
results_df.to_csv('gpt_multiple_choice_ablation_80.csv', index=False)

# Save batch accuracies to a separate CSV file
batch_scores_df = pd.DataFrame({
    'Batch': range(1, 5),
    'Accuracy': batch_accuracies
})
batch_scores_df.to_csv('batch_accuracies_80.csv', index=False)

print("\nResults saved to 'gpt_multiple_choice_ablation_80.csv'")
print("Batch accuracies saved to 'batch_accuracies_80.csv'")