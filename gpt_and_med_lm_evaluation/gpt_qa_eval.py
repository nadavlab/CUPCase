import pandas as pd
import bert_score
import os
from dotenv import load_dotenv
from openai import OpenAI
import time
import random

load_dotenv()

# Set up OpenAI API key
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Read the CSV file
ds = pd.read_csv('ablation_study_tokens.csv')

def process_batch(batch):
    results = []
    for _, row in batch.iterrows():
        case_presentation = row['case presentation']
        true_diagnosis = row['final diagnosis']
        distractor2 = row['distractor2']
        distractor3 = row['distractor3']
        distractor4 = row['distractor4']

        options = [true_diagnosis, distractor2, distractor3, distractor4]
        random.shuffle(options)
        options_text = "\n".join([f"{i+1}. {option}" for i, option in enumerate(options)])
        prompt = (f"Predict the diagnosis of this case presentation of a patient. Return only the correct index from the following list, for example: 3\n"
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

all_results = []

for batch_num in range(4):
    print(f"Processing batch {batch_num + 1}/4")
    # Randomly sample 250 rows
    batch = ds.sample(n=250, random_state=batch_num)

    batch_results = process_batch(batch)
    all_results.extend(batch_results)

    print(f"Completed batch {batch_num + 1}/4")
    time.sleep(10)  # Sleep for 10 seconds between batches

# Convert results to DataFrame
results_df = pd.DataFrame(all_results)

# Calculate BERTScore F1 using DeBERTa model
# model_type = "microsoft/deberta-xlarge-mnli"
# predictions = results_df['Generated diagnosis'].tolist()
# references = results_df['True diagnosis'].tolist()
# P, R, F1 = bert_score.score(predictions, references, lang="en", model_type=model_type)

# # Add BERTScore F1 to the DataFrame
# results_df['BERTScore F1'] = F1.tolist()

# Save the DataFrame to a CSV file
results_df.to_csv('gpt4_multiple_choice_batched.csv', index=False)

# Calculate accuracy
accuracy = sum(results_df['Correct']) / len(results_df)
print(f"Accuracy: {accuracy:.2f}")
