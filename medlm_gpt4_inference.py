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
ds = pd.read_csv('Case_report_w_images_dis_VF.csv')


def process_batch(batch):
    results = []
    for _, row in batch.iterrows():
        case_presentation = row['clean text']
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
                print(f"API error: {e}. Waiting for 60 seconds before retrying.")
                time.sleep(60)

        results.append({
            'Case presentation': case_presentation,
            'True diagnosis': true_diagnosis,
            'Generated diagnosis': generated_diagnosis
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
model_type = "microsoft/deberta-xlarge-mnli"
predictions = results_df['Generated diagnosis'].tolist()
references = results_df['True diagnosis'].tolist()
P, R, F1 = bert_score.score(predictions, references, lang="en", model_type=model_type)

# Add BERTScore F1 to the DataFrame
results_df['BERTScore F1'] = F1.tolist()

# Save the DataFrame to a CSV file
results_df.to_csv('gpt4_free_text_batched.csv', index=False)
