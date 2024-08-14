import pandas as pd
import os
import time
import concurrent.futures
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client with API key from environment variables
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Load the CSV file
df = pd.read_csv('combined_case_presentations.csv', encoding='utf-8')


# Function to call OpenAI API with gpt-4-turbo
def get_case_analysis(case_presentation):
    prompt = (
        f"""Below is a case presentation of a patient. Please remove any reference to Figures or tables, and remove any explicit reference to the final diagnosis from the text. 
         Additionally, remove any information about the patient's condition or treatment after the final diagnosis is made.
         Return both the final diagnosis and the clean text separately as follows: Clean text: <clean text> Final diagnosis: <final diagnosis>.
         Case presentation: {case_presentation}"""
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system",
             "content": "You are an experienced physician"},
            {"role": "user", "content": prompt}
        ],
        max_tokens=4000,
        temperature=0.2
    )
    print(response)
    return response.choices[0].message.content


# Function to process each case presentation and return the result
def process_case(case_presentation):
    result = get_case_analysis(case_presentation)
    try:
        clean_text_start = result.index('Clean text:') + len('Clean text:')
        clean_text_end = result.index('Final diagnosis:')
        clean_text = result[clean_text_start:clean_text_end].strip()

        final_diagnosis_start = result.index('Final diagnosis:') + len('Final diagnosis:')
        final_diagnosis = result[final_diagnosis_start:].strip()
    except ValueError:
        # In case of unexpected format, handle error
        clean_text = ''
        final_diagnosis = ''

    return {
        'Case presentation': case_presentation, 
        'clean text': clean_text,
        'final diagnosis': final_diagnosis
    }


# Control the rate of API calls by setting max_workers and introducing delay
max_workers = 10  # Maximum number of parallel requests
delay_between_requests = 1  # Delay in seconds between requests

# Process each case presentation using concurrent futures
results = []
checkpoint_interval = 20  # Number of iterations after which to save the checkpoint

with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    future_to_case = {executor.submit(process_case, case_presentation): case_presentation for case_presentation in
                      df['Case presentation']}

    for i, future in enumerate(concurrent.futures.as_completed(future_to_case)):
        case_presentation = future_to_case[future]
        try:
            result = future.result()
            results.append(result)
        except Exception as exc:
            print(f'Case presentation generated an exception: {exc}')

        # Introduce delay to control the rate of API calls
        time.sleep(delay_between_requests)

        # Save checkpoint every 'checkpoint_interval' iterations
        if (i + 1) % checkpoint_interval == 0:
            checkpoint_df = pd.DataFrame(results)
            checkpoint_filename = f'Processed_Case_Reports_checkpoint_{i + 1}.csv'
            checkpoint_df.to_csv(checkpoint_filename, index=False, encoding='utf-8')
            print(f"Checkpoint saved to '{checkpoint_filename}'.")

# Save final results to a new CSV file with UTF-8 encoding
results_df = pd.DataFrame(results)
results_df.to_csv('Processed_Case_Reports.csv', index=False, encoding='utf-8')

print("Processing complete. Results saved to 'Processed_Case_Reports.csv'.")
