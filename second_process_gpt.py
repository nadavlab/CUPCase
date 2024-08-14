import pandas as pd
import os
import time
import concurrent.futures
from dotenv import load_dotenv
from openai import OpenAI
import time

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client with API key from environment variables
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Load the CSV file
df = pd.read_csv('case_reports_with_top_5_final.csv', encoding='utf-8')


# Function to call OpenAI API with gpt-4-turbo
def get_case_analysis(case_presentation, final_diagnosis):
    prompt = (
        f"""Here is a case presentation, please remove any reference to the final diagnosis: {final_diagnosis} 
        return the result as follows - Clean text: <clean_text> end.\n{case_presentation}"""
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system",
             "content": "You are an experienced physician"},
            {"role": "user", "content": prompt}
        ],
        max_tokens=4000,
        temperature=0.5
    )
    print(response)
    return response.choices[0].message.content


# Function to process each case presentation and return the result
def process_case(case_presentation, final_diagnosis, max_retries=50, retry_delay=8):
    for attempt in range(max_retries):
        try:
            result = get_case_analysis(case_presentation, final_diagnosis)
            clean_text_start = result.index('Clean text:') + len('Clean text:')
            clean_text_end = result.index('end.')
            clean_text = result[clean_text_start:clean_text_end].strip()
            # Check if final_diagnosis is still in clean_text
            flag = final_diagnosis in clean_text
            return {
                'clean text': clean_text,
                'full': result,
                'flag': flag
            }
        except ValueError:
            # In case of unexpected format, handle error
            clean_text = ''
            flag = False
            return {
                'clean text': clean_text,
                'full': result,
                'flag': flag
            }
        except Exception as exc:
            print(f'Attempt {attempt + 1} failed with exception: {exc}')
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                raise


max_workers = 5  # Maximum number of parallel requests
delay_between_requests = 5  # Delay in seconds between requests

# Process each case presentation using concurrent futures
results = []
checkpoint_interval = 20  # Number of iterations after which to save the checkpoint

with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    future_to_case = {executor.submit(process_case, row['clean text'], row['final diagnosis']): row for _, row in df.iterrows()}

    for i, future in enumerate(concurrent.futures.as_completed(future_to_case)):
        row = future_to_case[future]
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
            checkpoint_filename = f'second_process_gpt_checkpoint_{i + 1}.csv'
            checkpoint_df.to_csv(checkpoint_filename, index=False, encoding='utf-8')
            print(f"Checkpoint saved to '{checkpoint_filename}'.")

# Save final results to a new CSV file with UTF-8 encoding
results_df = pd.DataFrame(results)
results_df.to_csv('second_process_gpt.csv', index=False, encoding='utf-8')

print("Processing complete. Results saved to 'second_process_gpt.csv'.")
