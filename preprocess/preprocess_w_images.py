import pandas as pd
import os
import time
import concurrent.futures
from dotenv import load_dotenv
from openai import OpenAI
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file
load_dotenv()

# Check for API key
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("API key not found. Please set the 'OPENAI_API_KEY' environment variable.")

# Initialize OpenAI client with API key from environment variables
client = OpenAI(api_key=api_key)

# Load the CSV file
df = pd.read_excel('dataset/combined_case_presentations.xlsx')


# Function to call OpenAI API with gpt-4-turbo
def get_case_analysis(case_presentation, retries=3):
    prompt = (
        f"""Below is a case presentation of a patient, please remove any explicit reference to the final diagnosis from the text. 
        Additionally, remove any information about the patient's condition or treatment after the final diagnosis is made. 
        Do not remove any references to Figures or Images in the text like (Fig 1.)
        Return both the final diagnosis and the clean text separately as follows:
        Clean text: <clean text>
        Final diagnosis: <final diagnosis>.
        Here is the Case presentation: {case_presentation}"""
    )

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an experienced physician."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=4000,
                temperature=0.2
            )
            logging.info(response)
            return response.choices[0].message.content
        except Exception as e:
            if 'rate limit' in str(e).lower():
                logging.warning(f"Rate limit error encountered. Retrying... ({attempt + 1}/{retries})")
                time.sleep(15)  # Wait for 60 seconds before retrying
            else:
                logging.error(f"Error in API call: {e}")
                break
    raise Exception("Failed to get response from OpenAI API after several retries")


# Function to process each case presentation and return the result
def process_case(case_presentation):
    result = get_case_analysis(case_presentation)
    try:
        clean_text_start = result.index('Clean text:') + len('Clean text:')
        clean_text_end = result.index('Final diagnosis:')
        clean_text = result[clean_text_start:clean_text_end].strip()

        final_diagnosis_start = result.index('Final diagnosis:') + len('Final diagnosis:')
        final_diagnosis = result[final_diagnosis_start:].strip()
    except ValueError as e:
        logging.error(f"Error processing case presentation: {e}")
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
                      df['Case Presentation']}

    for i, future in enumerate(concurrent.futures.as_completed(future_to_case)):
        case_presentation = future_to_case[future]
        try:
            result = future.result()
            results.append(result)
        except Exception as exc:
            logging.error(f'Case presentation generated an exception: {exc}')

        # Introduce delay to control the rate of API calls
        time.sleep(delay_between_requests)

        # Save checkpoint every 'checkpoint_interval' iterations
        if (i + 1) % checkpoint_interval == 0:
            checkpoint_df = pd.DataFrame(results)
            checkpoint_filename = f'output/Processed_Case_Reports_checkpoint_{i + 1}.csv'
            checkpoint_df.to_csv(checkpoint_filename, index=False, encoding='utf-8')
            logging.info(f"Checkpoint saved to '{checkpoint_filename}'.")

# Convert the results list to a DataFrame
results_df = pd.DataFrame(results)

# Add the 'Image Filenames' column from the original DataFrame
results_df['Image Filenames'] = df['Image Filenames']

# Save final results to a new Excel file with UTF-8 encoding
results_df.to_excel('output/Processed_Case_Reports_w_images.xlsx', index=False)

logging.info("Processing complete. Results saved to 'Processed_Case_Reports_w_images.xlsx'.")
