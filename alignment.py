import pandas as pd
from fuzzywuzzy import process

# Load the CSV files
df_clean_text = pd.read_csv('second_process_gpt.csv')
df_case_presentation = pd.read_csv('case_reports_with_top_5_final.csv')
# Define a function to get the first 10 words
def get_first_n_words(text, n=10):
    return ' '.join(text.split()[:n])

# Create a new column in both DataFrames with the first 10 words
df_case_presentation['first_10_words'] = df_case_presentation['Case presentation'].apply(lambda x: get_first_n_words(x, 10))
df_clean_text['first_10_words'] = df_clean_text['clean text'].apply(lambda x: get_first_n_words(x, 10))

# Function to match case presentations to clean texts using fuzzy matching
def fuzzy_match(case_text, choices, threshold=80):
    match, score = process.extractOne(case_text, choices)
    if score >= threshold:
        return match
    else:
        return None

# Get a list of unique first 10 words from the clean_text DataFrame
clean_text_choices = df_clean_text['first_10_words'].unique()

# Apply fuzzy matching to find the best match for each case presentation
df_case_presentation['matched_first_10_words'] = df_case_presentation['first_10_words'].apply(lambda x: fuzzy_match(x, clean_text_choices))

# Merge the DataFrames on the matched first 10 words
merged_df = pd.merge(df_case_presentation, df_clean_text[['clean text', 'first_10_words']], left_on='matched_first_10_words', right_on='first_10_words', how='left')

# Drop the auxiliary columns used for merging
merged_df.drop(columns=['first_10_words_x', 'first_10_words_y', 'matched_first_10_words'], inplace=True)

# Save the merged DataFrame to a new CSV
merged_df.to_csv('merged_case_presentation_clean_text.csv', index=False)
