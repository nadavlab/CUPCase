import pandas as pd
from transformers import AutoTokenizer

# Read the CSV file
df = pd.read_csv('Case_report_w_images_dis_VF.csv')

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

def truncate_text_by_tokens(text, percentage, tokenizer):
    tokens = tokenizer.tokenize(text)
    num_tokens = int(len(tokens) * percentage)
    truncated_tokens = tokens[:num_tokens]
    return tokenizer.convert_tokens_to_string(truncated_tokens)

# Create new dataframe with truncated text for all rows
result_df = df.copy()
result_df['20%'] = df['clean text'].apply(lambda x: truncate_text_by_tokens(x, 0.2, tokenizer))
result_df['40%'] = df['clean text'].apply(lambda x: truncate_text_by_tokens(x, 0.4, tokenizer))
result_df['60%'] = df['clean text'].apply(lambda x: truncate_text_by_tokens(x, 0.6, tokenizer))
result_df['80%'] = df['clean text'].apply(lambda x: truncate_text_by_tokens(x, 0.8, tokenizer))
result_df['100%'] = df['clean text']

# Reorder columns to have percentage columns at the end
cols = [col for col in result_df.columns if col not in ['20%', '40%', '60%', '80%', '100%']] + ['20%', '40%', '60%', '80%', '100%']
result_df = result_df[cols]

# Save the resulting dataframe to a CSV file
result_df.to_csv('ablation_study_tokens.csv', index=False)

# Print the first few rows of the resulting dataframe
print(result_df.head())
