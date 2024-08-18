import pandas as pd
import numpy as np
from transformers import AutoTokenizer

# Load the LLaMA-3-8B tokenizer from the Hugging Face cache folder
cache_folder = "/sise/nadav-group/nadavrap-group/ofir/hf_cache/models--meta-llama--Meta-Llama-3-8B/snapshots/1460c22666392e470910ce3d44ffeb2ab7dbd4df"
tokenizer = AutoTokenizer.from_pretrained(cache_folder)

# Read the CSV file
df = pd.read_csv('Case_report_w_images_dis_VF.csv')

# Tokenize and calculate token counts using LLaMA-3-8B tokenizer
df['token_count'] = df['final diagnosis'].apply(lambda x: len(tokenizer.tokenize(x)))

# Calculate statistics
token_counts = df['token_count']
stats = {
    "Statistic": ["Minimum tokens", "Maximum tokens", "Average tokens", "Median tokens", "75th percentile", "95th percentile"],
    "Value": [
        token_counts.min(),
        token_counts.max(),
        token_counts.mean(),
        token_counts.median(),
        np.percentile(token_counts, 75),
        np.percentile(token_counts, 95)
    ]
}
stats_df = pd.DataFrame(stats)

# Format the 'Value' column
stats_df['Value'] = stats_df['Value'].apply(lambda x: f"{x:.2f}" if isinstance(x, float) else x)
print(stats_df)
