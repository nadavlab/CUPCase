import pandas as pd
import numpy as np
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

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
