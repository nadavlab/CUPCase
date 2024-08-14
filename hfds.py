import pandas as pd
from datasets import Dataset

# Define the path to your CSV file
csv_file_path = 'ablation_study_tokens.csv'

# Load the CSV file into a DataFrame
df = pd.read_csv(csv_file_path)

# Define the percentages and corresponding file names
percentages = ["20%", "40%", "60%", "80%"]
file_names = ["dataset_20", "dataset_40", "dataset_60", "dataset_80"]

# Create and save datasets for each percentage
for percent, file_name in zip(percentages, file_names):
    # Create a new DataFrame with the specific percentage column
    new_df = pd.DataFrame({
        'clean text': df[percent],
        'final diagnosis': df['final diagnosis']
    })

    # Convert the DataFrame to a Hugging Face dataset
    dataset = Dataset.from_pandas(new_df)

    # Save the dataset to disk
    dataset.save_to_disk(f'/ablation_dataset/{file_name}')

    print(f"Saved dataset with {percent} to /ablation_dataset/{file_name}")
