import pandas as pd
import glob

# Define the range for X and Y
x_values = range(16, 101)
y_values = range(20, 101)

# Create a list to hold the dataframes
dfs = []

# Loop through all combinations of X and Y
for x in x_values:
    for y in y_values:
        # Create the file name pattern
        file_pattern = f"case_presentations_w_images_{x}_{y}.xlsx"
        # Use glob to find the files matching the pattern
        for file in glob.glob(file_pattern):
            # Read the CSV file and append the dataframe to the list
            df = pd.read_excel(file)
            dfs.append(df)

# Concatenate all dataframes into a single dataframe
combined_df = pd.concat(dfs, ignore_index=True)

# Save the combined dataframe to a new CSV file
combined_df.to_excel("combined_case_presentations.xlsx", index=False)

print("CSV files combined successfully!")
