import pandas as pd
import re


def remove_figures(text):
    # Remove patterns like (Fig. X), (Fig. Xa), (Fig. X a), (Fig. 1a, b), (Fig. 1 d), (Fig. 1 c)
    pattern = r'\(Fig\. \d+[a-z]?(?:, \d+[a-z]?)*\s?[a-z]?\)'
    return re.sub(pattern, '', text)


def clean_csv(input_file, output_file):
    try:
        # Read the CSV file
        df = pd.read_csv(input_file)

        # Ensure the column "clean text" exists
        if 'clean text' in df.columns:
            # Apply the regex removal to the "clean text" column
            df['clean text'] = df['clean text'].astype(str).apply(remove_figures)

            # Save the cleaned data to a new CSV file
            df.to_csv(output_file, index=False)
            print(f"Cleaned data has been saved to {output_file}")
        else:
            print("Column 'clean text' not found in the input file.")
    except Exception as e:
        print(f"An error occurred: {e}")


input_file = 'case_reports_with_top_5_similar_icd_texts.csv'
output_file = 'Case_reports_w_images_VF.csv'
clean_csv(input_file, output_file)
