import pandas as pd
import os
from PIL import Image as PILImage
from datasets import Dataset

def convert_to_linux_path(path):
    return '/sise/nadav-group/nadavrap-group/oriel/case_report_ds/' + path.replace('\\', '/')

def load_image(image_path):
    if pd.notna(image_path) and image_path.strip():
        image_path = convert_to_linux_path(image_path)
        if os.path.exists(image_path):
            try:
                with PILImage.open(image_path) as img:
                    return image_path  # Return the path instead of the image object
            except Exception as e:
                print(f"Error opening image {image_path}: {e}")
                return None
        else:
            # print(f"File not found: {image_path}")
            return None
    print("Invalid or empty image path provided.")
    return None

def extract_image_names(image_paths):
    return [os.path.basename(path).split('_')[-1].split('.')[0] for path in image_paths if pd.notna(path)]

df = pd.read_csv('case_reports_with_top_5_similar_icd_texts.csv')

df['Image Filenames'] = df['Image Filenames'].astype(str).str.split(',')

df['Image Filenames'] = df['Image Filenames'].apply(
    lambda paths: [path.strip() for path in paths if pd.notna(path) and path.strip()]
)

df = df[df['Image Filenames'].map(lambda paths: len(paths) > 0)]

df['Image Paths'] = df['Image Filenames'].apply(lambda paths: [load_image(path) for path in paths])
df['Image Names'] = df['Image Filenames'].apply(lambda paths: extract_image_names(paths))

# Filter out None values
df['Image Paths'] = df['Image Paths'].apply(lambda paths: [path for path in paths if path is not None])
df = df[df['Image Paths'].map(len) > 0]

df = df.drop(columns=['Image Filenames'])
df = df.rename(columns={'Image Paths': 'Image File Paths'})

# Convert DataFrame to Dataset
dataset = Dataset.from_pandas(df)

# Save the dataset
dataset.save_to_disk('/saved_dataset')

print("Dataset saved successfully!")
