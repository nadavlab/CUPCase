from transformers import AutoModel
from numpy.linalg import norm
import pandas as pd
from tqdm import tqdm

# Function to calculate cosine similarity
cos_sim = lambda a, b: (a @ b.T) / (norm(a) * norm(b))

# Load the model
model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True)  # trust_remote_code is needed to use the encode method

# Embedding function with batching
def batch_encode(texts, model, batch_size=100000):
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]
        embeddings.extend(model.encode(batch))
    return embeddings

# Embedding final diagnoses
case_reports_df = pd.read_excel('Second_Processed_Case_Reports_w_images_clean.csv')
final_diagnoses = case_reports_df['final diagnosis'].tolist()
case_reports_df['embedding'] = batch_encode(final_diagnoses, model)

print("Added embedding to case reports")

# Embedding ICD-10-CM descriptions
icd_df = pd.read_csv('icd_10_diagnosis/icd_diagnosis.csv', encoding='utf-8')
icd_descriptions = icd_df['LongDescription'].tolist()
icd_df['embedding'] = batch_encode(icd_descriptions, model)

print("Added embedding to icd-10-cm")

# Save the embeddings
case_reports_df.to_excel('case_reports_embedded_w_images.xlsx', index=False)
icd_df.to_csv('icd_10_cm_embedded.csv', encoding="utf-8")

# Finding top 5 closest diagnoses
results = []

for idx, case_row in case_reports_df.iterrows():
    similarities = []

    for icd_idx, icd_row in icd_df.iterrows():
        similarity = cos_sim(case_row['embedding'], icd_row['embedding'])
        similarities.append((icd_idx, similarity))

    # Sort the similarities and get the top 5
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)[:5]

    # Get the top 5 similar ICD texts
    top_5_texts = [icd_df.loc[sim[0], 'LongDescription'] for sim in similarities]

    # Append the result to the results list
    results.append({
        'case_report_idx': idx,
        'top_5_similar_icd_idxs': [sim[0] for sim in similarities],
        'top_5_similarities': [sim[1] for sim in similarities],
        'top_5_icd_texts': top_5_texts
    })

# Convert the results list to a dataframe
results_df = pd.DataFrame(results)

# Merge the top 5 ICD texts back into the case reports dataframe
case_reports_df = case_reports_df.merge(results_df[['case_report_idx', 'top_5_icd_texts']], left_index=True,
                                        right_on='case_report_idx', how='left')

# Save the updated case reports dataframe to a CSV file
case_reports_df.to_csv('case_reports_with_top_5_similar_icd_texts.csv', index=False)
