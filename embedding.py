# Outline:
# Import embedding model using transformers
# Embed the entire icd-10-cm dataset
# Embed all final diagnosis in the processed case reports csv
# For each final diagnosis, find top 5 closest diagnoses long description
# Anwsers are (hard) #1,#2,#3, (medium) #2, #3, #4, or (easy) #3, #4, #5

from transformers import AutoModel
from numpy.linalg import norm
import pandas as pd
from tqdm import tqdm

cos_sim = lambda a,b: (a @ b.T) / (norm(a)*norm(b))
model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True) # trust_remote_code is needed to use the encode method
embeddings = model.encode(['How is the weather today?', 'What is the current weather like today?'])
print(cos_sim(embeddings[0], embeddings[1]))

# Embedding final diagnoses
case_reports_df = pd.read_csv('Processed_Case_Reports.csv', encoding='utf-8')
embedding_list = []
for diag in tqdm(case_reports_df['final diagnosis'].tolist()):
    embedding_list.append(model.encode(diag))

case_reports_df['embedding'] = embedding_list
print("added embedding to case reports")

# Embedding icd-10-cm
icd_df = pd.read_csv('icd_diagnosis.csv', encoding='utf-8')
embedding_list = []
for diag in tqdm(icd_df['LongDescription'].tolist()):
    embedding_list.append(model.encode(diag))
icd_df['embedding'] = embedding_list
print("added embedding to icd-10-cm")
case_reports_df.to_csv('case_reports_embedded.csv', encoding="utf-8")
icd_df.to_csv('icd_10_cm_embedded.csv', encoding="utf-8")
results = []

# Loop through each row in the case reports dataframe
for idx, case_row in case_reports_df.iterrows():
    similarities = []

    # Calculate cosine similarity with each row in the icd dataframe
    for icd_idx, icd_row in tqdm(icd_df.iterrows()):
        similarity = cos_sim(case_row['embedding'], icd_row['embedding'])
        similarities.append((icd_idx, similarity))
        print(similarities)

    # Sort the similarities and get the top 5
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)[:5]

    # Get the top 5 similar ICD texts
    top_5_texts = [icd_df.loc[sim[0], 'text'] for sim in similarities]

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
