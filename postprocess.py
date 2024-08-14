import pandas as pd
import ast

df = pd.read_csv('Case_reports_w_images_VF.csv')

df['top_5_icd_texts'] = df['top_5_icd_texts'].apply(ast.literal_eval)

df[['distractor1', 'distractor2', 'distractor3', 'distractor4', 'distractor5']] = df['top_5_icd_texts'].apply(pd.Series)

df.to_csv('Case_report_w_images_dis_VF.csv')