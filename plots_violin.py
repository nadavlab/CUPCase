import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import bert_score
import seaborn as sns

# Load the CSV file
df = pd.read_csv('Case_report_w_images_dis_VF.csv')

def calculate_f1_scores(references, candidates, model_type='microsoft/deberta-xlarge-mnli', batch_size=500):
    P, R, F1 = bert_score.score(candidates, references, lang="en", model_type=model_type, batch_size=batch_size)
    return F1.numpy()

# Extract columns
final_diagnosis = df['final diagnosis'].tolist()
distractors = [df['distractor2'].tolist(), df['distractor3'].tolist(), df['distractor4'].tolist()]

# Calculate F1 scores
f1_scores = [calculate_f1_scores(final_diagnosis, distractor, batch_size=500) for distractor in distractors]

# Create a DataFrame to store the scores
scores_df = pd.DataFrame({
    'Distractor 2': f1_scores[0],
    'Distractor 3': f1_scores[1],
    'Distractor 4': f1_scores[2]
})

# Save scores to CSV
scores_df.to_csv('bert_scores_per_distractor.csv', index=False)

# Create a figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the violin plot
sns.violinplot(data=scores_df, ax=ax)

# Add labels and title
ax.set_xlabel('Distractor')
ax.set_ylabel('F1 Score')
ax.set_title('BERTScore F1 Score Distributions for Distractors')

# Save the plot as PNG and JPEG
plot_filename = 'f1_scores_violin_plot'
plt.savefig(f'{plot_filename}.png', format='png')
plt.savefig(f'{plot_filename}.jpeg', format='jpeg')

# Show the plot
plt.show()