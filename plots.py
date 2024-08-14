import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import bert_score
import seaborn as sns

# Load the CSV file
df = pd.read_csv('Case_report_w_images_dis_VF.csv')

def calculate_f1_scores(references, candidates, model_type='microsoft/deberta-xlarge-mnli'):
    P, R, F1 = bert_score.score(candidates, references, lang="en", model_type=model_type)
    return F1.numpy()

# Extract columns
final_diagnosis = df['final diagnosis'].tolist()
distractors = [df['distractor2'].tolist(), df['distractor3'].tolist(), df['distractor4'].tolist()]

# Calculate F1 scores
f1_scores = [calculate_f1_scores(final_diagnosis, distractor) for distractor in distractors]

# Create a figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the distributions
colors = ['blue', 'green', 'red']
labels = ['Distractor 2', 'Distractor 3', 'Distractor 4']
for i, scores in enumerate(f1_scores):
    sns.kdeplot(scores, shade=True, color=colors[i], label=labels[i], ax=ax)

# Add labels and legend
ax.set_xlabel('F1 Score')
ax.set_ylabel('Density')
ax.set_title('BERTScore F1 Score Distributions for Distractors')
ax.legend()

# Save the plot as PNG and JPEG
plot_filename = 'f1_scores_dist_plot'
plt.savefig(f'{plot_filename}.png', format='png')
plt.savefig(f'{plot_filename}.jpeg', format='jpeg')

# Show the plot
plt.show()