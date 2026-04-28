import pandas as pd
import numpy as np
import random

np.random.seed(42)
random.seed(42)

n_samples = 200

# Demographics
genders = np.random.choice(['Male', 'Female', 'Non-Binary'], n_samples, p=[0.45, 0.45, 0.1])
accents = np.random.choice(['Native', 'Non-Native'], n_samples, p=[0.7, 0.3])
years_experience = np.random.randint(1, 15, n_samples)

# True skill score (0-100)
# Let's say skill is independent of gender and accent
true_skill = np.random.normal(75, 10, n_samples).clip(0, 100)

# The biased AI model score
# Intentionally inject bias: Non-native accents get a severe -18 penalty
# Females get a -8 penalty
ai_score = true_skill.copy()
ai_score[accents == 'Non-Native'] -= 18
ai_score[genders == 'Female'] -= 8

# AI decision (Hire if score > 70)
ai_decision = (ai_score > 70).astype(int)

# True decision (Hire if true_skill > 70)
true_decision = (true_skill > 70).astype(int)

df = pd.DataFrame({
    'Candidate_ID': range(1, n_samples + 1),
    'Gender': genders,
    'Accent': accents,
    'Years_Experience': years_experience,
    'True_Skill_Score': np.round(true_skill, 1),
    'AI_Interview_Score': np.round(ai_score, 1),
    'AI_Hire_Decision': ai_decision,
    'True_Hire_Decision': true_decision
})

# Generate mock transcripts and Perspective API Toxicity Scores
def generate_transcript(accent, score):
    if accent == 'Non-Native':
        return "The candidate discussed their technical background but the automated transcription confidence was low. Some words were flagged as unclear. Score impacted."
    else:
        return "The candidate clearly articulated their experience and problem-solving skills. Transcription confidence high."

df['Transcript_Notes'] = df.apply(lambda row: generate_transcript(row['Accent'], row['AI_Interview_Score']), axis=1)

# Simulate Google Perspective API Toxicity Score (0 to 1)
# The biased AI model uses more negative language (microaggressions) when evaluating Non-Native speakers
df['Perspective_Toxicity_Score'] = np.where(df['Accent'] == 'Non-Native', np.random.normal(0.45, 0.1, n_samples), np.random.normal(0.15, 0.05, n_samples)).clip(0, 1)

df.to_csv('hr_hiring_data.csv', index=False)
print("hr_hiring_data.csv successfully generated with biased sample data and Perspective API scores.")
