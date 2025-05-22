import pandas as pd
import os

# Define file paths
# Change these paths to where your downloaded files are located
fake_path = "D:/DOWNLOADS/New folder/Fake.csv"
true_path = "D:/DOWNLOADS/New folder/Fake.csv"
output_dir = "D:/FakeDetector/data/news_data/"
output_file = os.path.join(output_dir, "news.csv")

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load datasets
print("Loading datasets...")
fake_df = pd.read_csv(fake_path)
true_df = pd.read_csv(true_path)

print(f"Fake news dataset: {fake_df.shape[0]} articles")
print(f"Real news dataset: {true_df.shape[0]} articles")

# Add labels
print("Adding labels...")
fake_df['label'] = 1  # 1 for fake
true_df['label'] = 0  # 0 for real

# Combine datasets
print("Combining datasets...")
combined_df = pd.concat([fake_df, true_df], axis=0)

# Shuffle the data
combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save to the project directory
print(f"Saving combined dataset to {output_file}...")
combined_df.to_csv(output_file, index=False)

print(f"Done! Combined dataset has {combined_df.shape[0]} articles")
print(f"Real news: {sum(combined_df['label'] == 0)}")
print(f"Fake news: {sum(combined_df['label'] == 1)}")
