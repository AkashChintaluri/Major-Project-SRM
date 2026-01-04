import pandas as pd

# Load the file we just created
df = pd.read_parquet("data/processed/pubmed_extracted.parquet")

print("--- Data Preview ---")
print(df.head())
print("\n--- Statistics ---")
print(f"Total Articles: {len(df)}")