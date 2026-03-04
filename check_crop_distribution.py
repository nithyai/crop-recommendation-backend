import pandas as pd
import matplotlib.pyplot as plt

# Load your CSV dataset
df = pd.read_csv("Crop_pre.csv")

# Show first 5 rows
print("First 5 rows of the dataset:")
print(df.head())

# Merge duplicate crop names (case-insensitive)
df['label'] = df['label'].str.lower()

# Count samples per crop after merging
crop_counts = df['label'].value_counts()
print("\nCorrected Crop distribution (after merging duplicates):")
print(crop_counts)

# Plot updated distribution
crop_counts.plot(kind='bar', title='Corrected Crop Distribution', ylabel='Number of Samples')
plt.show()
