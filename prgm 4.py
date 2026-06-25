import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes

data = load_diabetes()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target

print("First 5 Rows:\n", df.head())
print("\nDataset Info:\n")
df.info()

print("\nStatistical Summary:\n", df.describe())

print("\nMissing Values:\n", df.isnull().sum())

plt.figure(figsize=(8, 5))
sns.histplot(df['age'], kde=True, color='skyblue')
plt.title("Standardized Age Distribution")
plt.xlabel("Age (Scaled)")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(df['target'], kde=True, color='green')
plt.title("Disease Progression Distribution")
plt.xlabel("Progression Score")
plt.ylabel("Frequency")
plt.show()
