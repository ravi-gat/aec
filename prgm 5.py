import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes

data = load_diabetes()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target

df["bmi_cat"] = pd.cut(df["bmi"],
                       bins=[-1, 0, 0.05, 1],
                       labels=["Low", "Normal", "High"])

df["bp_scaled"] = (df["bp"] - df["bp"].min()) / (df["bp"].max() - df["bp"].min())

df_selected = df[["bmi", "bp", "s1", "target"]]

print(df.head())
print("\nSelected Features:\n", df_selected.head())

plt.figure(figsize=(6, 4))
plt.scatter(df["bmi"], df["target"], color="blue", alpha=0.5)
plt.title("BMI vs Diabetes Progression")
plt.xlabel("BMI (Scaled)")
plt.ylabel("Target Score")
plt.grid(True)
plt.show()
