from pathlib import Path
import pandas as pd
from sklearn.datasets import load_breast_cancer
def main():
    dataset = load_breast_cancer()
    feature_names = [name.replace(" ", "_")
        for name in dataset.feature_names]
    df = pd.DataFrame(dataset.data, columns=feature_names)
    df["target"] = dataset.target
    # Validation 1: Exactly 30 feature columns
    feature_count = df.drop(columns=["target"]).shape[1]
    if feature_count != 30:
        raise ValueError(f"Expected 30 features, found {feature_count}")

    # Validation 2: No missing values
    if df.isnull().any().any():
        raise ValueError("Dataset contains null values")
    output_path = Path("data/raw/breast_cancer.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dataset saved to: {output_path}")
    print(f"Dataset shape: {df.shape}")
    print(f"Feature count validated: {feature_count}")
    print("Null values validated: 0")
    print("Target mapping: 0 = malignant, 1 = benign")
if __name__ == "__main__":
    main()

