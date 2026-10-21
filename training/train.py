
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
import sklearn
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score,make_scorer,recall_score,roc_auc_score,)
from sklearn.model_selection import (StratifiedKFold,cross_validate,train_test_split,)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
def main():
    # 1. Command-line arguments
    parser = argparse.ArgumentParser(
        description="Train breast cancer classification models"
    )
    parser.add_argument(
        "--data-path",
        default="data/raw/breast_cancer.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts",
    )
    parser.add_argument(
        "--model-version",
        default="1.0.0",
    )
    args = parser.parse_args()
    # 2. Load training parameters
    with open("params.yaml", encoding="utf-8") as file:
        params = yaml.safe_load(file)
    seed = params["seed"]
    test_size = params["split"]["test_size"]
    folds = params["cross_validation"]["folds"]
    # 3. Load and validate DVC-tracked dataset
    df = pd.read_csv(args.data_path)
    if df.isnull().any().any():
        raise ValueError("Dataset contains null values")
    if "target" not in df.columns:
        raise ValueError("Target column is missing")
    X = df.drop(columns=["target"])
    y = df["target"]
    if X.shape[1] != 30:
        raise ValueError(
            f"Expected 30 features, found {X.shape[1]}"
        )
    if not set(y.unique()).issubset({0, 1}):
        raise ValueError("Unexpected target labels")
    feature_names = X.columns.tolist()
    print(f"Dataset shape: {df.shape}")
    print("Target mapping: 0=malignant, 1=benign")
    # 4. Stratified 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size,random_state=seed,stratify=y,)
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    # 5. Read model hyperparameters
    lr_params = params["models"]["logistic_regression"]
    rf_params = params["models"]["random_forest"]
    # 6. Define two distinct model families
    models = {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                C=lr_params["C"],
                max_iter=lr_params["max_iter"],
                class_weight=lr_params["class_weight"],
                random_state=seed,
            )),
        ]),
        "random_forest": Pipeline([
            ("classifier", RandomForestClassifier(
                n_estimators=rf_params["n_estimators"],
                max_depth=rf_params["max_depth"],
                min_samples_split=rf_params["min_samples_split"],
                class_weight=rf_params["class_weight"],
                random_state=seed,
                n_jobs=1,
            )),
        ]),
    }
    # 7. Five-fold stratified cross-validation
    cv = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=seed,
    )
    # Malignant is class 0.
    scoring = {
        "malignant_recall": make_scorer(
            recall_score,
            pos_label=0,
        ),
        "accuracy": "accuracy",
        "roc_auc": "roc_auc",
    }
    results = {}
    best_name = None
    best_recall = -1.0
    # 8. Evaluate every candidate using CV
    for name, model in models.items():
        scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
        )
        cv_metrics = {
            "malignant_recall": float(
                scores["test_malignant_recall"].mean()
            ),
            "accuracy": float(
                scores["test_accuracy"].mean()
            ),
            "roc_auc": float(
                scores["test_roc_auc"].mean()
            ),
        }
        results[name] = {
            "cross_validation": cv_metrics
        }
        print(f"\n{name} CV metrics: {cv_metrics}")
        # Selection uses CV malignant recall only.
        if cv_metrics["malignant_recall"] > best_recall:
            best_recall = cv_metrics["malignant_recall"]
            best_name = name
    # 9. Fit and evaluate every candidate on test data
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        malignant_index = list(model.classes_).index(0)
        malignant_probabilities = model.predict_proba(
            X_test
        )[:, malignant_index]

        test_metrics = {
            "malignant_recall": float(
                recall_score(
                    y_test,
                    predictions,
                    pos_label=0,
                )
            ),
            "accuracy": float(
                accuracy_score(y_test, predictions)
            ),
            "roc_auc": float(
                roc_auc_score(
                    (y_test == 0).astype(int),
                    malignant_probabilities,
                )
            ),
        }
        results[name]["test"] = test_metrics
        print(f"{name} test metrics: {test_metrics}")
    # 10. Select the model based on CV recall
    selected_model = models[best_name]
    selected_test_metrics = results[best_name]["test"]
    print(f"\nSelected model: {best_name}")
    print(f"Selected test metrics: {selected_test_metrics}")
    # 11. Enforce minimum malignant recall
    if selected_test_metrics["malignant_recall"] < 0.95:
        raise SystemExit("Training failed: test malignant recall below 0.95")
    # 12. Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # 13. Create model card
    model_card = {
        "model_version": args.model_version,
        "model_family": best_name,
        "feature_names": feature_names,
        "feature_count": len(feature_names),
        "metrics": {
            "candidates": results,
            "test": selected_test_metrics,
        },
        "seed": seed,
        "sklearn_version": sklearn.__version__,
    }
    # 14. Save trained pipeline
    joblib.dump(
        selected_model,
        output_dir / "model.joblib",
    )
    # 15. Save model card
    (output_dir / "model_card.json").write_text(
        json.dumps(model_card, indent=2),
        encoding="utf-8",
    )
    # 16. Save all evaluation metrics
    (output_dir / "metrics.json").write_text(
        json.dumps(
            {
                "selected_model": best_name,
                "candidates": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\nTraining completed successfully.")
    print(f"Artifacts saved to: {output_dir}")
if __name__ == "__main__":
    main()

