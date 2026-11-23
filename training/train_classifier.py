
from pathlib import Path
import sys

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

# Make the main project directory importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.classification_features import normalise_text


TRAINING_DIR = Path(__file__).resolve().parent

DATA_PATH = TRAINING_DIR / "classification_training_data.csv"

MODEL_DIR = TRAINING_DIR / "saved_models"

MODEL_PATH = MODEL_DIR / "dataset_classifier.joblib"


def train_classifier():

    print("\n======================================")
    print("DATASENTINEL DATA CLASSIFICATION MODEL")
    print("======================================")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Training data not found: {DATA_PATH}"
        )

    # Load labelled training examples
    df = pd.read_csv(
        DATA_PATH,
        keep_default_na=False
    )

    required_columns = [
        "sample_text",
        "domain",
        "subcategory",
        "dataset_type"
    ]

    missing = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing training columns: {missing}"
        )

    # Clean training data
    for column in required_columns:
        df[column] = df[column].astype(str).str.strip()

    df = df[
        (df["sample_text"] != "")
        & (df["domain"] != "")
        & (df["subcategory"] != "")
        & (df["dataset_type"] != "")
    ].copy()

    if len(df) < 2:
        raise ValueError(
            "At least two valid training examples are required."
        )

    # Convert sample descriptions into standardised text
    training_text = df["sample_text"].apply(
        normalise_text
    )

    # Create TF-IDF representations
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
        strip_accents="unicode"
    )

    training_matrix = vectorizer.fit_transform(
        training_text
    )

    # Store all labels with their corresponding examples
    training_records = df[
        required_columns
    ].to_dict(orient="records")

    # Save trained model components
    model_bundle = {
        "vectorizer": vectorizer,
        "training_matrix": training_matrix,
        "training_records": training_records,
        "model_version": "1.0",
        "training_examples": len(df)
    }

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        model_bundle,
        MODEL_PATH
    )

    print("\nTraining completed successfully!")

    print("Training examples:", len(df))
    print("Domains:", df["domain"].nunique())
    print("Subcategories:", df["subcategory"].nunique())
    print("Dataset types:", df["dataset_type"].nunique())

    print("\nRecognised domains:")

    for domain in sorted(df["domain"].unique()):
        print("-", domain)

    print("\nModel saved at:")
    print(MODEL_PATH)

    print("\n======================================")
    print("MODEL TRAINING COMPLETED")
    print("======================================")


if __name__ == "__main__":
    train_classifier()
