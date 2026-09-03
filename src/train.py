"""
==============================================================================
SPAM EMAIL & SMS CLASSIFICATION SYSTEM
==============================================================================
Author: Machine Learning Project Portfolio
Description: Step-by-step pipeline to clean, explore, train, evaluate, and
             save an NLP Spam Detection model using TF-IDF & Naive Bayes.
==============================================================================
"""

import os
import re
import string
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def clean_text(text: str) -> str:
    """
    Cleans raw text by:
    1. Lowercasing
    2. Removing web links & HTML tags
    3. Removing punctuation
    4. Stripping extra whitespace
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def run_pipeline():
    print("=" * 70)
    print("    STEP 1: INITIALIZE ENVIRONMENT & DIRECTORIES")
    print("=" * 70)
    os.makedirs("models", exist_ok=True)
    os.makedirs("assets/screenshots", exist_ok=True)
    print("[OK] Directories verified.")

    print("\n" + "=" * 70)
    print("    STEP 2: LOAD & EXPLORE THE DATASET")
    print("=" * 70)
    data_path = os.path.join("data", "spam.csv")
    df = pd.read_csv(data_path)
    print(f"Dataset Loaded Successfully! Shape: {df.shape}")
    print("\n--- First 5 Rows (df.head()) ---")
    print(df.head())

    print("\n--- Dataset Summary Info ---")
    print(f"Total Messages: {len(df)}")
    print(f"Legitimate (Ham) Count: {(df['label'] == 'ham').sum()}")
    print(f"Spam Count:             {(df['label'] == 'spam').sum()}")

    # Visualizing class distribution
    plt.figure(figsize=(7, 4.5))
    ax = sns.countplot(data=df, x="label", palette=["#4CAF50", "#F44336"], hue="label", legend=False)
    plt.title("Email/SMS Class Distribution (Ham vs Spam)", fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("Message Category", fontsize=11)
    plt.ylabel("Count", fontsize=11)
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2.0, p.get_height()),
                    ha="center", va="baseline", fontsize=11, fontweight="bold", xytext=(0, 4), textcoords="offset points")
    plt.tight_layout()
    plt.savefig("assets/screenshots/class_distribution.png", dpi=150)
    plt.close()
    print("[OK] Saved class distribution graph to 'assets/screenshots/class_distribution.png'")

    print("\n" + "=" * 70)
    print("    STEP 3: TEXT PREPROCESSING & CLEANING")
    print("=" * 70)
    df["cleaned_message"] = df["message"].apply(clean_text)
    print("Sample Before & After Cleaning:")
    sample_idx = 2
    print(f"RAW:     {df['message'].iloc[sample_idx]}")
    print(f"CLEANED: {df['cleaned_message'].iloc[sample_idx]}")

    X = df["cleaned_message"]
    y = df["label"].map({"ham": 0, "spam": 1})

    print("\n" + "=" * 70)
    print("    STEP 4: TRAIN-TEST DATASET SPLIT")
    print("=" * 70)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Training set: {len(X_train)} samples (80%)")
    print(f"Testing set:  {len(X_test)} samples (20%)")

    print("\n" + "=" * 70)
    print("    STEP 5: TF-IDF VECTORIZATION & MODEL TRAINING")
    print("=" * 70)
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)} n-gram features")

    model = MultinomialNB(alpha=0.1)
    model.fit(X_train_tfidf, y_train)
    print("[OK] Multinomial Naive Bayes model successfully trained!")

    print("\n" + "=" * 70)
    print("    STEP 6: MODEL EVALUATION ON UNSEEN TEST DATA")
    print("=" * 70)
    y_pred = model.predict(X_test_tfidf)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Overall Accuracy:  {acc * 100:.2f}%")
    print(f"Precision (Spam):  {prec * 100:.2f}%")
    print(f"Recall (Spam):     {rec * 100:.2f}%")
    print(f"F1-Score:          {f1 * 100:.2f}%")

    print("\nConfusion Matrix:")
    print(f"  True Ham (Correctly allowed): {cm[0][0]} | False Spam (False alarm): {cm[0][1]}")
    print(f"  False Ham (Missed spam):       {cm[1][0]} | True Spam (Correctly caught): {cm[1][1]}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Ham (Legitimate)", "Spam"]))

    # Save Confusion Matrix Heatmap
    plt.figure(figsize=(6.5, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Predicted Ham", "Predicted Spam"],
                yticklabels=["Actual Ham", "Actual Spam"],
                annot_kws={"size": 14, "weight": "bold"})
    plt.title("Confusion Matrix Heatmap", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig("assets/screenshots/confusion_matrix_heatmap.png", dpi=150)
    plt.close()
    print("[OK] Saved confusion matrix heatmap to 'assets/screenshots/confusion_matrix_heatmap.png'")

    print("\n" + "=" * 70)
    print("    STEP 7: SAVE TRAINED MODEL & VECTORIZER")
    print("=" * 70)
    joblib.dump(model, os.path.join("models", "spam_model.pkl"))
    joblib.dump(vectorizer, os.path.join("models", "tfidf_vectorizer.pkl"))
    print("[OK] Artifacts saved to 'models/' directory.")

    print("\n" + "=" * 70)
    print("    STEP 8: PREDICTIONS ON BRAND NEW UNSEEN MESSAGES")
    print("=" * 70)
    test_messages = [
        "Hey! Are you free for lunch tomorrow at 1 PM?",
        "URGENT! You have won a 1000 cash prize! Claim your reward now by calling 09061701461!",
        "Can you please send me the class notes from yesterday's lecture?",
        "Congratulations! Click here to claim your free gift card immediately."
    ]

    for msg in test_messages:
        cleaned_m = clean_text(msg)
        vec_m = vectorizer.transform([cleaned_m])
        pred = model.predict(vec_m)[0]
        prob = model.predict_proba(vec_m)[0]
        label = "SPAM [RED]" if pred == 1 else "NOT SPAM / HAM [GREEN]"
        conf = prob[pred] * 100
        print(f"\nMessage: \"{msg}\"")
        print(f"Result:  {label} (Confidence: {conf:.2f}%)")

    print("\n" + "=" * 70)
    print("    [PIPELINE EXECUTION COMPLETE]")
    print("=" * 70)


if __name__ == "__main__":
    run_pipeline()
