"""
Spam Email Detection - Single Message Prediction CLI
---------------------------------------------------
This script allows users to quickly test any email or SMS text message
using the pre-trained Naive Bayes model and TF-IDF vectorizer.
"""

import os
import sys
import joblib
from train import clean_text


def load_artifacts():
    model_path = os.path.join("models", "spam_model.pkl")
    vec_path = os.path.join("models", "tfidf_vectorizer.pkl")

    if not os.path.exists(model_path) or not os.path.exists(vec_path):
        print("[!] Error: Trained model files not found.")
        print("--> Please run 'python src/train.py' first to train and save the model.")
        sys.exit(1)

    model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)
    return model, vectorizer


def predict_message(message: str):
    model, vectorizer = load_artifacts()
    cleaned = clean_text(message)
    features = vectorizer.transform([cleaned])
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    ham_prob = probabilities[0] * 100
    spam_prob = probabilities[1] * 100

    print("\n" + "=" * 55)
    print("           EMAIL / SMS SPAM PREDICTION")
    print("=" * 55)
    print(f"Message: \"{message}\"")
    print("-" * 55)

    if prediction == 1:
        print(f"Prediction: [SPAM]")
        print(f"Spam Probability:     {spam_prob:.2f}%")
        print(f"Ham (Legit) Probability: {ham_prob:.2f}%")
    else:
        print(f"Prediction: [NOT SPAM (HAM)]")
        print(f"Ham (Legit) Probability: {ham_prob:.2f}%")
        print(f"Spam Probability:     {spam_prob:.2f}%")
    print("=" * 55 + "\n")


def main():
    if len(sys.argv) > 1:
        input_text = " ".join(sys.argv[1:])
        predict_message(input_text)
    else:
        print("=" * 55)
        print("  Interactive Spam Detection CLI")
        print(" (Type 'exit' or 'quit' to stop)")
        print("=" * 55)
        while True:
            try:
                user_msg = input("\nEnter email or SMS message: ").strip()
                if user_msg.lower() in ["exit", "quit"]:
                    print("Goodbye!")
                    break
                if not user_msg:
                    print("Please enter a valid message.")
                    continue
                predict_message(user_msg)
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye!")
                break


if __name__ == "__main__":
    main()
