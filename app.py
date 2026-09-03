import os
import sys
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Add src to system path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from train import clean_text

# Page Configuration
st.set_page_config(
    page_title="SpamGuard AI — Email & SMS Classifier",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-End Styling (CSS)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .header-container {
        background: linear-gradient(135deg, #090D16 0%, #111827 50%, #1E293B 100%);
        border-radius: 20px;
        padding: 2.2rem 2.5rem;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.5);
        color: white;
    }
    
    .header-title {
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin: 0;
        background: linear-gradient(90deg, #38BDF8 0%, #818CF8 50%, #C084FC 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .header-subtitle {
        color: #94A3B8;
        font-size: 1.05rem;
        margin-top: 0.6rem;
        font-weight: 400;
        line-height: 1.5;
    }
    
    .stat-badge {
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.12);
        padding: 8px 14px;
        border-radius: 10px;
        font-size: 0.88rem;
        color: #E2E8F0;
        display: inline-block;
        margin-top: 12px;
        margin-right: 8px;
        backdrop-filter: blur(8px);
    }

    .result-card {
        padding: 1.8rem;
        border-radius: 16px;
        margin-top: 1.2rem;
        margin-bottom: 1.2rem;
        animation: fadeIn 0.35s cubic-bezier(0.16, 1, 0.3, 1);
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05);
    }
    
    .result-spam {
        background: linear-gradient(135deg, #FFF1F2 0%, #FFE4E6 100%);
        border: 2px solid #F43F5E;
        color: #881337;
    }
    
    .result-ham {
        background: linear-gradient(135deg, #F0FDF4 0%, #DCFCE7 100%);
        border: 2px solid #22C55E;
        color: #14532D;
    }
    
    .chip {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
    }
    
    .chip-spam {
        background-color: #FECDD3;
        color: #9F1239;
        border: 1px solid #FDA4AF;
    }

    .chip-ham {
        background-color: #BBF7D0;
        color: #166534;
        border: 1px solid #86EFAC;
    }

    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        margin-bottom: 10px;
    }
    
    .metric-card h4 {
        margin: 0;
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .metric-card p {
        margin: 6px 0 0 0;
        font-size: 1.6rem;
        font-weight: 700;
        color: #0F172A;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(6px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_artifacts():
    model_path = os.path.join("models", "spam_model.pkl")
    vec_path = os.path.join("models", "tfidf_vectorizer.pkl")
    if not os.path.exists(model_path) or not os.path.exists(vec_path):
        return None, None
    model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)
    return model, vectorizer


@st.cache_data
def load_and_prepare_dataset():
    data_path = os.path.join("data", "spam.csv")
    if not os.path.exists(data_path):
        return None, None, None, None, None
    df = pd.read_csv(data_path)
    df["cleaned"] = df["message"].apply(clean_text)
    df["char_len"] = df["message"].apply(len)
    df["word_len"] = df["message"].apply(lambda x: len(str(x).split()))
    
    X = df["cleaned"]
    y = df["label"].map({"ham": 0, "spam": 1})
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    return df, X_train, X_test, y_train, y_test


def analyze_message_tokens(cleaned_text, vectorizer, model):
    """Calculates spam vs ham log-odds influence score for words in message."""
    words = cleaned_text.split()
    if not words or not hasattr(vectorizer, 'vocabulary_'):
        return []

    vocab = vectorizer.vocabulary_
    ham_log_probs = model.feature_log_prob_[0]
    spam_log_probs = model.feature_log_prob_[1]

    found_triggers = []
    seen = set()
    for word in words:
        if word in vocab and word not in seen:
            seen.add(word)
            idx = vocab[word]
            spam_lp = spam_log_probs[idx]
            ham_lp = ham_log_probs[idx]
            # Log-odds ratio: > 0 means pushes towards spam, < 0 means pushes towards ham
            log_odds = spam_lp - ham_lp
            found_triggers.append({
                "word": word,
                "spam_score": spam_lp,
                "log_odds": log_odds,
                "leaning": "Spam" if log_odds > 0 else "Ham"
            })

    # Sort descending by absolute impact
    found_triggers.sort(key=lambda x: abs(x["log_odds"]), reverse=True)
    return found_triggers[:10]


def set_input_text(text: str):
    st.session_state["user_message_input"] = text


def clear_input_text():
    st.session_state["user_message_input"] = ""


def main():
    model, vectorizer = load_artifacts()
    df, X_train, X_test, y_train, y_test = load_and_prepare_dataset()

    # Session State Initialization for Live Input
    if "user_message_input" not in st.session_state:
        st.session_state["user_message_input"] = ""

    # Hero Header Banner
    st.markdown("""
    <div class="header-container">
        <h1 class="header-title">🛡️ SpamGuard AI — Live Email & SMS Filter</h1>
        <p class="header-subtitle">Advanced text classification and intelligence engine powered by <b>TF-IDF N-Gram Vectorization</b> and <b>Multinomial Naive Bayes</b>.</p>
        <span class="stat-badge">⚡ Accuracy: <b>98.21%</b></span>
        <span class="stat-badge">🎯 Spam Precision: <b>97.78%</b></span>
        <span class="stat-badge">📦 5,572 Clean Records</span>
        <span class="stat-badge">⏱️ Inference: <b>&lt; 5ms</b></span>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar Architecture & Model Metadata
    with st.sidebar:
        st.subheader("⚙️ System Architecture")
        st.info(
            "**Classification Engine:** `MultinomialNB(alpha=0.1)`\n\n"
            "**Feature Extractor:** `TfidfVectorizer(1-2 ngrams, 5000 features)`\n\n"
            "**Split Strategy:** 80% Train (4,457) / 20% Test (1,115)"
        )
        
        st.markdown("---")
        st.subheader("📊 Key Benchmark Metrics")
        col_sb1, col_sb2 = st.columns(2)
        col_sb1.metric("Accuracy", "98.21%")
        col_sb2.metric("Precision", "97.78%")
        col_sb1.metric("Recall", "88.59%")
        col_sb2.metric("F1-Score", "92.96%")

        st.markdown("---")
        st.subheader("💡 Tips for Best Results")
        st.caption("• Test both authentic and tricky fraudulent messages.")
        st.caption("• Inspect the live word influence breakdown to see why words trigger the spam filter.")
        st.caption("• Explore Tab 3 for dynamic threshold and confusion matrix exploration.")

    # Main Navigation Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Live Message Classifier",
        "📑 Batch File Processing (.CSV / .TXT)",
        "📈 Visual Pipeline & Dynamic Confusion Matrix",
        "🎓 College Viva & Interview Guide"
    ])

    # ==========================================
    # TAB 1: LIVE MESSAGE CLASSIFIER
    # ==========================================
    with tab1:
        if model is None or vectorizer is None:
            st.error("⚠️ Model artifacts not found. Please train the model first by running `python src/train.py` in your terminal.")
            return

        col_input, col_samples = st.columns([3, 2], gap="large")

        with col_samples:
            st.markdown("### ⚡ Quick-Test Presets")
            st.caption("Click any preset button below to immediately load a test sample:")

            p1 = "URGENT! You have won £1,000 cash prize! Claim your reward now by texting CLAIM to 87121."
            p2 = "Hey, are you free for lunch tomorrow at 1 PM near the library?"
            p3 = "Congratulations! You have been selected for a free $500 Amazon gift card. Click http://bit.ly/gift to verify."
            p4 = "Hi team, please find attached the weekly sales report and presentation slides."
            p5 = "FINAL NOTICE: Your credit account is blocked. Verify your details immediately to avoid fees."

            st.button("🚨 Urgent Cash Prize (Spam)", on_click=set_input_text, args=(p1,), use_container_width=True)
            st.button("👥 Lunch Invitation (Ham)", on_click=set_input_text, args=(p2,), use_container_width=True)
            st.button("🎁 Free Gift Card Phishing (Spam)", on_click=set_input_text, args=(p3,), use_container_width=True)
            st.button("💼 Office Weekly Report (Ham)", on_click=set_input_text, args=(p4,), use_container_width=True)
            st.button("⚠️ Account Block Notice (Spam)", on_click=set_input_text, args=(p5,), use_container_width=True)

        with col_input:
            st.markdown("### ✍️ Enter Custom Email or SMS")
            
            # Bound directly to session state
            user_text = st.text_area(
                "Type or paste message contents:",
                height=150,
                placeholder="e.g. Winner! Call now to receive your guaranteed reward...",
                key="user_message_input"
            )

            btn_col1, btn_col2, btn_col3 = st.columns([1.5, 1.5, 3])
            with btn_col1:
                scan_btn = st.button("🚀 Analyze Now", type="primary", use_container_width=True)
            with btn_col2:
                st.button("🧹 Clear Input", on_click=clear_input_text, use_container_width=True)

            # Prediction Execution
            if user_text.strip():
                cleaned = clean_text(user_text)
                vec_features = vectorizer.transform([cleaned])
                pred = model.predict(vec_features)[0]
                probabilities = model.predict_proba(vec_features)[0]

                ham_pct = probabilities[0] * 100
                spam_pct = probabilities[1] * 100

                if pred == 1:
                    st.markdown(f"""
                    <div class="result-card result-spam">
                        <h2 style="margin:0 0 8px 0; color:#881337; font-size:1.6rem;">🔴 Verdict: SPAM DETECTED</h2>
                        <p style="margin:0; font-size:1.02rem; line-height:1.5;">This message contains trigger patterns, urgent phrasing, or promotional indicators characteristic of spam & phishing attempts.</p>
                        <hr style="border:none; border-top:1px solid #FDA4AF; margin:14px 0;">
                        <div style="font-size:1.05rem; font-weight:700;">Spam Probability: <span style="color:#E11D48;">{spam_pct:.2f}%</span> &nbsp;|&nbsp; Legitimate (Ham): {ham_pct:.2f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-card result-ham">
                        <h2 style="margin:0 0 8px 0; color:#14532D; font-size:1.6rem;">🟢 Verdict: LEGITIMATE (HAM)</h2>
                        <p style="margin:0; font-size:1.02rem; line-height:1.5;">This message appears natural, authentic, and safe for regular communication.</p>
                        <hr style="border:none; border-top:1px solid #86EFAC; margin:14px 0;">
                        <div style="font-size:1.05rem; font-weight:700;">Legitimate Probability: <span style="color:#16A34A;">{ham_pct:.2f}%</span> &nbsp;|&nbsp; Spam Probability: {spam_pct:.2f}%</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("#### 📊 Confidence Breakdown")
                st.progress(float(spam_pct / 100.0), text=f"Spam Likelihood: {spam_pct:.1f}%")

                # Detected keywords and Token Influence
                triggers = analyze_message_tokens(cleaned, vectorizer, model)
                if triggers:
                    st.markdown("#### 🔍 Influential Word Tokens in This Input:")
                    chips_html = "".join([
                        f'<span class="chip {"chip-spam" if t["leaning"] == "Spam" else "chip-ham"}">'
                        f'{"🔴" if t["leaning"] == "Spam" else "🟢"} <b>{t["word"]}</b> ({t["leaning"]})'
                        f'</span>' for t in triggers
                    ])
                    st.markdown(chips_html, unsafe_allow_html=True)
                    
                    with st.expander("📊 View Per-Token Log-Odds Contribution Chart"):
                        chart_df = pd.DataFrame(triggers)
                        fig, ax = plt.subplots(figsize=(8, 3.5))
                        colors = ['#F43F5E' if x > 0 else '#22C55E' for x in chart_df['log_odds']]
                        ax.barh(chart_df['word'], chart_df['log_odds'], color=colors)
                        ax.axvline(0, color='#64748B', linestyle='--', linewidth=0.8)
                        ax.set_xlabel("Relative Weight ( > 0 Spam Leaning | < 0 Ham Leaning )", fontsize=10)
                        ax.set_title("Word Influence on Classification Decision", fontsize=11, fontweight="bold")
                        ax.invert_yaxis()
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
            elif scan_btn:
                st.warning("⚠️ Please enter or select a message to analyze.")

    # ==========================================
    # TAB 2: BATCH FILE PROCESSING
    # ==========================================
    with tab2:
        st.subheader("📑 Batch Upload & Bulk Message Classifier")
        st.write("Upload a `.csv` or `.txt` file containing messages to classify multiple emails/SMS in one go.")

        uploaded_file = st.file_uploader("Upload CSV or TXT file (must contain a 'message' column for CSV)", type=["csv", "txt"])

        if uploaded_file is not None and model is not None and vectorizer is not None:
            try:
                if uploaded_file.name.endswith(".csv"):
                    batch_df = pd.read_csv(uploaded_file)
                else:
                    lines = uploaded_file.read().decode("utf-8").splitlines()
                    batch_df = pd.DataFrame({"message": [l for l in lines if l.strip()]})

                if "message" not in batch_df.columns:
                    st.warning("⚠️ Column named 'message' not found. Using the first available text column.")
                    batch_df.rename(columns={batch_df.columns[0]: "message"}, inplace=True)

                cleaned_series = batch_df["message"].apply(clean_text)
                batch_vec = vectorizer.transform(cleaned_series)
                batch_preds = model.predict(batch_vec)
                batch_probs = model.predict_proba(batch_vec)

                batch_df["Predicted_Label"] = ["SPAM 🔴" if p == 1 else "HAM 🟢" for p in batch_preds]
                batch_df["Spam_Confidence_%"] = [round(prob[1] * 100, 2) for prob in batch_probs]

                st.success(f"Successfully processed {len(batch_df)} messages!")
                
                col_b1, col_b2, col_b3 = st.columns(3)
                spam_count = int((batch_preds == 1).sum())
                ham_count = int((batch_preds == 0).sum())
                col_b1.metric("Total Rows", len(batch_df))
                col_b2.metric("Spam Found", f"{spam_count} ({spam_count/len(batch_df)*100:.1f}%)")
                col_b3.metric("Ham Found", f"{ham_count} ({ham_count/len(batch_df)*100:.1f}%)")

                st.dataframe(batch_df[["message", "Predicted_Label", "Spam_Confidence_%"]], use_container_width=True)

                # Download processed CSV
                csv_data = batch_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Classified CSV",
                    data=csv_data,
                    file_name="spam_classified_results.csv",
                    mime="text/csv",
                    type="primary"
                )
            except Exception as e:
                st.error(f"Error reading file: {e}")

    # ==========================================
    # TAB 3: VISUAL PIPELINE & DYNAMIC CONFUSION MATRIX
    # ==========================================
    with tab3:
        st.subheader("📈 Dynamic Visual Pipeline & Interactive Performance Lab")
        st.caption("Interact with model parameters, customize decision thresholds, and explore real-time confusion matrices and dataset distributions.")

        if model is not None and vectorizer is not None and X_test is not None:
            # Precompute test predictions
            X_test_vec = vectorizer.transform(X_test)
            y_test_probs = model.predict_proba(X_test_vec)[:, 1]

            st.markdown("### 🎛️ Interactive Decision Threshold & Confusion Matrix")
            st.markdown(
                "Adjust the **Spam Decision Threshold** below to observe how the **Confusion Matrix**, **Precision**, **Recall**, and **False Positives** update dynamically in real time."
            )

            threshold_col1, threshold_col2 = st.columns([2, 3])
            with threshold_col1:
                threshold = st.slider(
                    "Spam Probability Threshold",
                    min_value=0.01,
                    max_value=0.99,
                    value=0.50,
                    step=0.01,
                    help="Messages with spam probability ≥ this threshold will be flagged as SPAM."
                )

                # Compute dynamic metrics
                y_pred_dynamic = (y_test_probs >= threshold).astype(int)
                cm_dyn = confusion_matrix(y_test, y_pred_dynamic)
                acc_dyn = accuracy_score(y_test, y_pred_dynamic)
                prec_dyn = precision_score(y_test, y_pred_dynamic, zero_division=0)
                rec_dyn = recall_score(y_test, y_pred_dynamic, zero_division=0)
                f1_dyn = f1_score(y_test, y_pred_dynamic, zero_division=0)

                tn, fp, fn, tp = cm_dyn.ravel() if cm_dyn.shape == (2, 2) else (0, 0, 0, 0)

                m_col1, m_col2 = st.columns(2)
                m_col1.metric("Accuracy", f"{acc_dyn * 100:.2f}%")
                m_col2.metric("Precision (Spam)", f"{prec_dyn * 100:.2f}%")
                m_col1.metric("Recall (Spam)", f"{rec_dyn * 100:.2f}%")
                m_col2.metric("F1-Score", f"{f1_dyn * 100:.2f}%")

                st.markdown(f"""
                <div style="background:#F8FAFC; border:1px solid #E2E8F0; padding:14px; border-radius:12px; margin-top:10px;">
                    <div style="font-size:0.9rem; margin-bottom:4px;"><b>🟢 True Ham (Correct Inbox):</b> {tn}</div>
                    <div style="font-size:0.9rem; margin-bottom:4px; color:#DC2626;"><b>⚠️ False Positives (False Alarms):</b> {fp}</div>
                    <div style="font-size:0.9rem; margin-bottom:4px; color:#D97706;"><b>⚠️ False Negatives (Missed Spam):</b> {fn}</div>
                    <div style="font-size:0.9rem; color:#16A34A;"><b>🔴 True Spam (Caught):</b> {tp}</div>
                </div>
                """, unsafe_allow_html=True)

            with threshold_col2:
                # Dynamic Confusion Matrix Plot
                fig_cm, ax_cm = plt.subplots(figsize=(6, 4.5))
                sns.heatmap(
                    cm_dyn,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    cbar=False,
                    xticklabels=["Predicted Ham", "Predicted Spam"],
                    yticklabels=["Actual Ham", "Actual Spam"],
                    annot_kws={"size": 15, "weight": "bold"},
                    ax=ax_cm
                )
                ax_cm.set_title(f"Dynamic Confusion Matrix (Threshold = {threshold:.2f})", fontsize=12, fontweight="bold", pad=12)
                plt.tight_layout()
                st.pyplot(fig_cm)
                plt.close()

            st.markdown("---")
            
            # Interactive Dataset & Feature Explorer
            st.markdown("### 📊 Dataset Exploration & Distribution Insights")
            d_tab1, d_tab2 = st.columns(2)

            with d_tab1:
                st.markdown("#### Message Length Distribution (Ham vs Spam)")
                fig_len, ax_len = plt.subplots(figsize=(7, 4.2))
                sns.histplot(data=df, x="char_len", hue="label", bins=40, palette={"ham": "#22C55E", "spam": "#F43F5E"}, kde=True, ax=ax_len)
                ax_len.set_xlim(0, 300)
                ax_len.set_xlabel("Character Count")
                ax_len.set_ylabel("Message Frequency")
                ax_len.set_title("Character Length by Class", fontsize=11, fontweight="bold")
                plt.tight_layout()
                st.pyplot(fig_len)
                plt.close()

            with d_tab2:
                st.markdown("#### Top Vocabulary Features (Spam Indicative)")
                # Extract top spam words
                vocab = vectorizer.vocabulary_
                spam_log_probs = model.feature_log_prob_[1]
                inv_vocab = {v: k for k, v in vocab.items()}
                top_indices = np.argsort(spam_log_probs)[-15:]
                top_words = [inv_vocab[i] for i in top_indices]
                top_scores = [spam_log_probs[i] for i in top_indices]

                fig_top, ax_top = plt.subplots(figsize=(7, 4.2))
                ax_top.barh(top_words, top_scores, color="#EF4444")
                ax_top.set_xlabel("Log Probability in Spam Class")
                ax_top.set_title("Top 15 Most Indicative Spam Terms", fontsize=11, fontweight="bold")
                plt.tight_layout()
                st.pyplot(fig_top)
                plt.close()

            with st.expander("🖼️ View Saved High-Resolution Benchmark Assets"):
                c1, c2 = st.columns(2)
                if os.path.exists("assets/screenshots/class_distribution.png"):
                    c1.image("assets/screenshots/class_distribution.png", caption="Benchmark Class Distribution", use_container_width=True)
                if os.path.exists("assets/screenshots/confusion_matrix_heatmap.png"):
                    c2.image("assets/screenshots/confusion_matrix_heatmap.png", caption="Static Benchmark Confusion Matrix", use_container_width=True)

        else:
            st.info("Train the model and load the dataset to view dynamic visual pipelines.")

    # ==========================================
    # TAB 4: COLLEGE VIVA & INTERVIEW GUIDE
    # ==========================================
    with tab4:
        st.subheader("🎓 College Viva, Project Defense & Technical Q&A")

        with st.expander("❓ 1. What is the difference between Ham and Spam?"):
            st.write(
                "**Ham** represents genuine, legitimate, and safe messages (e.g. personal communications, work requests, college notices).\n\n"
                "**Spam** refers to bulk, unsolicited messages such as lottery frauds, phishing links, and deceptive marketing."
            )

        with st.expander("❓ 2. How does TF-IDF feature extraction work?"):
            st.write(
                "- **TF (Term Frequency):** Measures the frequency of a word in a specific message.\n"
                "- **IDF (Inverse Document Frequency):** Reduces the score of common English words (like *the*, *is*, *at*) and raises the score of informative keywords (*lottery*, *urgent*, *claim*, *prize*).\n"
                "- Transforms human text into numerical feature vectors that machine learning models can compute."
            )

        with st.expander("❓ 3. Why did you use Multinomial Naive Bayes instead of complex Deep Learning?"):
            st.write(
                "- **Fast & Lightweight:** Trains in milliseconds without requiring GPUs.\n"
                "- **Well-Suited for Text:** Word counts and frequencies follow discrete multinomial distributions where Naive Bayes excels.\n"
                "- **High Benchmark Score:** Achieved 98.21% accuracy with 97.78% precision, proving high efficiency with minimal compute overhead."
            )

        with st.expander("❓ 4. Why is Precision prioritized over Recall for Spam detection?"):
            st.write(
                "In spam filtering, a **False Positive** (wrongly categorizing an important email as spam) is far more disruptive than a **False Negative** (a minor spam message landing in the inbox). High precision guarantees genuine messages are never lost."
            )

        with st.expander("❓ 5. What is Laplace Smoothing (alpha=0.1) in Naive Bayes?"):
            st.write(
                "Laplace smoothing prevents the **Zero Probability Problem** when an unseen word appears in a new test message, preventing the entire probability product from multiplying to zero."
            )


if __name__ == "__main__":
    main()
