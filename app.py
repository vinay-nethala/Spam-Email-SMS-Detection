import os
import sys
import joblib
import pandas as pd
import streamlit as st

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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .header-container {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        border-radius: 16px;
        padding: 2.2rem;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        color: white;
    }
    
    .header-title {
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin: 0;
        background: linear-gradient(90deg, #38BDF8 0%, #818CF8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .header-subtitle {
        color: #94A3B8;
        font-size: 1.05rem;
        margin-top: 0.5rem;
        font-weight: 400;
    }
    
    .stat-badge {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 8px 14px;
        border-radius: 10px;
        font-size: 0.88rem;
        color: #E2E8F0;
        display: inline-block;
        margin-top: 10px;
    }

    .result-card {
        padding: 1.8rem;
        border-radius: 14px;
        margin-top: 1.5rem;
        animation: fadeIn 0.4s ease-in-out;
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
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 6px;
    }
    
    .chip-spam {
        background-color: #FECDD3;
        color: #9F1239;
    }

    .chip-ham {
        background-color: #BBF7D0;
        color: #166534;
    }
    
    .trigger-word {
        background-color: #FEF08A;
        color: #854D0E;
        font-weight: 600;
        padding: 2px 6px;
        border-radius: 4px;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(8px); }
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


def analyze_message_tokens(cleaned_text, vectorizer, model):
    """Identifies words that contributed most toward spam classification."""
    words = cleaned_text.split()
    feature_names = vectorizer.get_feature_names_out()
    vocab = vectorizer.vocabulary_
    spam_log_probs = model.feature_log_prob_[1]

    found_triggers = []
    for word in set(words):
        if word in vocab:
            idx = vocab[word]
            score = spam_log_probs[idx]
            found_triggers.append((word, score))

    # Sort descending by spam log probability
    found_triggers.sort(key=lambda x: x[1], reverse=True)
    return found_triggers[:8]


def main():
    model, vectorizer = load_artifacts()

    # Hero Header Banner
    st.markdown("""
    <div class="header-container">
        <h1 class="header-title">🛡️ SpamGuard AI — Live Email & SMS Filter</h1>
        <p class="header-subtitle">Advanced text classification engine using <b>TF-IDF N-Gram Vectorization</b> and <b>Multinomial Naive Bayes</b>.</p>
        <span class="stat-badge">⚡ Test Accuracy: <b>98.21%</b></span>
        <span class="stat-badge">🎯 Spam Precision: <b>97.78%</b></span>
        <span class="stat-badge">📦 5,572 Clean Records</span>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar Navigation & Settings
    with st.sidebar:
        st.subheader("⚙️ System Control & Architecture")
        st.info(
            "**Classification Engine:** `MultinomialNB(alpha=0.1)`\n\n"
            "**Feature Extractor:** `TfidfVectorizer(1-2 ngrams, 5000 feats)`\n\n"
            "**Stratified Split:** 80% Train (4,457) / 20% Test (1,115)"
        )
        
        st.markdown("---")
        st.subheader("📊 Key Benchmark Scores")
        col_sb1, col_sb2 = st.columns(2)
        col_sb1.metric("Accuracy", "98.21%")
        col_sb2.metric("Precision", "97.78%")
        col_sb1.metric("Recall", "88.59%")
        col_sb2.metric("F1-Score", "92.96%")

        st.markdown("---")
        st.caption("Developed with Python, Scikit-learn & Streamlit.")

    # Main Navigation Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Live Message Classifier",
        "📑 Batch File Processing (.CSV / .TXT)",
        "📈 Visual Pipeline & Confusion Matrix",
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

        if "input_box" not in st.session_state:
            st.session_state.input_box = ""

        with col_samples:
            st.markdown("### ⚡ Quick-Test Presets")
            st.caption("Click any preset button below to load an authentic sample:")

            p1 = "URGENT! You have won £1,000 cash prize! Claim your reward now by texting CLAIM to 87121."
            p2 = "Hey, are you free for lunch tomorrow at 1 PM near the library?"
            p3 = "Congratulations! You have been selected for a free $500 Amazon gift card. Click http://bit.ly/gift to verify."
            p4 = "Hi team, please find attached the weekly sales report and presentation slides."
            p5 = "FINAL NOTICE: Your credit account is blocked. Verify your details immediately to avoid fees."

            if st.button("🚨 Urgent Cash Prize (Spam)", use_container_width=True):
                st.session_state.input_box = p1
            if st.button("👥 Lunch Invitation (Ham)", use_container_width=True):
                st.session_state.input_box = p2
            if st.button("🎁 Free Gift Card Phishing (Spam)", use_container_width=True):
                st.session_state.input_box = p3
            if st.button("💼 Office Weekly Report (Ham)", use_container_width=True):
                st.session_state.input_box = p4
            if st.button("⚠️ Account Block Notice (Spam)", use_container_width=True):
                st.session_state.input_box = p5

        with col_input:
            st.markdown("### ✍️ Enter Custom Email or SMS")
            user_text = st.text_area(
                "Type or paste message contents:",
                value=st.session_state.input_box,
                height=150,
                placeholder="e.g. Winner! Call now to receive your guaranteed reward...",
                key="text_area_widget"
            )

            btn_col1, btn_col2 = st.columns([1, 4])
            with btn_col1:
                scan_btn = st.button("🚀 Analyze Now", type="primary", use_container_width=True)
            with btn_col2:
                if st.button("🧹 Clear Input", use_container_width=False):
                    st.session_state.input_box = ""
                    st.rerun()

            # Prediction Execution
            if scan_btn or user_text:
                if not user_text.strip():
                    st.warning("Please enter or select a message to analyze.")
                else:
                    cleaned = clean_text(user_text)
                    vec_features = vectorizer.transform([cleaned])
                    pred = model.predict(vec_features)[0]
                    probabilities = model.predict_proba(vec_features)[0]

                    ham_pct = probabilities[0] * 100
                    spam_pct = probabilities[1] * 100

                    if pred == 1:
                        st.markdown(f"""
                        <div class="result-card result-spam">
                            <h2 style="margin:0 0 8px 0; color:#881337;">🔴 Verdict: SPAM DETECTED</h2>
                            <p style="margin:0; font-size:1.05rem;">This message contains trigger patterns, urgent phrasing, or promotional indicators typically used in spam and phishing.</p>
                            <hr style="border:none; border-top:1px solid #FDA4AF; margin:12px 0;">
                            <div style="font-size:1.1rem; font-weight:700;">Spam Probability: {spam_pct:.2f}% &nbsp;|&nbsp; Legitimate (Ham): {ham_pct:.2f}%</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="result-card result-ham">
                            <h2 style="margin:0 0 8px 0; color:#14532D;">🟢 Verdict: LEGITIMATE (HAM)</h2>
                            <p style="margin:0; font-size:1.05rem;">This message appears natural, authentic, and safe for regular inbox communication.</p>
                            <hr style="border:none; border-top:1px solid #86EFAC; margin:12px 0;">
                            <div style="font-size:1.1rem; font-weight:700;">Legitimate Probability: {ham_pct:.2f}% &nbsp;|&nbsp; Spam Probability: {spam_pct:.2f}%</div>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown("#### 📊 Confidence Breakdown")
                    st.progress(float(spam_pct / 100.0), text=f"Spam Likelihood: {spam_pct:.1f}%")

                    # Detected keywords
                    triggers = analyze_message_tokens(cleaned, vectorizer, model)
                    if triggers:
                        st.markdown("#### 🔍 Influential Word Tokens Detected:")
                        chips_html = "".join([f'<span class="chip {"chip-spam" if pred==1 else "chip-ham"}">{t[0]}</span>' for t in triggers])
                        st.markdown(chips_html, unsafe_allow_html=True)

    # ==========================================
    # TAB 2: BATCH FILE PROCESSING
    # ==========================================
    with tab2:
        st.subheader("📑 Batch Upload & Bulk Message Classifier")
        st.write("Upload a `.csv` or `.txt` file containing messages to classify multiple emails/SMS in one go.")

        uploaded_file = st.file_uploader("Upload CSV file (must contain a 'message' column)", type=["csv", "txt"])

        if uploaded_file is not None:
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
                spam_count = (batch_preds == 1).sum()
                ham_count = (batch_preds == 0).sum()
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
    # TAB 3: VISUAL PIPELINE & METRICS
    # ==========================================
    with tab3:
        st.subheader("📈 Machine Learning Performance & Visual Assets")
        
        col_img1, col_img2 = st.columns(2)
        with col_img1:
            if os.path.exists("assets/screenshots/class_distribution.png"):
                st.image("assets/screenshots/class_distribution.png", caption="Dataset Class Distribution (5,572 records)", use_container_width=True)
            if os.path.exists("assets/screenshots/top_spam_features.png"):
                st.image("assets/screenshots/top_spam_features.png", caption="Top 15 Most Indicative Spam Vocabulary Features", use_container_width=True)

        with col_img2:
            if os.path.exists("assets/screenshots/confusion_matrix_heatmap.png"):
                st.image("assets/screenshots/confusion_matrix_heatmap.png", caption="Confusion Matrix on 1,115 Unseen Test Messages", use_container_width=True)
            if os.path.exists("assets/screenshots/message_length_distribution.png"):
                st.image("assets/screenshots/message_length_distribution.png", caption="Message Length Analysis (Spam vs Ham)", use_container_width=True)

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


if __name__ == "__main__":
    main()
