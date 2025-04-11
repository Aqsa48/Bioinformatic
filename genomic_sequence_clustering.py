import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# ---------------------- Sequence Preprocessing ---------------------- #
def shorten_sequence(seq_str, length=20):
    return seq_str[:length]  # Truncate to a specific length

def pre_process(seq_str):
    seq_lst = []
    for i in range(0, len(seq_str), 3):
        if len(seq_str[i:i+3]) == 3:
            seq_lst.append(seq_str[i:i+3])
    return seq_lst

def clean_sequence(seq):
    cleaned_seq = ''.join([c for c in seq if c in 'ATCG'])  # Remove ambiguous characters
    return " ".join(pre_process(cleaned_seq))

def extract_codons_from_sequence(seq):
    cleaned_seq = ''.join([c for c in seq if c in 'ATCG'])  # Ensure only valid nucleotides
    codons = []
    for i in range(0, len(cleaned_seq) - 2, 3):
        codon = cleaned_seq[i:i+3]
        if len(codon) == 3:
            codons.append(codon)
    return codons

# ---------------------- Optimized Batch-Wise Association Rule Mining ---------------------- #
def perform_association_rule_mining(processed_sequences, min_support=0.05, min_confidence=0.5, batch_size=100):
    all_frequent_itemsets = []
    all_rules = []

    for i in range(0, len(processed_sequences), batch_size):
        batch = processed_sequences[i:i + batch_size]
        transactions = [extract_codons_from_sequence(seq.replace(" ", "")) for seq in batch]
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df = pd.DataFrame(te_ary, columns=te.columns_)

        frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
        if not frequent_itemsets.empty:
            rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
            all_frequent_itemsets.append(frequent_itemsets)
            all_rules.append(rules)

    final_itemsets = pd.concat(all_frequent_itemsets).drop_duplicates().reset_index(drop=True) if all_frequent_itemsets else pd.DataFrame()
    final_rules = pd.concat(all_rules).drop_duplicates().reset_index(drop=True) if all_rules else pd.DataFrame()
    return final_itemsets, final_rules

# ---------------------- Streamlit UI ---------------------- #
st.set_page_config(layout="wide")
st.title("🔬 Genetic Evolution of SARS-CoV-2 | Clustering and Apriori Rule Mining")

# Sidebar for dataset upload
st.sidebar.header("📁 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
    st.sidebar.success("✅ File uploaded successfully!")
    
    col1, col2, col3 = st.sidebar.columns(3)
    with col1:
        view_data = st.button("📊 View Data")
    with col2:
        preprocess_data = st.button("⚙️ Preprocess")
    with col3:
        download_data = st.button("⬇️ Download")

    if view_data:
        st.write("### 📄 Original Dataset Preview")
        st.dataframe(df.head())

    if preprocess_data:
        if "Header" in df.columns and "Sequence" in df.columns:
            df["Original_Processed_Sequence"] = df["Sequence"].apply(clean_sequence)
            df["Shortened_Sequence"] = df["Sequence"].apply(shorten_sequence)
            df["Shortened_Processed_Sequence"] = df["Shortened_Sequence"].apply(clean_sequence)

            st.session_state["original_df"] = df
            st.session_state["shortened_df"] = df[['Header', 'Shortened_Processed_Sequence']]

            st.write("### ✅ Preprocessed Datasets")
            st.dataframe(df.head())
        else:
            st.error("❌ The uploaded file must contain 'Header' and 'Sequence' columns.")

    if download_data:
        if "original_df" in st.session_state:
            csv = st.session_state["original_df"].to_csv(index=False)
            st.download_button("Download Preprocessed Original Dataset", csv, "original_preprocessed_data.csv", "text/csv")
        else:
            st.error("❗ Please preprocess the data first.")

    if "original_df" in st.session_state and "shortened_df" in st.session_state:
        original_df = st.session_state["original_df"]
        shortened_df = st.session_state["shortened_df"]

        # -------------------- TF-IDF + Clustering -------------------- #
        st.sidebar.header("⚙️ Clustering Options")
        tfidf_range = st.sidebar.slider("Top TF-IDF words", 10, 50, 10)
        num_clusters = st.sidebar.slider("Number of Clusters (K)", 2, 10, 3)

        vectorizer = TfidfVectorizer()
        X_original = vectorizer.fit_transform(original_df["Original_Processed_Sequence"])
        tfidf_feature_names = vectorizer.get_feature_names_out()
        word_counts = X_original.toarray().sum(axis=0)
        word_counts_df = pd.DataFrame({'Word': tfidf_feature_names, 'Count': word_counts.flatten()})
        word_counts_df = word_counts_df.sort_values(by='Count', ascending=False).head(tfidf_range)

        st.write("## 🧬 TF-IDF Feature Extraction (Original Data)")
        st.dataframe(word_counts_df)

        # Clustering
        model = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=100, n_init=10, random_state=42)
        original_df['Cluster'] = model.fit_predict(X_original)

        st.write("## 🔍 Cluster-wise Codon Analysis")

        # Codon Extraction per Cluster
        codon_clusters = {i: [] for i in range(num_clusters)}
        for i, row in original_df.iterrows():
            codons = extract_codons_from_sequence(row['Original_Processed_Sequence'])
            codon_clusters[row['Cluster']].extend(codons)

        # Top 10 Codons per Cluster
        top_codons_per_cluster = {}
        for cluster_id, codons in codon_clusters.items():
            codon_count = pd.Series(codons).value_counts().head(10)
            top_codons_per_cluster[cluster_id] = codon_count

        codon_df = pd.DataFrame(top_codons_per_cluster).fillna(0).astype(int)

        fig, ax = plt.subplots(figsize=(10, 6))
        codon_df.plot(kind='bar', ax=ax)
        ax.set_title('Top 10 Codons per Cluster')
        ax.set_xlabel('Codons')
        ax.set_ylabel('Count')
        ax.set_xticklabels(codon_df.index, rotation=45)
        st.pyplot(fig)

        total_codons = sum(len(extract_codons_from_sequence(seq)) for seq in original_df['Original_Processed_Sequence'])
        st.write(f"### Total Number of Codons in Dataset: {total_codons}")

        # -------------------- Association Rule Mining -------------------- #
        st.sidebar.markdown("---")
        run_rule_mining = st.sidebar.checkbox("Run Association Rule Mining")

        if run_rule_mining:
            st.write("## 🔗 Association Rule Mining")

            min_support = st.sidebar.slider("Minimum Support", 0.01, 0.5, 0.08, 0.01)
            min_confidence = st.sidebar.slider("Minimum Confidence", 0.1, 1.0, 0.8, 0.05)

            try:
                with st.spinner("⏳ Mining rules..."):
                    frequent_itemsets, rules = perform_association_rule_mining(
                        shortened_df["Shortened_Processed_Sequence"],
                        min_support,
                        min_confidence,
                        batch_size=100
                    )

                if not frequent_itemsets.empty:
                    st.write("### 📋 Frequent Itemsets (Top 10)")
                    st.dataframe(frequent_itemsets.head(10))

                if not rules.empty:
                    st.write("### 📎 Association Rules (Top 10)")
                    st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))

                    rules_csv = rules.to_csv(index=False)
                    st.download_button("⬇️ Download Rules as CSV", rules_csv, "association_rules.csv", "text/csv")
                else:
                    st.warning("⚠️ No rules found. Try lowering support or confidence.")

            except Exception as e:
                st.error(f"❌ Error during rule mining: {e}")
