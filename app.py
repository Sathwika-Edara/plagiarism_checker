import os
import sys

import joblib

from ingest import rebuild_vectorizer, create_database_and_table

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

from src.highlighter import (
    split_into_sentences,
    vectorize_sentences,
    calculate_sentence_similarity_matrix,
    identify_matching_sentences,
    generate_html_report
)

from src.utils import read_uploaded_file, get_all_documents_from_db, get_document_by_filename, \
    delete_document_by_filename
from src.vectorizer import vectorize_corpus
from src.similarity import calculate_similarity


VECTORIZER_PATH = os.path.join("models", "tfidf_vectorizer.joblib")
def load_vectorizer(path):
    try:
        vectorizer=joblib.load(path)
        return vectorizer
    except FileNotFoundError:
        st.error(
            f"Error: Vectorizer model not found at '{path}'. "
            "Please run the `ingest.py` script to create the model."
        )
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred while loading the model: {e}")
        return None


def load_css():
    st.markdown("""
        <style>
            .highlight {
                background-color: #fff8c4; /* A softer, pastel yellow */
                padding: 2px 4px;        /* Adds a little space around the text */
                border-radius: 4px;      /* Gives the highlight slightly rounded corners */
                border: 1px solid #f0e68c; /* A subtle border to define the highlight */
            }
        </style>
    """, unsafe_allow_html=True)
def corpus_management():
    # ===============================
    # 📚 CORPUS MANAGEMENT SECTION
    # ===============================

    st.subheader("📚 Corpus Management")

    corpus_docs = get_all_documents_from_db()

    if not corpus_docs:
        st.warning("Corpus is currently empty. Please add reference documents.")
    else:
        st.success(f"Corpus contains {len(corpus_docs)} documents.")
        with st.expander("View corpus documents"):
            for filename, _ in corpus_docs:
                with st.expander(f"📄 {filename}"):
                    if st.button(f"👁 View content", key=f"view_{filename}"):
                        doc = get_document_by_filename(filename)
                        if doc:
                            st.text_area(
                                "Extracted Text",
                                doc[1],
                                height=300,
                                disabled=True
                            )
                    if st.button(f"🗑 Delete {filename}", key=f"del_{filename}"):
                        st.session_state[f"confirm_{filename}"] = True

                    if st.session_state.get(f"confirm_{filename}", False):
                        st.warning(f"Are you sure you want to delete **{filename}**?")

                        col1, col2 = st.columns(2)

                        with col1:
                            if st.button("✅ Yes, delete", key=f"yes_{filename}"):
                                delete_document_by_filename(filename)
                                rebuild_vectorizer()
                                st.success(f"Deleted {filename}")
                                st.session_state.pop(f"confirm_{filename}")
                                st.rerun()

                        with col2:
                            if st.button("❌ Cancel", key=f"no_{filename}"):
                                st.session_state.pop(f"confirm_{filename}")

    st.subheader("➕ Add Documents to Corpus")

    corpus_uploads = st.file_uploader(
        "Upload reference documents (corpus)",
        type=["txt", "pdf", "docx"],
        accept_multiple_files=True,
        key="corpus_uploader"
    )

    if st.button("Add to Corpus"):
        if not corpus_uploads:
            st.warning("Please upload at least one document.")
        else:
            from src.utils import insert_document_into_db

            for file in corpus_uploads:
                text = read_uploaded_file(file)
                insert_document_into_db(file.name, text)
            with st.spinner("Updating model..."):
                rebuild_vectorizer()
            st.success("Documents added to corpus successfully.")
            st.rerun()

    st.divider()
def main():
    st.title("📄 Plagiarism Checker")
    st.markdown("""
        Welcome to **Plagiarism Checker**! This tool helps you compare two text documents 
        and calculate their similarity score to detect potential plagiarism.

        **How to use it:**
        1. Upload the first document (the 'source' or 'original' text).
        2. Upload the second document (the 'suspect' text to be checked).
        3. Click the 'Check for Plagiarism' button to see the result.

        The app supports `.txt`, `.pdf`, and `.docx` file formats.
        """, unsafe_allow_html=True)
    mode = st.radio(
        "Select Comparison Mode:",
        ("Compare two files", "Compare against corpus"),
        horizontal=True,
        label_visibility="collapsed"  # Hides the "Select Comparison Mode:" label text
    )


    uploaded_file1 = None
    uploaded_file2 = None
    st.divider()
    if mode == "Compare two files":
        st.subheader("Mode: Compare Two Files")
        st.info("Upload two documents below to calculate the similarity between them.")
        col1, col2 = st.columns(2)
        with col1:
            uploaded_file1 = st.file_uploader(
                "Upload the Source Document",
                type=['txt', 'pdf', 'docx'],
                key="file1"
            )
        with col2:
            uploaded_file2 = st.file_uploader(
                "Upload the Suspect Document",
                type=['txt', 'pdf', 'docx'],
                key="file2"
            )

    elif mode == "Compare against corpus":
        corpus_management()
        corpus_docs = get_all_documents_from_db()
        if not corpus_docs:
            st.error("Corpus is empty. Please add documents before comparing.")
            st.stop()
        st.subheader("Mode: Compare Against Corpus")
        st.info("Upload a single document to check it against all documents in the database.")

        # In this mode, we only need one file uploader.
        uploaded_file1 = st.file_uploader(
            "Upload the Suspect Document",
            type=['txt', 'pdf', 'docx'],
            key="corpus_check_file"
        )

    st.divider()
    if st.button("Check for Plagiarism"):
        if mode == "Compare two files":
            if not uploaded_file1 or not uploaded_file2:
                st.warning("Please upload both files.")
                return

            with st.spinner("Analyzing documents... This may take a moment."):
                try:
                    source_text = read_uploaded_file(uploaded_file1)
                    suspect_text = read_uploaded_file(uploaded_file2)

                    vectors = vectorize_corpus([source_text, suspect_text])
                    similarity = calculate_similarity(
                        vectors[0:1],
                        vectors[1:2]
                    )
                    tab_summary, tab_report = st.tabs(["📊 Summary", "📄 Detailed Report"])
                    with tab_summary:
                        st.success("Analysis complete!")
                        st.progress(min(max(similarity / 100, 0.0), 1.0))
                        st.metric(label="Document Similarity", value=f"{similarity:.2f}%")
                        st.divider()
                    with tab_report:
                        st.caption("Highlighted sentences indicate potential plagiarism.")
                        st.subheader("Detailed Plagiarism Report")

                        # 2. Split into sentences
                        source_sentences = split_into_sentences(source_text)
                        suspect_sentences = split_into_sentences(suspect_text)

                        # 3. Vectorize sentences
                        source_vectors, suspect_vectors = vectorize_sentences(source_sentences, suspect_sentences)

                        # 4. Calculate sentence similarity matrix
                        sim_matrix = calculate_sentence_similarity_matrix(source_vectors, suspect_vectors)

                        # 5. Identify matching sentences based on a threshold
                        plagiarized_indices = identify_matching_sentences(sim_matrix, threshold=0.8)

                        # 6. Generate the HTML report
                        html_report = generate_html_report(suspect_sentences, plagiarized_indices)

                        # 7. Render the final HTML report
                        # We use unsafe_allow_html=True because we have constructed our own HTML.
                        # This is safe because we used html.escape() in our generator function.
                        st.markdown(html_report, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"An error occurred during detailed analysis: {e}")
        elif mode == "Compare against corpus":
            vectorizer = load_vectorizer(VECTORIZER_PATH)
            if vectorizer is None:
                st.stop()
            if uploaded_file1:
                with st.spinner("Loading corpus from database and analyzing..."):
                    try:
                        suspect_text=read_uploaded_file(uploaded_file1)
                        corpus_docs=get_all_documents_from_db()
                        suspect_filename = uploaded_file1.name
                        corpus_filenames = {doc[0] for doc in corpus_docs}

                        if suspect_filename in corpus_filenames:
                            st.error(
                                "This document already exists in the corpus. "
                                "A document cannot be compared against itself."
                            )
                            st.stop()
                        if not corpus_docs:
                            st.error(
                        "Could not retrieve documents from the corpus database. Is the database empty? Please run ingest.py.")
                        else:
                            st.success(f"Successfully loaded {len(corpus_docs)} documents from the corpus.")
                            corpus_texts = [doc[1] for doc in corpus_docs]
                            master_corpus = [suspect_text] + corpus_texts
                            vectors = vectorize_corpus(master_corpus)
                            st.success("Vectorization complete! Now calculating similarities...")
                            suspect_vector = vectors[0:1]
                            corpus_vectors = vectors[1:]
                            similarity_scores_matrix = cosine_similarity(suspect_vector, corpus_vectors)
                            scores = similarity_scores_matrix.flatten()
                            corpus_filenames = [doc[0] for doc in corpus_docs]
                            results = list(zip(corpus_filenames, scores))
                            sorted_results = sorted(results, key=lambda item: item[1], reverse=True)
                            st.subheader("Top 5 Most Similar Documents from Corpus")
                            if not sorted_results:
                                st.info("No documents in the corpus to compare against.")
                            else:
                                top_5_results = sorted_results[:5]

                                for filename, score in top_5_results:
                                    percentage_score = score * 100
                                    st.markdown(f"**- {filename}:** `{percentage_score:.2f}%` similar")
                                    st.progress(score)

                                st.markdown("---")
                                st.info(
                                    "This report shows the documents from the corpus with the highest textual similarity to your uploaded document.")
                    except Exception as e:
                        st.error(f"An error occurred during processing: {e}")
            else:
                st.warning("Please upload a file to check against the corpus.")

        else:
            st.warning("Please upload both documents before checking for plagiarism.")





if __name__ == "__main__":
    create_database_and_table()
    main()
