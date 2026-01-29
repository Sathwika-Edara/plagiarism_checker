# 📄 Plagiarism Checker Pro

A sophisticated plagiarism detection tool that compares documents, calculates similarity scores, and generates detailed, highlighted reports. Built with Python, Streamlit, and Scikit-learn, this application features a high-performance, database-backed engine for checking against a large corpus of documents.

<img width="780" height="727" alt="image" src="https://github.com/user-attachments/assets/49969556-f7d4-44b0-9997-dfd16ab9da83" />


## ✨ Features

-   **Dual Comparison Modes**:
    -   **Compare Two Files**: Directly compare two documents (`.txt`, `.pdf`, `.docx`) for similarity.
    -   **Compare Against Corpus**: Check a document against other documents.
-   **Detailed Plagiarism Highlighting**: Generates a sentence-by-sentence analysis and visually highlights matching sentences in the suspect document.
-   **Dual Interfaces**:
    -   **Web Application**: A user-friendly interface built with Streamlit.
    -   **Command-Line Interface (CLI)**: A powerful CLI for scriptable, automated checks.
-   **Advanced Text Processing**: A robust pipeline that handles tokenization, lowercasing, stopword removal, and lemmatization.

## 🛠️ Tech Stack

-   **Language**: Python 3.9+
-   **Web Framework**: Streamlit
-   **Machine Learning**: Scikit-learn (for TF-IDF and Cosine Similarity)
-   **NLP**: NLTK (for tokenization, stopwords, and lemmatization)
-   **Database**: SQLite3
-   **File Handling**: `python-docx`, `PyPDF2`
-   **Model/Data Persistence**: `joblib`, `pickle`

