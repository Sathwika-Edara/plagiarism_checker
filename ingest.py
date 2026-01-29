import os

import sys
import sqlite3
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer

from src.preprocessing import preprocess_text

ROOT_DIR = os.path.dirname(__file__)
sys.path.append(ROOT_DIR)



BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATABASE_FILE = os.path.join(BASE_DIR, "corpus.db")
MODELS_DIR = "models"
VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
create_table_sql = '''
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL UNIQUE,
    text_content TEXT NOT NULL,
    upload_date TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
'''




def create_database_and_table():
    try:
        conn=sqlite3.connect(DATABASE_FILE)
        cursor=conn.cursor()
        cursor.execute(create_table_sql)
        conn.commit()
        print(f"Database '{DATABASE_FILE}' is ready and 'documents' table exists.")
        conn.close()

    except sqlite3.Error as e:
        print(f"Database error: {e}")
        sys.exit(1)

def rebuild_vectorizer():
    try:
        os.makedirs(MODELS_DIR, exist_ok=True)
        conn=sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT text_content FROM documents")
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            print("Corpus is empty. No vectorizer to build.")
            if os.path.exists(VECTORIZER_PATH):
                os.remove(VECTORIZER_PATH)
                print("Removed old vectorizer file.")
            return None

        texts = [r[0] for r in rows]
        try:
            vectorizer = TfidfVectorizer(tokenizer=preprocess_text)
            vectorizer.fit(texts)
            joblib.dump(vectorizer, VECTORIZER_PATH)
            print(f"Vectorizer built with {len(texts)} document(s).")
            return vectorizer
        except ValueError as e:
            # Handles "empty vocabulary" error
            print(f"Cannot build vectorizer: {e}")
            if os.path.exists(VECTORIZER_PATH):
                os.remove(VECTORIZER_PATH)
            return None

    except sqlite3.Error as e:
        print(f"Database error: {e}")
        sys.exit(1)


def main():
    create_database_and_table()
    rebuild_vectorizer()



if __name__ == "__main__":
    main()