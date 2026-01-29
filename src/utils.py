import os
import sqlite3

import docx
from PyPDF2 import PdfReader
from io import BytesIO

from nltk.corpus.reader import documents


def read_txt_file(file_bytes: bytes) -> str | None:
    try:
        return file_bytes.decode("utf-8")
    except Exception as e:
        print(f"Error reading TXT file: {e}")
        return None

def read_docx_file(file_stream) -> str | None:
    try:
        document = docx.Document(file_stream)
        return "\n".join(p.text.strip() for p in document.paragraphs)
    except Exception as e:
        print(f"Error reading DOCX file: {e}")
        return None



def read_pdf_file(file_stream) -> str | None:
    try:
        reader = PdfReader(file_stream)

        if reader.is_encrypted:
            print("Error: PDF is encrypted.")
            return None

        text = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)

        return "".join(text)

    except Exception as e:
        print(f"Error reading PDF file: {e}")
        return None








def read_uploaded_file(uploaded_file):
    _, extension = os.path.splitext(uploaded_file.name.lower())
    file_bytes = uploaded_file.getvalue()
    file_stream = BytesIO(file_bytes)

    if extension == ".txt":
        return read_txt_file(file_bytes)
    elif extension == ".docx":
        return read_docx_file(file_stream)
    elif extension == ".pdf":
        return read_pdf_file(file_stream)
    else:
        raise ValueError(f"Unsupported file type: {extension}")
conn=None
def get_all_documents_from_db(db_path="corpus.db"):
    if not os.path.exists(db_path):
        print(f"Database file not fount at {db_path}")
        return []
    try:
        conn=sqlite3.connect(db_path)
        cursor=conn.cursor()
        select_query = "SELECT filename, text_content FROM documents"
        cursor.execute(select_query)
        documents=cursor.fetchall()
        return documents
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return []  # Return an empty list on error

    finally:
        if conn:
            conn.close()


def insert_document_into_db(filename: str, text: str,db_path="corpus.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT OR IGNORE INTO documents (filename, text_content) VALUES (?, ?)",
            (filename, text)
        )
        conn.commit()
    finally:
        conn.close()


def get_document_by_filename(filename, db_path="corpus.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT filename, text_content, upload_date FROM documents WHERE filename = ?",
        (filename,)
    )
    row = cursor.fetchone()
    conn.close()
    return row


def delete_document_by_filename(filename, db_path="corpus.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "DELETE FROM documents WHERE filename = ?",
        (filename,)
    )
    conn.commit()
    conn.close()
