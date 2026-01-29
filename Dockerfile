FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ✅ Pre-download NLTK data (covers both old and new NLTK)
RUN python -c "import nltk; \
    nltk.download('punkt'); \
    nltk.download('punkt_tab'); \
    nltk.download('stopwords'); \
    nltk.download('wordnet')"

COPY . .

RUN mkdir -p models

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]