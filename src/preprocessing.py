import string
from typing import List
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Ensure required resources exist
for resource in ("punkt", "stopwords", "wordnet"):
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource)

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

def tokenize_text(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    return word_tokenize(text)


def normalize_case(tokens: List[str]) -> List[str]:
    return [t.lower() for t in tokens]


def remove_punctuation(tokens: List[str]) -> List[str]:
    return [t for t in tokens if t.isalpha()]


def filter_stopwords(tokens: List[str]) -> List[str]:
    return [t for t in tokens if t not in STOP_WORDS]


def lemmatize_tokens(tokens: List[str]) -> List[str]:
    return [LEMMATIZER.lemmatize(t) for t in tokens]


def preprocess_text(text: str) -> List[str]:
    tokens = tokenize_text(text)
    tokens = normalize_case(tokens)
    tokens = remove_punctuation(tokens)
    tokens = filter_stopwords(tokens)
    tokens = lemmatize_tokens(tokens)
    return tokens
