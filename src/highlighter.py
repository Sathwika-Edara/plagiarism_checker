import nltk
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import html
from src.vectorizer import vectorize_corpus


def split_into_sentences(text):
    if not text:
        return []
    sentences=nltk.sent_tokenize(text)
    return sentences


def vectorize_sentences(source_sentences, suspect_sentences):
    if not source_sentences and not suspect_sentences:
        return (None, None)

    all_sentences = source_sentences + suspect_sentences
    vectorized_corpus = vectorize_corpus(all_sentences)
    num_source_sentences = len(source_sentences)
    source_vectors = vectorized_corpus[:num_source_sentences]
    suspect_vectors = vectorized_corpus[num_source_sentences:]

    return source_vectors, suspect_vectors


def calculate_sentence_similarity_matrix(source_vectors, suspect_vectors):
    if source_vectors is None or suspect_vectors is None:
        return np.array([])
    if source_vectors.shape[0] == 0 or suspect_vectors.shape[0] == 0:
        return np.array([])
    similarity_matrix = cosine_similarity(suspect_vectors, source_vectors)

    return similarity_matrix

def identify_matching_sentences(similarity_matrix, threshold=0.8):
    if similarity_matrix.size == 0:
        return set()
    matching_pairs = np.argwhere(similarity_matrix >= threshold)
    plagiarized_suspect_indices = [match[0] for match in matching_pairs]
    return set(plagiarized_suspect_indices)


def generate_html_report(suspect_sentences, plagiarized_indices):
    highlighted_html_parts = []
    for i, sentence in enumerate(suspect_sentences):
        safe_sentence = html.escape(sentence)
        if i in plagiarized_indices:
            highlighted_sentence = f'<mark class="highlight">{safe_sentence}</mark>'
            highlighted_html_parts.append(highlighted_sentence)
        else:
            highlighted_html_parts.append(safe_sentence)
    return " ".join(highlighted_html_parts)