from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import spmatrix
import numpy as np

def calculate_similarity(vector1: spmatrix, vector2: spmatrix) -> float:
    similarity_matrix = cosine_similarity(vector1, vector2)
    similarity_score = similarity_matrix[0, 0]
    percentage_score = round(similarity_score * 100, 2)
    return percentage_score