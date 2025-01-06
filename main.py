from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import random
import pickle
import os

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "words_dataset.pkl")

def load_words_from_pickle(file_path: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at {file_path}")
    with open(file_path, "rb") as pkl_file:
        return pickle.load(pkl_file)

WORD_DATABASE = load_words_from_pickle(DATASET_PATH)

class WordRequest(BaseModel):
    word_count: int = 1000  

def calculate_similarity(input_word: str, word_list: list[str]) -> list[dict]:

    input_vector = np.array([[ord(char) for char in input_word]])
    word_vectors = [np.array([ord(char) for char in word]) for word in word_list]

    max_length = max(len(input_word), max(len(word) for word in word_list))
    input_vector = np.pad(input_vector, ((0, 0), (0, max_length - input_vector.shape[1])), constant_values=0)
    word_vectors = [np.pad(w, (0, max_length - len(w)), constant_values=0) for w in word_vectors]

    similarities = cosine_similarity([input_vector[0]], word_vectors)[0]
    return [{"word": word, "similarity": sim} for word, sim in zip(word_list, similarities)]

@app.post("/get-word-and-similarities")
async def get_word_and_similarities(request: WordRequest):
    try:

        selected_word = random.choice(WORD_DATABASE)

        similarities = calculate_similarity(selected_word, WORD_DATABASE)
        sorted_similarities = sorted(similarities, key=lambda x: x["similarity"], reverse=True)[:request.word_count]

        response = {
            "answer": {"word": selected_word},  
            "similar_words": sorted_similarities  
        }

        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))