from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import random
import pickle
import os

app = FastAPI()

# 프로젝트 루트 디렉토리를 기준으로 데이터셋 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "words_dataset.pkl")

# 피클 파일에서 데이터셋 로드
def load_words_from_pickle(file_path: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at {file_path}")
    with open(file_path, "rb") as pkl_file:
        return pickle.load(pkl_file)

# 데이터셋 로드
WORD_DATABASE = load_words_from_pickle(DATASET_PATH)

# Pydantic 모델 정의
class WordRequest(BaseModel):
    word_count: int = 1000  # 반환할 단어 수 기본값

# 유사도 계산 함수
def calculate_similarity(input_word: str, word_list: list[str]) -> list[dict]:
    # 입력 단어와 단어 리스트를 벡터화
    input_vector = np.array([[ord(char) for char in input_word]])
    word_vectors = [np.array([ord(char) for char in word]) for word in word_list]

    # 벡터 길이를 맞추기 위해 패딩 추가
    max_length = max(len(input_word), max(len(word) for word in word_list))
    input_vector = np.pad(input_vector, ((0, 0), (0, max_length - input_vector.shape[1])), constant_values=0)
    word_vectors = [np.pad(w, (0, max_length - len(w)), constant_values=0) for w in word_vectors]

    # 코사인 유사도 계산
    similarities = cosine_similarity([input_vector[0]], word_vectors)[0]
    return [{"word": word, "similarity": sim} for word, sim in zip(word_list, similarities)]

@app.post("/get-word-and-similarities")
async def get_word_and_similarities(request: WordRequest):
    try:
        # 정답 단어 선정
        selected_word = random.choice(WORD_DATABASE)

        # 유사 단어 계산
        similarities = calculate_similarity(selected_word, WORD_DATABASE)
        sorted_similarities = sorted(similarities, key=lambda x: x["similarity"], reverse=True)[:request.word_count]

        # 응답 데이터 구성
        response = {
            "answer": {"word": selected_word},  # 정답 단어
            "similar_words": sorted_similarities  # 유사 단어 리스트 (유사도 포함)
        }

        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))