
# pip install gensim fastapi uvicorn pydantic pickle

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from gensim.models import KeyedVectors
import pickle
import random
import os


app = FastAPI()


MODEL_PATH = "./GoogleNews-vectors-negative300.bin.gz"  # 로컬 환경에서 Word2Vec 모델 경로
DATASET_PATH = "./words_dataset.pkl"  # 단어 데이터셋 경로


try:
    print("Word2Vec 모델 로드 중...")
    model = KeyedVectors.load_word2vec_format(MODEL_PATH, binary=True)
    print("Word2Vec 모델 로드 성공")
except FileNotFoundError:
    raise Exception(f"모델 파일을 찾을 수 없습니다: {MODEL_PATH}")
except Exception as e:
    raise Exception(f"모델 로드 중 오류 발생: {e}")


def load_words_from_pickle(file_path: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"데이터셋 파일을 찾을 수 없습니다: {file_path}")
    with open(file_path, "rb") as pkl_file:
        return pickle.load(pkl_file)

try:
    print("단어 데이터베이스 로드 중...")
    WORD_DATABASE = load_words_from_pickle(DATASET_PATH)
    print(f"단어 데이터베이스 로드 성공: {len(WORD_DATABASE)}개의 단어")
except Exception as e:
    raise Exception(f"데이터베이스 로드 중 오류 발생: {e}")


class WordRequest(BaseModel):
    word_count: int = 1000  


@app.post("/get-word-and-similarities")
async def get_word_and_similarities(request: WordRequest):
    try:
        
        selected_word = random.choice(WORD_DATABASE)

        
        if selected_word not in model.key_to_index:
            raise HTTPException(status_code=s404, detail=f"선택한 단어 '{selected_word}'를 모델에서 찾을 수 없습니다.")

        
        similar_words_with_scores = model.most_similar(selected_word, topn=request.word_count)

        
        response = {
            "answer": {"word": selected_word},
            "similar_words": [
                {"word": word, "similarity": float(score)}
                for word, score in similar_words_with_scores
            ]
        }
        return response
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
