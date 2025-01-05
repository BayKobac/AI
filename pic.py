import pickle
import os

# 현재 스크립트 파일의 경로
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
txt_file_path = os.path.join(BASE_DIR, "words_alpha.txt")
pkl_file_path = os.path.join(BASE_DIR, "words_dataset.pkl")

# 텍스트 파일에서 단어 로드
def load_words_from_file(file_path):
    with open(file_path, "r") as file:
        return file.read().splitlines()

# 데이터셋 로드
words = load_words_from_file(txt_file_path)

# 피클 파일로 저장
with open(pkl_file_path, "wb") as file:
    pickle.dump(words, file)

print(f"피클 파일 생성 완료: {pkl_file_path}")

