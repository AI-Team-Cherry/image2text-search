# 🖼️ Image-to-Text Search with CLIP & FAISS

이 프로젝트는 **CLIP 모델**과 **FAISS(Vector Similarity Search)**를 활용하여  
자연어 텍스트 쿼리로부터 유사한 패션 이미지를 검색하는 시스템입니다.  
패션 도메인 데이터셋을 기반으로, 검색 결과를 시각적으로 확인할 수 있습니다.

---

## 🚀 프로젝트 개요
- 상품 이미지를 임베딩(embedding) 벡터로 변환
- 텍스트 쿼리를 CLIP 텍스트 인코더로 벡터화
- FAISS를 이용해 이미지 임베딩과 텍스트 임베딩 간 코사인 유사도 기반 검색
- 검색된 이미지를 직관적으로 시각화하여 결과 확인

---

## 📌 주요 기능
1. **데이터 처리**
   - `caption(fashion-clip)_embedding.csv` (텍스트 임베딩)
   - `image_embedding.csv` (이미지 임베딩)
   - 공통 키(`image_file`)를 기준으로 병합 및 벡터 변환

2. **임베딩 정규화**
   - L2 normalization 적용
   - float32 변환을 통한 연산 최적화

3. **FAISS 인덱스 구축**
   - `IndexFlatIP` (Inner Product) 방식
   - Cosine similarity 기반 검색 지원

4. **텍스트 → 이미지 검색**
   - CLIP(`openai/clip-vit-base-patch32`) 텍스트 인코더 사용
   - 입력 문장을 벡터로 변환 후 FAISS 검색 수행
   - 검색 결과 중복 제거 및 Top-K 이미지 반환

5. **시각화**
   - Matplotlib 기반 이미지 출력
   - 검색 결과를 행렬 형태(최대 5개 가로 배치)로 표시

---

## 📂 파일 구조
merge + FAISS.py # 메인 실행 스크립트
caption(fashion-clip)_embedding.csv # 텍스트 임베딩 CSV
image_embedding.csv # 이미지 임베딩 CSV
only_product_images/ # 상품 이미지 폴더



---

## 🛠 실행 방법 (Step by Step)

### 1) 환경 준비
Python 3.9+ 환경에서 필요한 패키지를 설치하세요.

```bash
pip install pandas numpy faiss-cpu torch transformers pillow matplotlib

2) 데이터 준비
같은 폴더 안에 아래 파일/폴더가 있어야 합니다:

merge + FAISS.py

caption(fashion-clip)_embedding.csv

image_embedding.csv

only_product_images/

3) 실행
터미널에서 실행:

bash
코드 복사
python "merge + FAISS.py"
4) 검색 함수 실행
파이썬 코드 내에서 원하는 쿼리를 넣어 실행합니다:

python
코드 복사
search_and_show("blue, shirt, striped", top_k=5)
