# # 1. 라이브러리 임포트
import pandas as pd
import numpy as np
import faiss
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import matplotlib.pyplot as plt
import os, math


# # 2. CSV 불러오기
caption_csv = "caption(fashion-clip)_embedding.csv"
image_csv   = "image_embedding.csv"
image_dir = "only_product_images"  

caption_df = pd.read_csv(caption_csv)
image_df   = pd.read_csv(image_csv)

# 공통 키로 merge (image_file 기준)
merged = pd.merge(image_df, caption_df, on="image_file")
print("병합된 데이터 크기:", merged.shape)


# # 3. 임베딩 컬럼 지정
txt_cols = [c for c in merged.columns if c.startswith("dim_")]  # caption CSV
img_cols = [str(i) for i in range(1, 513)]                      # image CSV

# numpy array 변환
merged["text_emb"]  = merged[txt_cols].values.tolist()
merged["image_emb"] = merged[img_cols].values.tolist()

# float32 변환
merged["text_emb"]  = merged["text_emb"].apply(lambda x: np.array(x, dtype=np.float32))
merged["image_emb"] = merged["image_emb"].apply(lambda x: np.array(x, dtype=np.float32))

print("임베딩 변환 완료!")


# # 4. 정규화 함수 정의
def l2_normalize(v):
    return v / np.linalg.norm(v, axis=1, keepdims=True)


# # 5. FAISS 인덱스 구축 (Cosine 유사도 기반)
emb_matrix = np.vstack(merged["image_emb"].values)
emb_matrix = l2_normalize(emb_matrix)

dim = emb_matrix.shape[1]
index = faiss.IndexFlatIP(dim)   # Inner Product = Cosine similarity (정규화된 벡터 기준)
index.add(emb_matrix)

print(f"FAISS index (cosine) 구축 완료! (dim={dim}, size={index.ntotal})")


# # 6. 검색 함수 정의 (텍스트 → 이미지)
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def search_and_show(query_text, top_k=10):
    # 텍스트 → 임베딩
    inputs = processor(text=[query_text], return_tensors="pt", padding=True)
    with torch.no_grad():
        query_emb = model.get_text_features(**inputs).cpu().numpy().astype(np.float32)

    # 정규화
    query_emb = l2_normalize(query_emb)

    # FAISS 검색 (넉넉히 뽑고 중복 제거)
    D, I = index.search(query_emb, k=top_k*3)
    unique_results = []
    seen = set()
    for idx in I[0]:
        img_file = merged.iloc[idx]["image_file"]
        if img_file not in seen:
            seen.add(img_file)
            unique_results.append(merged.iloc[idx])
        if len(unique_results) == top_k:
            break

    results = pd.DataFrame(unique_results)[["image_file", "predicted_caption"]]

    # 동적 행/열 계산 (최대 5개씩 가로로 배치)
    cols = min(5, top_k)
    rows = math.ceil(top_k / cols)

    # 이미지 크기 조정 (3x3)
    plt.figure(figsize=(3*cols, 3*rows))

    for idx, img_file in enumerate(results["image_file"]):
        img_path = os.path.join(image_dir, img_file)
        try:
            img = Image.open(img_path).convert("RGB")
        except:
            img = Image.new("RGB", (224, 224), (200, 200, 200))

        ax = plt.subplot(rows, cols, idx + 1)
        ax.imshow(img)
        ax.axis("off")

    plt.suptitle(f"Query: {query_text}", fontsize=14)

    # 간격 조정 (wspace=가로, hspace=세로 여백)
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    plt.show()


# # 7. 테스트
search_and_show("blue, shirt, striped", top_k=5)

