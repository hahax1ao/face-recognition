import os

import cv2
import numpy as np
import mediapipe as mp
import base64

from tqdm import tqdm
mp_face_mesh = mp.solutions.face_mesh

# 加载 MediaPipe 人脸网格模型
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)


def extract_face_embedding(image):
    """ 提取人脸关键点，并计算特征向量 """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    if not results.multi_face_landmarks:
        return None  # 未检测到人脸

    # 选取第一个检测到的人脸
    face_landmarks = results.multi_face_landmarks[0]

    # 提取 468 个关键点的 (x, y, z) 坐标作为特征
    face_embedding = np.array([[landmark.x, landmark.y, landmark.z] for landmark in face_landmarks.landmark]).flatten()

    return face_embedding


def cosine_similarity(vec1, vec2):
    """ 计算两个特征向量的余弦相似度 """
    #return 1 - cosine(vec1, vec2)
    vec1 = np.asarray(vec1)
    vec2 = np.asarray(vec2)

    # 计算点积
    dot_product = np.dot(vec1, vec2)

    # 计算 L2 范数（模长）
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    # 避免除零
    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def find_best_match(input_image, known_faces):
    """
    1:N 人脸比对
    input_image: 用户上传的图片
    known_faces: 已知人脸数据库，格式 {"name": "张三", "embedding": np.array([...])}
    """
    input_embedding = extract_face_embedding(input_image)
    if input_embedding is None:
        return None, 0  # 没有检测到人脸

    best_match = None
    best_similarity = 0

    for face in tqdm(known_faces):
        similarity = cosine_similarity(input_embedding, face["embedding"])
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = face

    return best_match, best_similarity


# 加载已知人脸数据库
def load_known_faces(folder_path="faces"):
    """ 遍历 `faces/` 目录加载所有 `.jpeg` 人脸库，并显示进度条 """
    known_faces = []
    image_files = [f for f in os.listdir(folder_path) if f.endswith(".jpeg") or f.endswith(".jpg")]

    print(f"检测到 {len(image_files)} 张人脸图片，开始加载...")

    for filename in tqdm(image_files, desc="加载人脸库", unit="img"):
        name = os.path.splitext(filename)[0]  # 去掉扩展名作为名字
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)

        # 提取人脸特征
        embedding = extract_face_embedding(image)
        if embedding is not None:
            known_faces.append({"name": name, "embedding": embedding, "image": image_path})
        else:
            print(f"无法提取人脸: {name} ({image_path})")

    print(f"人脸库加载完成，共 {len(known_faces)} 张有效人脸")
    return known_faces

# 预加载人脸库
known_faces_db = load_known_faces()

def compare_faces_api(input_image):
    best_match, similarity = find_best_match(input_image, known_faces_db)

    if best_match:
        # 读取匹配图片并返回 NumPy 数组
        match_image = cv2.imread(best_match["image"])  # 读取匹配图片
        match_image = cv2.cvtColor(match_image, cv2.COLOR_BGR2RGB)  # 转换为 RGB 格式

        return match_image  # 直接返回 NumPy 图片数组

    else:
        # 返回一张空白图片（防止错误）
        return np.zeros((100, 100, 3), dtype=np.uint8)
