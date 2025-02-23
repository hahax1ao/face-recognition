import cv2
import mediapipe as mp

# 初始化 Mediapipe 人脸检测模型
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# 读取图像
image = cv2.imread("face.png")
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 进行检测
results = face_detection.process(rgb_image)

# 画出人脸框
if results.detections:
    for detection in results.detections:
        bboxC = detection.location_data.relative_bounding_box
        h, w, _ = image.shape
        x, y, w_box, h_box = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
        cv2.rectangle(image, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)

cv2.imshow("Face Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
