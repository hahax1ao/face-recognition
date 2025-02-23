import base64
import io
import mediapipe as mp
import cv2
import gradio as gr
import numpy as np
from gradio_webrtc import AdditionalOutputs, WebRTC
from PIL import Image

from face_mesh import compare_faces_api

# 初始化 Mediapipe 人脸检测模型
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

def compare_faces():
    """返回最新的 WebRTC 采集帧，同时返回标注后的人脸图像"""
    if latest_frame is None:
        return None, None  # 避免空帧报错

    # 确保 `latest_frame` 是 numpy 数组
    if not isinstance(latest_frame, np.ndarray):
        return None, None

    match_result = compare_faces_api(latest_frame)
    return Image.fromarray(match_result)


def detection(image):
    global latest_frame
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 进行检测
    results = face_detection.process(rgb_image)
    # 画出人脸框
    if results.detections:
        latest_frame = image.copy()
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = image.shape
            x, y, w_box, h_box = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)

            # 计算圆的中心点和半径
            center_x = x + w_box // 2
            center_y = y + h_box // 2
            radius = max(w_box, h_box) // 2  # 取较大者，确保脸部被圈住

            # 绘制红色圆形人脸框
            cv2.circle(image, (center_x, center_y), radius, (0, 0, 255), 3)

    return image


css = """.my-group {max-width: 600px !important; max-height: 600 !important;}
                      .my-column {display: flex !important; justify-content: center !important; align-items: center !important};"""

with gr.Blocks(css=css) as iface:
    gr.HTML(
        """
    <h1 style='text-align: center'>实时人脸检测及识别</h1>
    """
    )
    with gr.Column(elem_classes=["my-column"]):
        with gr.Group(elem_classes=["my-group"]):
            image = WebRTC(
                label="Stream",
                #rtc_configuration=rtc_configuration,
                mode="send-receive",
                modality="video",
                track_constraints={
                    "width": {"exact": 800},
                    "height": {"exact": 600},
                    "aspectRatio": {"exact": 1.33333},
                },
                rtp_params={"degradationPreference": "maintain-resolution"},
            )
            image.stream(
                fn=detection, inputs=[image], outputs=[image], time_limit=90
            )
        with gr.Group(elem_classes=["my-group"]):
            # 普通上传图片检测
            btn = gr.Button("开始检测")
            result_img = gr.Image(label="检测结果", interactive=False)
            btn.click(fn=compare_faces, inputs=[], outputs=[result_img])

iface.launch()