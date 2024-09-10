import os
import datetime

import cv2
from ultralytics import YOLO
# import cvzone
# import tensorflow as tf
# from tensorflow.keras.models import load_model
import numpy as np
import time  # Import time module for frame rate control

yoga_poses = [
    "Downward-Facing Dog Pose",
    "Downward-Facing Tree Pose",
    "Fire Log Pose",
    "Blissful Child Pose",
    "Endless Pose",
    "Anjaneyasana (Lunge Pose)",
    "Half Frog Pose",
    "Half Moon Pose",
    "Half Spinal Twist Pose",
    "Half Feathered Peacock Pose",
    "Half Forward Fold Pose",
    "Eight-Limbed Salutation Pose",
    "Eight-Angle Pose",
    "Bound Angle Pose",
    "Crane Pose",
    "Child Pose",
    "Fierce Pose",
    "Bharadvaja's Twist Pose I",
    "Frog Pose",
    "Cobra Pose",
    "Arm Pressure Pose",
    "Cat-Cow Pose",
    "Splendid Pose",
    "Wheel Pose",
    "Staff Pose",
    "Bow Pose",
    "Durvasasana (Sage Durvasa's Pose)",
    "Two-Legged Inverted Staff Pose",
    "One-Legged Koundinyasana I",
    "One-Legged Koundinyasana II",
    "One-Legged King Pigeon Pose",
    "One-Legged King Pigeon Pose II",
    "Ganda Bherundasana (Three-Legged Pose)",
    "Fetus Pose",
    "Garuda Pose",
    "Cow Face Pose",
    "Plow Pose",
    "Hanuman Pose",
    "Head-to-Knee Pose",
    "Pigeon Pose",
    "Heron Pose",
    "Turtle Pose",
    "Hanging Pose",
    "Makara Downward-Facing Dog Pose",
    "Crocodile Pose",
    "Garland Pose",
    "Marichi's Pose I",
    "Marichi's Pose III",
    "Cat Pose",
    "Fish Pose",
    "Peacock Pose",
    "Dancer's Pose",
    "Big Toe Pose",
    "Lotus Pose",
    "Gate Pose",
    "Full Boat Pose",
    "Revolved Head-to-Knee Pose",
    "Revolved Side Angle Pose",
    "Revolved Triangle Pose",
    "Side Crane Pose",
    "Intense Side Stretch Pose",
    "Noose Pose",
    "Seated Forward Bend Pose",
    "Plank Pose",
    "Feathered Peacock Pose",
    "Wide-Leg Forward Bend Pose",
    "Upward Plank Pose",
    "Locust Pose",
    "Supported Cobra Pose",
    "Supported Shoulder Stand Pose",
    "Supported Headstand Pose",
    "Corpse Pose",
    "Bridge Pose",
    "Lion Pose",
    "Easy Pose",
    "Reclining Bound Angle Pose",
    "Reclining Spinal Twist Pose",
    "Reclining Big Toe Pose",
    "Reclining Hero Pose",
    "Mountain Pose",
    "Firefly Pose",
    "Scale Pose",
    "Balance Pose",
    "Seated Wide Angle Pose",
    "Upward Bow Pose",
    "Upward Hands Pose",
    "Upward-Facing Dog Pose",
    "Upward Extended Foot Pose",
    "Camel Pose",
    "Chair Pose",
    "Extended Child's Pose",
    "Forward Fold Pose",
    "Extended Horse Pose",
    "Extended Hand-to-Big-Toe Pose",
    "Extended Side Angle Pose",
    "Extended Triangle Pose",
    "Thunderbolt Pose",
    "Side Plank Pose",
    "Legs-Up-the-Wall Pose",
    "Warrior I Pose",
    "Warrior II Pose",
    "Warrior III Pose",
    "Hero Pose",
    "Tree Pose",
    "Scorpion Pose",
    "Yogic Sleep Pose"
]

classNames = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
    "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet",
    "tvmonitor", "laptop", "mouse", "remote", "keyboard", "mobile phone", "microwave", "oven", "toaster",
    "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

import time
import cv2
from ultralytics import YOLO
import cvzone
import numpy as np
import tensorflow as tf


# 载入模型，只需一次
def load_yoga_model(model_path):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, kernel_size=(3, 3), padding='valid', activation='relu', input_shape=(64, 64, 3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='valid'),

        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), padding='valid', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='valid'),

        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), padding='valid', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='valid'),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(107, activation='softmax')
    ])
    model.load_weights(model_path)
    return model


def imgProcess(image, yoga_model):
    target_size = (64, 64)
    img = cv2.resize(image, target_size)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = yoga_model.predict(img)
    predicted_class_index = np.argmax(prediction)
    return predicted_class_index


def yogaPoseDetect(pose_mode, detect_model, video_path=0, video_save_dir=None, save_name=None):
    model = YOLO(pose_mode)
    yoga_model = load_yoga_model(detect_model)

    cap = cv2.VideoCapture(video_path)
    cap.set(3, 1280)
    cap.set(4, 720)

    # 目标帧率和时间控制
    desired_fps = 30
    frame_time = 1.0 / desired_fps
    last_frame_time = time.time()

    # For save result video
    output = None
    #if video_save_dir is not None:
    save_dir = os.path.join(video_save_dir, save_name)
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    # 编码器：“DIVX"、”MJPG"、“XVID”、“X264"; XVID MPEG4 Codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    output = cv2.VideoWriter(os.path.join(video_save_dir, save_name), fourcc, fps, size)


    while True:
        success, img = cap.read()
        if not success:
            break

        current_time = time.time()
        elapsed_time = current_time - last_frame_time

        if elapsed_time >= frame_time:
            # 执行YOLO推理
            results = model(img, stream=True)

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = round(float(box.conf[0]), 2)
                    id = int(box.cls[0])
                    class_name = classNames[id]

                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    if class_name == "person":
                        cropped_img = img[y1:y2, x1:x2]
                        predicted_pose = imgProcess(cropped_img, yoga_model)
                        cvzone.putTextRect(img, f'{yoga_poses[predicted_pose]}', (max(0, x1), max(40, y1)))

            cv2.imshow("Cam footage. Press 'Q' to exit.", img)
            if output is not None:
                output.write(img)
            last_frame_time = current_time

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if output is not None:
        output.release()
    cv2.destroyAllWindows()

def yogaPoseDetectByFrames(model # loaded yolo model
                           ,yoga_model  # loaded yoga-model.h5
                           ,frames # list of frames
                           ):
    # store result
    result_frames=[]
    while len(frames)!=0:
        img=frames[0]
        frames.pop(0)
        # 执行YOLO推理
        results = model(img)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = round(float(box.conf[0]), 2)
                id = int(box.cls[0])
                class_name = classNames[id]

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                if class_name == "person":
                    cropped_img = img[y1:y2, x1:x2]
                    predicted_pose = imgProcess(cropped_img,yoga_model)
                    cvzone.putTextRect(img, f'{yoga_poses[predicted_pose]}', (max(0, x1), max(40, y1)))
                    result_frames.append(img)
    return result_frames

# if __name__ == '__main__':
#     yogaPoseDetect(pose_mode='yolov8n.pt',  # 关键点检测模型
#                    detect_model='yoga-model.h5',  # 训练好的瑜伽评估模型
#                    video_path=0,  # 需要检测的视频文件路径，若为0，则打开摄像头
#                    video_save_dir='./results'  # 保存结果视频路径
#                    )






