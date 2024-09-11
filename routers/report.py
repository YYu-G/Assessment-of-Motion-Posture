import base64
import os
import io

import cv2
import numpy as np
import torch

from PIL import Image
from tensorflow.python.ops.signal.shape_ops import frame
from ultralytics import YOLO
from Inference import LSTM
from config import temp_path,s_state,executor
from datetime import datetime
import time
from flask_socketio import SocketIO, emit
from flask import Blueprint, request, jsonify, json
from config import socketio,INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, OUTPUT_DIM
from modelInference.yoga import load_yoga_model
from servicers.report import (report_shape,report_fitness,report_yoga,
                              report_fitness_realtime,report_yoga_realtime,create_video_from_frames)

report = Blueprint('report', __name__)


# Load the YOLOv8 model of yoga
model = YOLO('modelFile/yolov8n.pt')
#load yoga model
yoga_model=load_yoga_model('modelFile/yoga-model.h5')


#load yolo of exercises
ex_model=YOLO('modelFile/yolov8s-pose.pt')
# Load exersice model

with open(os.path.join('modelFile', 'idx_2_category.json'), 'r') as f:
    idx_2_category = json.load(f)
detect_model = LSTM(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, OUTPUT_DIM)
model_path = os.path.join('modelFile', 'best_model2.pt')
model_weight = torch.load(model_path, map_location=torch.device('cpu'))
# model_weight = torch.load(model_path)
detect_model.load_state_dict(model_weight)

yoga_model=load_yoga_model('modelFile/yoga-model.h5')

@report.route('/shape',methods=['POST'])
def shape():
    data=request.form
    owner=data['userID']
    if 'file1' not in request.files:
        return jsonify({
            'code': 1,
            'message': '图片部分'
        })
    if 'file2' not in request.files:
        return jsonify({
            'code': 1,
            'message': '图片部分'
        })
    photo1=request.files['file1']
    photo2=request.files['file2']

    timestamp = time.time()
    temp_name1 = f"{timestamp}_{photo1.filename}"
    temp_name2 = f"{timestamp}_{photo2.filename}"

    tp1 = os.path.join(temp_path, temp_name1)
    tp2 = os.path.join(temp_path, temp_name2)
    photo1.save(dst=tp1)#保存图片
    photo2.save(dst=tp2)
    return report_shape(owner, temp_name1,temp_name2,tp1,tp2,1)

@report.route('/fitness',methods=['POST'])
def fitness():
    data=request.form
    owner=data['userID']
    #mID=data['modelID']
    if 'video' not in request.files:
        return jsonify({
            'code': 1,
            'message': '视频部分'
        })
    video=request.files['video']
    timestamp=time.time()
    temp_name=f'{timestamp}_{video.filename}'
    tp=os.path.join(temp_path,temp_name)
    print('tp')
    video.save(tp)
    return report_fitness(owner,video.filename,tp,1)

@report.route('/yoga',methods=['POST'])
def yoga():
    data=request.form
    owner=data['userID']
    if 'video' not in request.files:
        return jsonify({
            'code': 1,
            'message': '视频部分'
        })
    video=request.files['video']
    timestamp = time.time()
    temp_name = f'{timestamp}_{video.filename}'
    tp = os.path.join(temp_path, temp_name)
    video.save(tp)
    return report_yoga(owner,video.filename,tp,1)

@socketio.on('fitness')
def fitness_realtime(data):

    frame=np.frombuffer(data,np.uint8)
    im=cv2.imdecode(frame,cv2.IMREAD_COLOR)
    userID=1

    t=time.time()

    frames_name = f'{t}.txt'
    frames_path=os.path.join(temp_path,frames_name)

    #frames=[]#帧列表
    frame_path = os.path.join(temp_path, f'{t}_frames.jpg')
    cv2.imwrite(frame_path, im)  # 保存每一帧
    #frames.append(frame_path)

    # with open(frames_path, 'w') as file:
    #     for line in frames:
    #         file.write(','.join(str(item) for item in line) + '\n')

    #future = executor.submit(report_fitness_realtime, data)

    result=report_fitness_realtime(frame_path,userID,ex_model,detect_model,idx_2_category)

    socketio.emit('response',result)
    return 0

@socketio.on('yoga')
def yoga_realtime(data):

    frame = np.frombuffer(data, np.uint8)
    im = cv2.imdecode(frame, cv2.IMREAD_COLOR)
    userID=1
    type='yoga'

    t=time.time()

    frames_name = f'{t}.jpg'
    frames_path=os.path.join(temp_path,frames_name)

    frames=[]#帧列表
    frame_path = os.path.join(temp_path, f'{t}_frames.jpg')
    cv2.imwrite(frame_path, im)  # 保存每一帧

    result= report_yoga_realtime(frame_path,userID,model,yoga_model)

    socketio.emit('response', result)
    return 0




# @report.route('/fit_mess')
# @socketio.on('fitness')
# def handle_fitness_message(frame_data,id):#Base64 编码的字符串/二进制数据
#     # 解码 base64 数据
#     image_bytes = base64.b64decode(frame_data.split(',')[1])
#     report_fitness_realtime(image_bytes,id)
#
#     operation = {
#         # report_fitness_realtime()
#     }
#
#     # 生成 base64 编码的字符串
#     processed_frame_base64 = base64.b64encode(processed_frame).decode('utf-8')
#     # 发送处理后的帧数据到前端
#     emit('processed_frame', {
#         'data': 'data:image/jpeg;base64,' + processed_frame_base64,
#         'client_id': id
#     }, to=id)
#     return
#
# # @report.route('/yoga_mess')
# @socketio.on('yoga')
# def handle_yoga_message(frame_data,id):#Base64 编码的字符串/二进制数据
#     # 解码 base64 数据
#     image_bytes = base64.b64decode(frame_data.split(',')[1])
#     report_yoga_realtime(image_bytes,id)
#
#     operation={
#         # report_yoga_realtime()
#     }
#
#     # 生成 base64 编码的字符串
#     processed_frame_base64 = base64.b64encode(processed_frame).decode('utf-8')
#     # 发送处理后的帧数据到前端
#     emit('processed_frame', {
#         'data': 'data:image/jpeg;base64,' + processed_frame_base64,
#         'client_id': id
#     }, to=id)
#         # print('received message: ' + data)
#     # # 向客户端发送一个回复
#     # socketio.emit('response', {'data': 'Message received!'})
#     return

