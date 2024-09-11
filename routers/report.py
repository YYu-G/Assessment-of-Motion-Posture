import base64
import os
import io

import cv2
import numpy as np

from PIL import Image
from tensorflow.python.ops.signal.shape_ops import frame

from config import temp_path,s_state
from datetime import datetime
import time
from flask_socketio import SocketIO, emit
from flask import Blueprint, request, jsonify
from config import socketio
from servicers.report import (report_shape,report_fitness,report_yoga,
                              report_fitness_realtime,report_yoga_realtime,create_video_from_frames)

report = Blueprint('report', __name__)

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
    #frames_data = data['frames']
    # 将Base64字符串解码为图像
    # image_data = base64.b64decode(data.split(',')[1])
    # image = Image.open(io.BytesIO(image_data))
    #
    # # 将PIL图像转换为numpy数组
    # frame = np.array(image)
    frame=np.frombuffer(data,np.uint8)
    im=cv2.imdecode(frame,cv2.IMREAD_COLOR)
    userID=1

    # userID=data['id']
    # type=data['type']
    # state=data['state']

    # if frame==None:
    #     print('video')
    #     return create_video_from_frames(id,'fitness')

    t=time.time()

    frames_name = f'{t}.txt'
    frames_path=os.path.join(temp_path,frames_name)

    frames=[]#帧列表
    frame_path = os.path.join(temp_path, f'{t}_frames.png')
    cv2.imwrite(frame_path, im)  # 保存每一帧
    frames.append(frame_path)
    # for i, frame in enumerate(frames_data):
    #     frame_path = os.path.join(temp_path, f'{t}_frames_{i}.npy')
    #     np.save(frame_path, frame)#保存每一帧
    #     frames.append(frame_path)

    with open(frames_path, 'w') as file:
        for line in frames:
            file.write(','.join(str(item) for item in line) + '\n')

    return report_fitness_realtime(frame_path,userID,s_state)


@socketio.on('yoga')
def fitness_realtime(data):
    frames_data = data['frames']
    userID=data['id']
    type=data['type']
    if frames_data:
        return create_video_from_frames(id,type)
    else:
        t=time.time()

        frames_name = f'{t}.txt'
        frames_path=os.path.join(temp_path,frames_name)

        frames=[]#帧列表
        for i, frame in enumerate(frames_data):
            frame_path = os.path.join(temp_path, f'{t}_frames_{i}.npy')
            np.save(frame_path, frame)#保存每一帧
            frames.append(frame_path)

        with open(frames_path, 'w') as file:
            for line in frames:
                file.write(','.join(str(item) for item in line) + '\n')

        return report_yoga_realtime(frames_path,userID)


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

