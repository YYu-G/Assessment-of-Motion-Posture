import base64
import os

from tensorflow import timestamp

from config import temp_path
from datetime import datetime
import time
from flask_socketio import SocketIO, emit
from flask import Blueprint, request, jsonify
from config import socketio
from servicers.report import report_shape,report_fitness,report_yoga,report_fitness_realtime,report_yoga_realtime

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

    return report_yoga(owner,video.filename,tp,1)

# @report.route('/fit_mess')
@socketio.on('fitness')
def handle_fitness_message(frame_data,id):#Base64 编码的字符串/二进制数据
    # 解码 base64 数据
    image_bytes = base64.b64decode(frame_data.split(',')[1])
    report_fitness_realtime(image_bytes,id)

    operation = {
        # report_fitness_realtime()
    }

    # 生成 base64 编码的字符串
    processed_frame_base64 = base64.b64encode(processed_frame).decode('utf-8')
    # 发送处理后的帧数据到前端
    emit('processed_frame', {
        'data': 'data:image/jpeg;base64,' + processed_frame_base64,
        'client_id': id
    }, to=id)
    return

# @report.route('/yoga_mess')
@socketio.on('yoga')
def handle_yoga_message(frame_data,id):#Base64 编码的字符串/二进制数据
    # 解码 base64 数据
    image_bytes = base64.b64decode(frame_data.split(',')[1])
    report_yoga_realtime(image_bytes,id)

    operation={
        # report_yoga_realtime()
    }

    # 生成 base64 编码的字符串
    processed_frame_base64 = base64.b64encode(processed_frame).decode('utf-8')
    # 发送处理后的帧数据到前端
    emit('processed_frame', {
        'data': 'data:image/jpeg;base64,' + processed_frame_base64,
        'client_id': id
    }, to=id)
        # print('received message: ' + data)
    # # 向客户端发送一个回复
    # socketio.emit('response', {'data': 'Message received!'})
    return