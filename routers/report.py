import base64
from flask_socketio import SocketIO, emit
from flask import Blueprint, request, jsonify
from config import socketio
from servicers.report import report_shape,report_fitness,report_yoga,report_fitness_realtime,report_yoga_realtime

report = Blueprint('report', __name__)

@report.route('/shape',methods=['POST'])
def shape():
    data=request.form
    owner=data['userID']
    if 'file' not in request.files:
        return jsonify({
            'code': 1,
            'message': '图片部分'
        })
    photo=request.files['file']
    return report_shape(owner, photo,1)

@report.route('/fitness',methods=['POST'])
def fitness():
    data=request.form
    owner=data['userID']
    mID=data['modelID']
    if 'file' not in request.files:
        return jsonify({
            'code': 1,
            'message': '图片部分'
        })
    video=request.files['video']

    return report_fitness(owner, video,1)

@report.route('/yoga',methods=['POST'])
def yoga():
    data=request.form
    owner=data['userID']
    if 'file' not in request.files:
        return jsonify({
            'code': 1,
            'message': '图片部分'
        })
    video=request.files['video']
    return report_yoga(owner, video,1)

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