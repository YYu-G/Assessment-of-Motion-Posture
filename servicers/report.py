from flask import jsonify,send_file
from datetime import datetime
from moviepy.video.compositing.concatenate import concatenate_videoclips

from config import rep_file_path,temp_path,model_file_path
from daos.model import model_dao
from daos.reportData import report_dao
from moviepy.editor import ImageSequenceClip
import os

from modelInference.counter import exercise_counter
from modelInference.image import shape_image
from modelInference.yoga import yogaPoseDetect

def report_shape(ownerID,photo,model_id):

    des={

    }

    timestamp = datetime.utcnow().isoformat()

    temp_name = f"{timestamp}_{photo.filename}"
    tp=os.path.join(temp_path, temp_name)
    photo.save(tp)#保存上传图片

    model=model_dao.search_model_by_id(model_id)
    model_path=os.path.join(model_file_path,model.modelFileURL)#获得模型路径

    #保存路径
    save_name = f"{timestamp}_{photo.filename}"
    save_path=os.path.join(rep_file_path,save_name)
    txt_name=f"{timestamp}_{ownerID}.txt"
    txt_path=os.path.join(rep_file_path,txt_name)

    shape_image('code/runs/pose/train/weights/best.pt',tp,save_path,txt_path)#生成分析图片

    # 获取当前的日期和时间
    now = datetime.now()
    # 只获取日期部分
    current_date = now.date()

    id=report_dao.add_report(ownerID,'shape',des,save_name,current_date)

    return send_file(save_path, mimetype='image/jpeg')
    # return jsonify({
    #     'code':0,
    #     'message':'生成成功',
    #     'data':id
    # })

def report_fitness(ownerID,video,model_id):

    des={

    }

    timestamp = datetime.utcnow().isoformat()
    file_name = f"{timestamp}_{video.filename}"
    fp=os.path.join(rep_file_path, file_name)
    tp=os.path.join(temp_path,file_name)
    video.save(tp)#保存上传的视频
    model_name=model_dao.search_model_by_id(model_id)

    #'../yolov8s-pose.pt'
    exercise_counter(pose_model='modelFile/yolov8s-pose.pt',  # pose模型
                    detector_model_path='modelFile',  # 训练完的检测姿态模型路径
                    detector_model_file='1.pt',#模型名称
                    video_file=tp,  # 视频文件
                    video_save_dir=fp) # 视频保存路径)

    # 获取当前的日期和时间
    now = datetime.now()
    # 只获取日期部分
    current_date = now.date()
    id = report_dao.add_report(ownerID, 'fitness', des, file_name, current_date)
    # 创建 HTTP 响应

    # 设置响应头
    # response.headers['Content-Type'] = 'video/mp4'

    # 发送响应
    return send_file(fp,mimetype='video/mp4')
    # return jsonify({
    #     'code': 0,
    #     'message': '生成成功',
    #     'data': id
    # })


def report_yoga(owner,video,model_id):

    des={

    }

    timestamp = datetime.utcnow().isoformat()
    video_name = f"{timestamp}_{video.filename}"
    video_path=os.path.join(temp_path, video_name)
    video.save(video_path)

    model_name=model_dao.search_model_by_id(id)
    model_path=os.path.join(model_file_path,model_name)

    timestamp2 = datetime.utcnow().isoformat()
    file_name=f"{timestamp2}_{video.filename}"
    file_path=os.path.join(rep_file_path,file_name)

    yogaPoseDetect('modelFile/yolov8n.pt','modelFile/yoga-model.h5',video_path,file_path)

    # 获取当前的日期和时间
    now = datetime.now()
    # 只获取日期部分
    current_date = now.date()
    id = report_dao.add_report(owner, 'yoga', des, file_name, current_date)

    return send_file(file_path, mimetype='video/mp4')
    # return jsonify({
    #     'code': 0,
    #     'message': '生成成功',
    #     'data': id
    # })

# 一个函数来生成视频文件并保存
def create_video_from_frames(frames):
    clips = [ImageSequenceClip(m, fps=24).set_duration(1) for m in frames]
    final_clip = concatenate_videoclips(clips, method="compose")
    timestamp = datetime.utcnow().isoformat()
    filename=f'{timestamp}.mp4'
    filepath = os.path.join(rep_file_path, filename)
    final_clip.write_videofile(filepath, fps=24)
    return filename

def report_fitness_realtime(image_bytes,id):
    timestamp = datetime.utcnow().isoformat()

    tempname = f'{timestamp}_{id}.jpg'
    temppath = os.path.join(temp_path, tempname)

    filename=f'{timestamp}_{id}.jpg'
    filepath=os.path.join(temp_path,filename)

    with open(temppath, 'wb') as f:#保存上传帧
        f.write(image_bytes)

    op = {

    }

    with open(filepath, 'wb') as f:  # 保存处理后的帧
        f.write()

    # 打开帧缓存文件，如果文件不存在则创建一个新文件
    frames_name=f'{id}.txt'
    frames_path=os.path.join(temp_path,frames_name)
    with open(frames_path, 'r+') as file:
        # 读取文件内容
        content = file.readlines()
    # 解析帧列表
    frames = [line.strip().split(',') for line in content]
    frames.append(filepath)#添加新帧
    # 写回修改后的内容
    with open(frames_path, 'w') as file:
        for line in frames:
            file.write(','.join(str(item) for item in line) + '\n')

    return {

    }

def report_yoga_realtime(image_bytes,id):
    timestamp = datetime.utcnow().isoformat()

    tempname = f'{timestamp}_{id}.jpg'
    temppath = os.path.join(temp_path, tempname)

    filename = f'{timestamp}_{id}.jpg'
    filepath = os.path.join(temp_path, filename)

    with open(temppath, 'wb') as f:#保存上传帧
        f.write(image_bytes)

    op={

    }

    with open(filepath, 'wb') as f:#保存处理后的帧
        f.write()

    # 打开帧缓存文件，如果文件不存在则创建一个新文件
    frames_name = f'{id}.txt'
    frames_path = os.path.join(temp_path, frames_name)
    with open(frames_path, 'r+') as file:
        # 读取文件内容
        content = file.readlines()
    # 解析帧列表
    frames = [line.strip().split(',') for line in content]
    frames.append(filepath)  # 添加新帧
    # 写回修改后的内容
    with open(frames_path, 'w') as file:
        for line in frames:
            file.write(','.join(str(item) for item in line) + '\n')

    return {

    }