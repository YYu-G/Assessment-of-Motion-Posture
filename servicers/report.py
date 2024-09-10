from flask import jsonify, send_file
from datetime import datetime
from moviepy.video.compositing.concatenate import concatenate_videoclips
import time
from moviepy.editor import VideoFileClip

from config import rep_file_path,temp_path
from daos.model import model_dao
from daos.reportData import report_dao
from moviepy.editor import ImageSequenceClip
import os


from modelInference.image import image
from modelInference.counter import exercise_counter
from modelInference.yoga import yogaPoseDetect


import shutil

def report_shape(ownerID,photo_name1,photo_name2,tp1,tp2,model_id):

    des='0'

    # model=model_dao.search_model_by_id(model_id)
    # model_path=os.path.join(model_file_path,model.modelFileURL)#获得模型路径

    degree_list,dir=image('../modelFile/yolov8s-pose',tp1,tp2)#生成分析图片

    t=time.time()
    name1=f'{t}_{photo_name1}'
    t = time.time()
    name2=f'{t}_{photo_name2}'
    #保存路径
    save_path1=os.path.join(dir,photo_name1)
    save_path2 = os.path.join(dir, photo_name2)
    rep_path1=os.path.join(rep_file_path,name1)
    rep_path2=os.path.join(rep_file_path,name2)

    shutil.copyfile(save_path1,rep_path1)
    shutil.copyfile(save_path2,rep_path2)

    # 获取当前的日期和时间
    now = datetime.now()
    # 只获取日期部分
    current_date = now.date()

    id=report_dao.add_report(ownerID,'shape',des,name1+';'+name2,current_date)

    # 返回处理后的图片路径
    return jsonify({
        'urls': [
            f'/api/download/{name1}',
            f'/api/download/{name2}'
        ]
    })

    # send_from_directory(os.path.dirname(save_path1), os.path.basename(save_path1))
    # return send_from_directory(os.path.dirname(save_path2), os.path.basename(save_path2))

    # return jsonify({
    #     'code':0,
    #     'message':'生成成功',
    #     'data':id
    # })

def report_fitness(ownerID,video_name,tp,model_id):

    des=''

    timestamp = time.time()
    temp_name = f'{timestamp}_{video_name}'
    t_path = os.path.join(temp_path, temp_name)
    save_name=f'{timestamp}_{video_name}'
    save_path=os.path.join(rep_file_path,save_name)

    model_name=model_dao.search_model_by_id(model_id)


    #'../yolov8s-pose.pt'
    exercise_counter(pose_model='modelFile/yolov8s-pose.pt',  # pose模型
                    detector_model_path='modelFile',  # 训练完的检测姿态模型路径
                    detector_model_file='best_model2.pt',#模型名称
                    video_file=tp,  # 视频文件
                    video_save_dir=temp_path,
                     video_save_name=temp_name) # 视频保存路径)

    # 使用moviepy转换视频
    clip = VideoFileClip(t_path)
    clip.write_videofile(save_path, codec='libx264')

    # 清理上传的临时文件
    clip.close()
    #os.remove(temp_path)

    # 获取当前的日期和时间
    now = datetime.now()
    # 只获取日期部分
    current_date = now.date()
    id = report_dao.add_report(ownerID, 'fitness', des, save_name, current_date)

    #return send_file(save_path)
    return jsonify({
       'url':f'/api/download/{save_name}'})
    #return send_file(save_path,mimetype='video/mp4')

def report_yoga(owner,video_name,tp,model_id):

    des=''

    #model_name=model_dao.search_model_by_id(id)
    #model_path=os.path.join(model_file_path,model_name)

    timestamp = time.time()
    temp_name = f'{timestamp}_{video_name}'
    t_path = os.path.join(temp_path, temp_name)
    save_name=f"{timestamp}_{video_name}"
    save_path=os.path.join(rep_file_path,save_name)

    yogaPoseDetect('modelFile/yolov8n.pt','modelFile/yoga-model.h5',tp,t_path)
    #yogaPoseDetect('modelFile/yolov8n.pt', 'modelFile/yoga-model.h5', 0, save_path)

    # 使用moviepy转换视频
    clip = VideoFileClip(t_path)
    clip.write_videofile(save_path, codec='libx264')

    # 清理上传的临时文件
    clip.close()
    # os.remove(temp_path)

    # 获取当前的日期和时间
    now = datetime.now()
    # 只获取日期部分
    current_date = now.date()
    id = report_dao.add_report(owner, 'yoga', des, save_name, current_date)

    return jsonify({
        'url':
            f'/api/download/{save_name}'
    })

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