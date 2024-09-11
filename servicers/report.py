import cv2
from flask import jsonify, send_file
from datetime import datetime
from moviepy.video.compositing.concatenate import concatenate_videoclips
import time
from moviepy.editor import VideoFileClip

from config import rep_file_path,temp_path
from config import s_state,s_counter
from daos.model import model_dao
from daos.reportData import report_dao
from moviepy.editor import ImageSequenceClip
import os
import numpy as np

from modelInference.image import image
from modelInference.counter import exercise_counter,exercise_counter_by_frames
from modelInference.yoga import yogaPoseDetect,yogaPoseDetectByFrames

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

    yogaPoseDetect('modelFile/yolov8n.pt','modelFile/yoga-model.h5',tp,temp_path,temp_name)
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
def create_video_from_frames(id,type):

    frames_path=os.path.join(temp_path,f'save_{id}.txt')
    # 打开帧缓存文件
    with open(frames_path, 'r+') as file:
        # 读取文件内容
        content = file.readlines()
        # 解析帧列表
    t_frames_dir = [line.strip().split(',') for line in content]

    frames = []
    with open(frames_path, 'r+') as file:
        # 读取文件内容
        content = file.readlines()

    for line in content:
        # 去除行尾的换行符
        frame_path = line.strip()
        f = np.load(frame_path)
        frames.append(f)

    clips = [ImageSequenceClip(m, fps=24).set_duration(1) for m in frames]
    final_clip = concatenate_videoclips(clips, method="compose")
    timestamp = datetime.utcnow().isoformat()
    filename=f'{timestamp}.mp4'
    filepath = os.path.join(rep_file_path, filename)
    final_clip.write_videofile(filepath, fps=24)

    #生成报告
    # 获取当前的日期和时间
    des=''
    now = datetime.now()
    # 只获取日期部分
    current_date = now.date()
    id = report_dao.add_report(id, type, des, filename, current_date)

    return filename

def report_fitness_realtime(frame_path, id,model,model_fitness,idx_2_category):
    t = time.time()

    frames = []
    f = cv2.imread(frame_path)
    #f = np.load(frame_path)
    #print(f)
    #f.show()

    frames.append(f)

    global s_state
    global s_counter
    save_json=exercise_counter_by_frames(model,model_fitness,
                                           frames,idx_2_category,s_counter,s_state)

    s_state=save_json.get('state')

    t_frames_name = f'save_{id}.txt'
    save_frames_path = os.path.join(temp_path, t_frames_name)

    # 打开帧缓存文件，如果文件不存在则创建一个新文件
    with open(save_frames_path, 'r+') as file:
        # 读取文件内容
        content = file.readlines()
        # 解析帧列表
    t_frames_dir = [line.strip().split(',') for line in content]

    save_frames=save_json.get('frames')

    for i, frame in enumerate(save_frames):
        f_path = os.path.join(temp_path, f'{t}_save_frame_{i}.npy')
        cv2.imwrite(f_path, frame)  # 保存每一帧
        t_frames_dir.append(f_path)  # 添加新帧

        # 写回修改后的内容
    with open(save_frames_path, 'w') as file:
        for line in frames:
            file.write(','.join(str(item) for item in line) + '\n')

    # frames = [line.strip().split(',') for line in content]

    return save_frames

def report_yoga_realtime(frame_path,id,model):
    t = time.time()

    frames = []
    f = cv2.imread(frame_path)
    # f = np.load(frame_path)
    frames.append(f)
    # frames = []
    # with open(frames_path, 'r+') as file:
    #     # 读取文件内容
    #     content = file.readlines()
    #
    #     for line in content:
    #         # 去除行尾的换行符
    #         frame_path = line.strip()
    #         f = np.load(frame_path)
    #         frames.append(f)

    save_frames = yogaPoseDetectByFrames(model, 'modelFile/yoga-model.h5', frames)


    t_frames_name=f'save_{id}.txt'
    save_frames_path=os.path.join(temp_path,t_frames_name)

    #打开帧缓存文件，如果文件不存在则创建一个新文件
    with open(save_frames_path, 'r+') as file:
        # 读取文件内容
        content = file.readlines()
        # 解析帧列表
    t_frames_dir = [line.strip().split(',') for line in content]

    for i, frame in enumerate(save_frames):
        f_path = os.path.join(temp_path, f'{t}_save_frames_{i}.npy')
        np.save(f_path, frame)#保存每一帧
        t_frames_dir.append(f_path)#添加新帧

        # 写回修改后的内容
    with open(save_frames_path, 'w') as file:
        for line in frames:
            file.write(','.join(str(item) for item in line) + '\n')

    # frames = [line.strip().split(',') for line in content]
    return save_frames

