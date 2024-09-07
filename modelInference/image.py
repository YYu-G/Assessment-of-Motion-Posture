from ultralytics import YOLO
import os
import shutil  # 用于移动文件


def shape_image(model_path,image_path,save_path,text_path):
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # pt文件路径
    pth_path = os.path.join(current_dir, model_path)#'runs/pose/train/weights/best.pt')

    # 加载模型
    pose_model = YOLO(pth_path)

    # 测试图像路径
    image_path = image_path#'./test_imgs/001.jpg'


    # 执行预测并保存图像到默认目录
    results = pose_model.predict(save_path, imgsz=600, save=True)
    save_dir = results[0].save_dir

    # 将预测结果保存到 txt 文件
    #results_file_path = os.path.join(text_path, 'results.txt')

    with open(text_path, 'w') as f:
        for result in results:
            f.write(f"Image Path: {result.path}\n")
            f.write(f"Original Shape: {result.orig_shape}\n")
            f.write(f"Detection Boxes: {result.boxes}\n")
            f.write(f"Keypoints: {result.keypoints}\n")
            f.write("\n")

