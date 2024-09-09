import image


if __name__ == '__main__':
    #模型
    model_path = '../modelFile/yolov8s-pose.pt'
    # 测试图像路径
    image_path_front = './test_imgs/001.jpg'  # 正面
    image_path_side = './test_imgs/003.png'  # 侧面

    degree_list,dir=image.image(model_path,image_path_front,image_path_side)

    def print_body_feature(degree_list):
        shoulder_degree = degree_list[0]
        leg_degree = degree_list[1]
        back_degree = degree_list[2]
        neck_degree = degree_list[3]

        if shoulder_degree <= 5:
            print('正常肩高')
        elif 5 <= shoulder_degree <= 10:
            print('轻度高低肩')
        elif shoulder_degree > 10:
            print('高低肩')

        if leg_degree < 9:
            print('O型腿')
        elif 9 <= leg_degree <= 18:
            print('正常腿型')
        elif leg_degree > 18:
            print('X型腿')

        if back_degree < 20:
            print('无驼背症状')
        elif back_degree >= 20:
            print('驼背')

        if neck_degree < 30:
            print('无头前倾症状')
        elif 30 <= neck_degree < 40:
            print('轻微头前倾')
        elif neck_degree >= 40:
            print('头前倾')

    print(degree_list)
    print_body_feature(degree_list)







