import os
import torch
from torch.autograd import Variable
from net import NetWork
import numpy as np
import cv2
import time



def load_model(path):

    # =============================================
    net_model = NetWork(1, 1)
    # =============================================

    net_model.load_state_dict(torch.load(path))
    para = sum([np.prod(list(p.size())) for p in net_model.parameters()])
    type_size = 4
    print('Model {} : params: {:4f}M'.format(net_model._get_name(), para * type_size / 1000 / 1000))
    net_model.eval()
    net_model.cuda()
    return net_model


# 调用模型
def generate_fusion_image(model, img1, img2):

    fuse_image = model.forward(img1, img2)
    return fuse_image[0]


def main():
    start = time.time()
    # run demo
    test_path = './IV_images/'
    output_path = './IV_images/'
    model_path = './model/final_model.model'

    #test_path = './out/'
    #output_path = './out/outputs/'
    #model_path = './model/.model'
    with torch.no_grad():
        print('=======begin========')

        #  加载模型
        # =======================================
        model = load_model(model_path)
        # =======================================

        for i in range(1, 11):

            index = i
            infrared_path = test_path + 'IR/' + 'IR'+ str(index) + '.png'
            visible_path  = test_path + 'VIS/'+ 'VIS'+ str(index) + '.png'
            run_demo(model, infrared_path, visible_path, output_path, index)
    print('Done......')
    finish = time.time()
    total = finish - start
    print(total)


def run_demo(model, infrared_path, visible_path, output_path_root, index):
    ir_img = get_test_images(infrared_path)
    vis_img = get_test_images(visible_path)


    # 数据转移到 GPU
    ir_img = ir_img.cuda()
    vis_img = vis_img.cuda()

    # 创建变量
    ir_img = Variable(ir_img, requires_grad=False)
    vis_img = Variable(vis_img, requires_grad=False)

    # 产生融合图像
    img_fusion = generate_fusion_image(model, ir_img, vis_img)
    ############################ multi outputs ##############################################
    file_name = 'fusion_' + str(index) +  '.png'
    output_path = output_path_root + file_name
    img = img_fusion.cpu().clamp(0, 255).data[0].numpy()
    cv2.imwrite(output_path, img)

    print(output_path)



def get_test_images(paths):

    image = cv2.imread(paths, cv2.IMREAD_GRAYSCALE)
    image = np.reshape(image, [1, 1, image.shape[0], image.shape[1]])
    image = np.array(image)
    image = torch.from_numpy(image).float()
    return image


if __name__ == '__main__':
    main()
