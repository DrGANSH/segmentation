import time
import torch
import cv2
import numpy as np
from torch import nn
from nets.build_BiSeNet import BiSeNet
from nets.bisenetv2 import BiSeNetV2
import torch.nn.functional as F


def get_FPS(image, test_interval):
    # ---------------------------------------------------------#
    #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
    #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
    # ---------------------------------------------------------#

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    orininal_h = np.array(image).shape[0]
    orininal_w = np.array(image).shape[1]
    # ---------------------------------------------------------#
    #   给图像增加灰条，实现不失真的resize
    #   也可以直接resize进行识别
    # ---------------------------------------------------------#
    # image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))
    input_shape = [128, 128]
    image_data = cv2.resize(image, (input_shape[1], input_shape[0]), cv2.INTER_AREA)
    image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)
    # ---------------------------------------------------------#
    #   添加上batch_size维度
    # ---------------------------------------------------------#
    # image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

    with torch.no_grad():
        images = torch.from_numpy(image_data)
        model_path = 'logs/best_epoch_weights.pth'

        images = images.cuda()
        # net = BiSeNet(num_classes=2, context_path='resnet18')
        net = BiSeNetV2(n_classes=2)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net.load_state_dict(torch.load(model_path, map_location=device))
        net = net.eval()
        net = nn.DataParallel(net)
        net = net.cuda()

        # ---------------------------------------------------#
        #   图片传入网络进行预测
        # ---------------------------------------------------#
        pr = net(images)[0]
        # ---------------------------------------------------#
        #   取出每一个像素点的种类
        # ---------------------------------------------------#
        pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()
        # --------------------------------------#
        #   将灰条部分截取掉
        # --------------------------------------#

        # ---------------------------------------------------#
        #   进行图片的resize
        # ---------------------------------------------------#
        pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)
        pr = pr.argmax(axis=-1)
    t1 = time.time()
    for _ in range(test_interval):
        with torch.no_grad():
            # ---------------------------------------------------#
            #   图片传入网络进行预测
            # ---------------------------------------------------#
            pr = net(images)[0]
            # ---------------------------------------------------#
            #   取出每一个像素点的种类
            # ---------------------------------------------------#
            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()
            # --------------------------------------#
            #   将灰条部分截取掉
            # --------------------------------------#

            # ---------------------------------------------------#
            #   进行图片的resize
            # ---------------------------------------------------#
            pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)
            # ---------------------------------------------------#
            #   取出每一个像素点的种类
            # ---------------------------------------------------#
            pr = pr.argmax(axis=-1)
    t2 = time.time()
    tact_time = (t2 - t1) / test_interval
    return tact_time


def preprocess_input(image):
    image /= 255.0
    return image


if __name__ == '__main__':
    img="15.jpg"
    # t1=time.time()
    image=cv2.imread(img)
    test_interval=1000
    tact_time = get_FPS(image, test_interval)
    print(str(tact_time) + ' seconds, ' + str(1 / tact_time) + 'FPS, @batch_size 1')

    # t2=time.time()
    # FPS=1.0/(t2-t1)
    # print("FPS",FPS)
    # cv2.imshow("2",r_image)
    # cv2.waitKey(0)