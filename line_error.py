import os
from os.path import join
import cv2
import numpy as np
"""评价导航线误差，pytorch和tensorrt一样使用，先用IOU的代码获取预测图像
一定要用opencv4.4（用的环境pytorch2），其他用环境pytorch7
"""
def top(trg1):
    a=trg1[0][0]
    b = trg1[1][0]
    c = trg1[2][0]
    if a[1]<b[1] and a[1]<c[1]:
        x = (b[0] + c[0]) // 2
        y = (b[1] + c[1]) // 2
        # cv2.line(imgor, (int(x), int(y)), (int(a[0]),int(a[1])), (0, 255, 255), 7)
        e=tuple(a)
        f = (int(x), int(y))
    elif c[1]<a[1] and b[1]>c[1]:
        x = (b[0] + a[0]) // 2
        y = (b[1] + a[1]) // 2
        # cv2.line(imgor, (int(x), int(y)),(int(c[0]),int(c[1])), (0, 255, 255), 7)
        e = tuple(c)
        f = (int(x), int(y))
    else:
        x = (a[0] + c[0]) // 2
        y = (a[1] + c[1]) // 2
        # cv2.line(imgor, (int(x), int(y)), (int(b[0]),int(b[1])), (0, 255, 255), 7)
        e = tuple(b)
        f = (int(x), int(y))
    return f,e

def calcu(point):
    y_1 = 600
    y_2 = 450
    x_n=point[0][0]-point[1][0]
    y_n=point[0][1]-point[1][1]
    k=y_n/x_n
    b=point[0][1]-k*point[0][0]
    x_1=(y_1-b)/k
    x_2 = (y_2 - b) / k
    # f_p=(int(x_1),y_1)
    # e_p = (int(x_2), y_2)
    # return f_p,e_p,k
    return x_1,x_2
#处理预测掩码图
def ang(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th, binary = cv2.threshold(gray_img, 30, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # print("contours",contours[0])
    # line1的方法
    area,trg1=cv2.minEnclosingTriangle(contours[0])
    f, e = top(trg1)
    r_point = np.zeros((2, 2), dtype=int)
    r_point[0] = f
    r_point[1] = e
    # point = np.zeros((2, 2), dtype=int)
    # point[0] = f
    # point[1] = e
    # line2的方法，好一点
    """UNet:设置为0.06"""
    # epsilon = 0.06 * cv2.arcLength(contours[0], True)
    # approx = cv2.approxPolyDP(contours[0], epsilon, True)
    # print("approx",approx.shape)
    # img, f, e = top(approx, imgor)

    return r_point



def error(gt_dir, pred_dir, png_name_list):
    gt_imgs = [join(gt_dir, x + ".png") for x in png_name_list]
    pred_imgs = [join(pred_dir, x + ".png") for x in png_name_list]
    x1=0
    x2=0
    for ind in range(len(gt_imgs)):
        print("pred_imgs[ind]",pred_imgs[ind])
        pred = cv2.imread(pred_imgs[ind])
        label = cv2.imread(gt_imgs[ind])
        pred_point = ang(pred)
        label_point = ang(label)
        pred_x1,pred_x2=calcu(pred_point)
        label_x1,label_x2 = calcu(label_point)
        t1=abs(pred_x1-label_x1)
        t2=abs(pred_x2-label_x2)
        x1+=t1
        x2+=t2
        print("t1", t1)
        print("t2", t2)
    x1=x1/(len(gt_imgs))
    x2 = x2 / (len(gt_imgs))
    return x1,x2


if __name__ == '__main__':
    VOCdevkit_path = 'VOCdevkit'
    image_ids = open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/linetset/val.txt"), 'r').read().splitlines()
    gt_dir = os.path.join(VOCdevkit_path, "VOC2007/SegmentationClass/")
    miou_out_path = "miou_out"
    pred_dir = os.path.join(miou_out_path, 'linetest')

    x1,x2=error(gt_dir, pred_dir, image_ids)
    print("x1",x1)
    print("x2",x2)