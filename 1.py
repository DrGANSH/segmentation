from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import  cv2 #要用4.4版本的opencv
pred="miou_out/linetest/103.png"
label="VOCdevkit/VOC2007/SegmentationClass/8.png"
# # pred = Image.open(pred)
# # label = Image.open(label)
# # # print("")
# # # plt.figure("Image")  # 图像窗口名称
# # plt.imshow(pred)
# # # plt.axis('on')  # 关掉坐标轴为 off
# # # plt.title('image')  # 图像题目
# # plt.show()
# # plt.imshow(label)
# # plt.show()
# # # image.show()
# pred=cv2.imread(pred)
# label=cv2.imread(label)
# # print("label",label.shape)
# # print("",label[553,821])
# pred = cv2.cvtColor(pred, cv2.COLOR_BGR2GRAY)
# label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
# # print("label",label.shape)
# # print("pred",pred.shape)
# pred = np.where(pred > 10, 1, 0).astype('uint8')
# label = np.where(label > 10, 1, 0).astype('uint8')
# #
# label   = Image.fromarray(np.uint8(label))
# plt.imshow(label)
# plt.show()
#
# pred   = Image.fromarray(np.uint8(pred))
# plt.imshow(pred)
# plt.show()


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


def ang(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th, binary = cv2.threshold(gray_img, 30, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    print("contours",contours[0])
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

# pred = Image.open(pred)
# plt.imshow(pred)
# plt.show()
pred1 = cv2.imread(pred)
pred = cv2.cvtColor(pred1, cv2.COLOR_BGR2GRAY)
th, pred = cv2.threshold(pred, 30, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(pred, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
pred1=cv2.drawContours(pred1,contours,-1,(0,0,255),5)
cv2.imshow("pred1",pred1)
cv2.waitKey(0)
print("contours",contours)
area,trg1=cv2.minEnclosingTriangle(contours[0])

pred   = Image.fromarray(np.uint8(pred))
plt.imshow(pred)
plt.show()

# pred = cv2.imread(pred)
# pred_point = ang(pred)
# pred = Image.open(pred)
# print("pred_point",pred_point)