import cv2
# from PIL import Image
# img1="img/or/344.jpg"
# img2="img/png/344.jpg"
# image = Image.open(img1)
# old_img=Image.open(img2)
# image   = Image.blend(old_img, image, 0.3)
# image.save("img/jpg/344.jpg")


def top(trg1,imgor):
    a=trg1[0][0]
    b = trg1[1][0]
    c = trg1[2][0]
    if a[1]<b[1] and a[1]<c[1]:
        x = (b[0] + c[0]) // 2
        y = (b[1] + c[1]) // 2
        cv2.line(imgor, (int(x), int(y)), (int(a[0]),int(a[1])), (0, 255, 0), 15)
        e=tuple(a)
        f = (int(x), int(y))
    elif c[1]<a[1] and b[1]>c[1]:
        x = (b[0] + a[0]) // 2
        y = (b[1] + a[1]) // 2
        cv2.line(imgor, (int(x), int(y)),(int(c[0]),int(c[1])), (0, 255, 0), 15)
        e = tuple(c)
        f = (int(x), int(y))
    else:
        x = (a[0] + c[0]) // 2
        y = (a[1] + c[1]) // 2
        cv2.line(imgor, (int(x), int(y)), (int(b[0]),int(b[1])), (0, 255, 0), 15)
        e = tuple(b)
        f = (int(x), int(y))
    return imgor,f,e

#处理预测掩码图
def ang(img,imgor):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th, binary = cv2.threshold(gray_img, 30, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # line1的方法
    area,trg1=cv2.minEnclosingTriangle(contours[0])
    img, f, e = top(trg1, imgor)
    # point = np.zeros((2, 2), dtype=int)
    # point[0] = f
    # point[1] = e
    # line2的方法，好一点
    """UNet:设置为0.06"""
    # epsilon = 0.06 * cv2.arcLength(contours[0], True)
    # approx = cv2.approxPolyDP(contours[0], epsilon, True)
    # print("approx",approx.shape)
    # img, f, e = top(approx, imgor)

    return img

if __name__ == '__main__':
    img1 = "img/or/344.jpg"
    img2 = "img/bisenet1/png/344.jpg"
    image1=cv2.imread(img1)
    image2 = cv2.imread(img2)
    imgs = ang(image2, image1)
    cv2.imwrite("img/line/344.jpg",imgs)