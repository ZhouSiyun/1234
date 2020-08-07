import cv2
import numpy as np

# 对图像进行处理
def imageProcess(path):
    image = cv2.imread(path)
    image = skinMask(image)
    # 形态学处理，去掉背景的杂质点
    kernel = np.ones((11, 11), np.uint8)  # 设置卷积核
    dilation = cv2.dilate(image, kernel)  # 膨胀操作
    erosion = cv2.erode(dilation, kernel)  # 腐蚀操作
    contour = cv2.Canny(dilation, 30, 70)
    return contour # 返回的是边缘信息

def skinMask(image):
    YCrCb = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
    (y, cr, cb) = cv2.split(YCrCb)  # 拆分出Y,Cr,Cb值
    cr1 = cv2.GaussianBlur(cr, (5, 5), 0)
    _, skin = cv2.threshold(cr1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # Ostu处理
    res = cv2.bitwise_and(image, image, mask=skin)
    return skin


