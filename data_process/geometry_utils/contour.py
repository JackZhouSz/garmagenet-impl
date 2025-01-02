import cv2 as cv
import numpy as np
import random

# 读取图像
src = cv.imread('./geo_imgs/panel_10/geo_img_4.png')

# 创建与src相同大小的黑色图像
dst = np.zeros((src.shape[0] * 2, src.shape[1] * 2, 3), dtype=np.uint8)

# 将图像转换为灰度图像
src_gray = cv.cvtColor(src, cv.COLOR_RGBA2GRAY)

# 应用阈值操作
_, src_thresh = cv.threshold(src_gray, 10, 200, cv.THRESH_BINARY)

# cv.imshow('bin', src_thresh)
# cv.waitKey(0)

# 查找轮廓
contours, hierarchy = cv.findContours(src_thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

# 创建一个列表来存储逼近多边形
poly = []

# 逼近每个轮廓为多边形
for cnt in contours:
    epsilon = 0.0001 * cv.arcLength(cnt, True)
    approx = cv.approxPolyDP(cnt, epsilon, True)
    poly.append(approx)

# 用随机颜色绘制多边形轮廓
for i in range(len(poly)):
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    cv.drawContours(dst, poly, i, color, 1, 8, hierarchy)

# 显示结果
cv.imshow('canvasOutput', dst)
cv.waitKey(0)
cv.destroyAllWindows()

# 释放资源
cv.imwrite('output.png', dst)  # 可以保存结果
