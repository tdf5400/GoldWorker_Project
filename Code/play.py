import cv2 as cv
import numpy as np
from math import *
import GetImg as Get

def drawAxis(img, p_, q_, colour, scale):
    p = list(p_)
    q = list(q_)

    angle = atan2(p[1] - q[1], p[0] - q[0])  # angle in radians
    hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))
    # Here we lengthen the arrow by a factor of scale
    q[0] = p[0] - scale * hypotenuse * cos(angle)
    q[1] = p[1] - scale * hypotenuse * sin(angle)
    cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv.LINE_AA)
    # create the arrow hooks
    p[0] = q[0] + 9 * cos(angle + pi / 4)
    p[1] = q[1] + 9 * sin(angle + pi / 4)
    cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv.LINE_AA)
    p[0] = q[0] + 9 * cos(angle - pi / 4)
    p[1] = q[1] + 9 * sin(angle - pi / 4)
    cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv.LINE_AA)

def getOrientation(pts, img):
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i, 0] = pts[i, 0, 0]
        data_pts[i, 1] = pts[i, 0, 1]
    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv.PCACompute2(data_pts, mean)
    # mean, eigenvectors, eigenvalues = cv.PCACompute(data_pts, mean, 2) #image, mean=None, maxComponents=10
    # Store the center of the object
    cntr = (int(mean[0, 0]), int(mean[0, 1]))

    cv.circle(img, cntr, 3, (255, 0, 255), 2)  # 在PCA中心位置画一个圆圈
    p1 = (
    cntr[0] + 0.02 * eigenvectors[0, 0] * eigenvalues[0, 0], cntr[1] + 0.02 * eigenvectors[0, 1] * eigenvalues[0, 0])
    p2 = (
    cntr[0] - 0.02 * eigenvectors[1, 0] * eigenvalues[1, 0], cntr[1] - 0.02 * eigenvectors[1, 1] * eigenvalues[1, 0])
    drawAxis(img, cntr, p1, (0, 255, 0), 1)  # 绿色，较长轴
    drawAxis(img, cntr, p2, (255, 255, 0), 1)  # 黄色
    angle = atan2(eigenvectors[0, 1], eigenvectors[0, 0])  # orientation in radians #PCA第一维度的角度
    cv.imshow('Result', img)
    return angle


# 钩子
hook = cv.imread('..\\img\\Hook.jpg')
cv.imshow("Hook", hook)

hook_gray = cv.cvtColor(hook, cv.COLOR_BGR2GRAY)
ret, hook_thre = cv.threshold(hook_gray, 200, 255, cv.THRESH_BINARY)


# 寻找Hook轮廓
hook_contour = cv.findContours(hook_thre, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)[0]
for i in hook_contour:
    area = cv.contourArea(i)
    if 500 < area < (19040-500):  # 去除面积过大过小
        cv.drawContours(hook, i, -1, (255, 0, 0), 2)
        hook_contour = i
cv.imshow("Hook", hook)

# 图片寻找角度
img = cv.imread('..\\testImg\\0.jpg')
img = cv.resize(img, (640, 480))
cv.imshow("raw", img)

Get.getItem(img)


# # 标注
# for c, j in enumerate(hook_contour):
#     angel = getOrientation(hook_contour, img)
#     print(angel)



cv.waitKey(0)
cv.destroyAllWindows()
