"""
获取窗口图像
"""
import win32gui
import win32con
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import *
import PyQt5.QtGui as QtGui
import sys
import cv2 as cv
import numpy as np


class Locate:  # 坐标表
    ORGIN = (321, 63)  # 原点（出钩点）
    HOOKAREA = ((290, 40), (350, 100))  # 钩子区域(x1, y1), (x2, y2)
    

def get_window(windowName):
    """
    寻找窗口
    :param windowName: 窗口名
    :return: 窗口句柄
    """
    windows = []
    win32gui.EnumWindows(lambda hwnd, parm: parm.append(hwnd), windows)
    for window in windows:
        title = win32gui.GetWindowText(window)

        if title.find(windowName) >= 0:
            hwnd = win32gui.FindWindow(None, win32gui.GetWindowText(window))
            return hwnd
    else:
        return None


def get_img(hwnd):
    """
    获取窗口图像
    :param hwnd: 窗口句柄
    :return: opencv图像
    """

    # 截图
    app = QApplication(sys.argv)
    screen = QApplication.primaryScreen()
    img = screen.grabWindow(hwnd).toImage()

    # 转为np.array
    ptr = img.constBits()
    ptr.setsize(img.byteCount())
    img = np.array(ptr).reshape(img.height(), img.width(), 4)

    return img[:,:,0:3]



class Items:
    item = ""

    class Property:  # 物品属性
        gold_0_path = '..\\img\\Gold_0.jpg'
        gold_1_path = '..\\img\\Gold_1.jpg'
        gold_2_path = '..\\img\\Gold_2.jpg'
        thre = {"gold_2": 0}

    def threshold(self):
        if self.item is "gold_2":
            pass

    def __init__(self, itemName):
        item = itemName

    def path(self):
        pass


def getItem_temp(img):
    """
    模版匹配识别物品
    :return:物品坐标（array）
    """
    # 读物品模版
    template = cv.imread(Items.Property.gold_2_path)
    template_h, template_w = template.shape[:2]
    # 图像四周以黑色填充，增加边界物体的识别率
    img_h, img_w, img_d = img.shape
    img_new = np.zeros((img_h + template_h, img_w + 2 * template_w, img_d), dtype=np.uint8)
    img_new = img  # [:img_h, template_w:img_w+template_w] = img
    # 转灰度
    img_gray = cv.cvtColor(img_new, cv.COLOR_BGR2GRAY)
    template_gray = cv.cvtColor(template, cv.COLOR_BGR2GRAY)
    # 模板匹配
    result = cv.matchTemplate(img_gray, template_gray, cv.TM_CCOEFF_NORMED)
    locate = np.where(0.8 < result)

    for item in zip(*locate[::-1]):
        bottom_right = (item[0] + template_w, item[1] + template_h)
        cv.rectangle(img, item, bottom_right, (0, 0, 255), 2)
    cv.imshow('result', img)


def getItem_Cascade(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    thre = cv.inRange(img, (0,0,0), (150,150,255))
    cv.imshow('thre', thre)

    gold_cascade = cv.CascadeClassifier('..\\testImg\\Train\\output_hook\\cascade.xml')
    golds = gold_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=80)

    for (x, y, w, h) in golds:
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255))

    cv.imshow('golds', img)
    return img


# 获取钩子信息
def getHook(raw):
    hookPoint = Locate.ORGIN  # 出钩点
    HOOKAREA = Locate.HOOKAREA  # 钩子区域(x1, y1), (x2, y2)
    raw = cv.resize(raw, (640, 480))    # 分辨率为640x480
    img = raw[HOOKAREA[0][1]:HOOKAREA[1][1], HOOKAREA[0][0]:HOOKAREA[1][0], :]  # 截取部分
    imgHSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)     # HSV色彩
    thre = cv.inRange(imgHSV, (0, 0, 113), (128, 93, 161))

    # 形态学运算
    morph = thre
    kernel = np.ones((3, 3), dtype=np.uint8)
    morph = cv.dilate(morph, kernel)
    # kernel = np.ones((3, 3), dtype=np.uint8)
    # morph = cv.morphologyEx(morph, cv.MORPH_CLOSE, kernel)
    # kernel = np.ones((2, 2), dtype=np.uint8)
    # morph = cv.morphologyEx(morph, cv.MORPH_OPEN, kernel)
    # kernel = np.ones((17, 17), dtype=np.uint8)
    # morph = cv.morphologyEx(morph, cv.MORPH_CLOSE, kernel)

    # 边缘检测
    contours, hier = cv.findContours(morph, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    maxArea = None  # 取最大面积的区域
    for c in contours:
        # 匹配最小垂直矩形
        if maxArea is None: # 初值
            maxArea = c
            continue
        w0, h0 = cv.boundingRect(maxArea)[2:4]
        x, y, w, h = cv.boundingRect(c)
        if w0*h0 < w*h:
            maxArea = c
    if maxArea is None:     # 无轮廓
        return None
    # x, y, w, h = cv.boundingRect(maxArea)               # 转换竖直矩形参数
    # cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0))  # 画钩子竖直矩形
    rect = cv.minAreaRect(maxArea)      # 拟合最小矩形
    if rect[1][0] > rect[1][1]:     # 角度计算
        angel = rect[2] - 90
    else:
        angel = rect[2]
    angel = angel + 180     # 角度反转
    box = np.int0(cv.boxPoints(rect))   # 计算最小矩形离散点
    cv.drawContours(img, [box], 0, (0, 255, 0), 1)  # 画最小矩形

    cv.line(img, (hookPoint[0]-HOOKAREA[0][0], hookPoint[1]-HOOKAREA[0][1]), (int(rect[0][0]), int(rect[0][1])),
            (0,0,255), thickness=2)
    # 拼接回原图像
    raw[HOOKAREA[0][1]:HOOKAREA[1][1], HOOKAREA[0][0]:HOOKAREA[1][0], :] = img
    cv.imshow('img', raw)
    # cv.imshow('hookArea', img)
    # cv.imshow('thre', thre)
    # cv.imshow('morph', morph)
    return angel

# testimg = cv.imread('../testImg/1.jpg')
# getHook(testimg)
# # getItem_Cascade(testimg)
# cv.waitKey(0)
# cv.destroyAllWindows()


if __name__ == "__main__":
    if 1:
        hwnd = get_window("game.swf")
        # 更改窗口大小
        win32gui.SetWindowPos(hwnd, win32con.HWND_NOTOPMOST, 0, 0, 640, 480, win32con.SHOW_ICONWINDOW)

        while True:
            img = get_img(hwnd)
            print(img.shape)
            cv.imshow("demo", img)
            angel = getHook(img)
            print(angel)

            key = cv.waitKey(100)
            if key == 27:  # Esc退出
                cv.destroyAllWindows()
                break
