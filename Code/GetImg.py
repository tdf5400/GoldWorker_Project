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

    return img


# if __name__ == "__main__":
#     hwnd = get_window("game.swf")
#     # 更改窗口大小
#     win32gui.SetWindowPos(hwnd, win32con.HWND_NOTOPMOST, 0, 0, 640, 480, win32con.SHOW_ICONWINDOW)
#
#     while True:
#         img = get_img(hwnd)
#
#         cv.imshow("demo", img)
#
#         key = cv.waitKey(10)
#         if key == 27:  # Esc退出
#             cv.destroyAllWindows()
#             break


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


testimg = cv.imread('../testImg/0.jpg')
getItem_Cascade(testimg)
cv.waitKey(0)
cv.destroyAllWindows()
