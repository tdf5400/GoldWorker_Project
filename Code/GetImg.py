"""
该文件存放获取窗口图像相关函数
"""
import win32gui
import win32con
from PyQt5.QtWidgets import QApplication
import cv2 as cv
import numpy as np
import sys


def get_window_view(windowName):
    """
    寻找可视窗口
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


def get_window_ctr(parentHwnd):
    """
    寻找供操控的子窗口
    :param parentHwnd:父窗口的句柄，若无子句柄，则返回父窗口句柄
    :return: None-查找失败, 其他-操控的子窗口句柄
    """
    hwnd = win32gui.FindWindowEx(parentHwnd, 0, None, '')
    # 获取所有子窗口
    hwndChildList = []

    if len(hwndChildList):
        win32gui.EnumChildWindows(hwnd, lambda hwnd, param: param.append(hwnd), hwndChildList)
        for i in hwndChildList:
            if 'FlashPlayer' in win32gui.GetClassName(i):
                return i

    return parentHwnd


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

    return img[:, :, 0:3]


if __name__ == "__main__":
    if 1:
        # hwnd = get_window_view("game.swf")
        hwnd = get_window_view("Adobe Flash")
        # 更改窗口大小
        win32gui.SetWindowPos(hwnd, win32con.HWND_NOTOPMOST, 0, 0, 800, 600, win32con.SHOW_ICONWINDOW)

        while True:
            # img = cv.imread('../dataset/100.jpg')
            img = get_img(hwnd)
            img = cv.resize(img, (800, 600))
            cv.imshow("img", img)


            key = cv.waitKey(100)
            if key == 27:  # Esc退出
                cv.destroyAllWindows()
                break
    elif 1:
        img = cv.imread('../dataset/101.jpg')
        img = cv.resize(img, (640, 480))
        # cv.imshow('raw', img)
        print(getNumber(img, CLASS_LEVEL))
        cv.waitKey(0)
