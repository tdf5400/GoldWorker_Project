"""
该文件存放获取窗口图像相关函数
"""
import win32gui, win32con, win32api
from PyQt5.QtWidgets import QApplication
import cv2 as cv
import numpy as np
import sys
from config import cfg
from threading import Thread

def get_window_view(winName):
    """
    寻找可视窗口
    :param winName: 窗口名
    :return: 窗口句柄
    """
    windows = []
    win32gui.EnumWindows(lambda hwnd, parm: parm.append(hwnd), windows)
    for window in windows:
        title = win32gui.GetWindowText(window)

        if title.find(winName) >= 0:
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


class Window:
    """
    窗口观测器，负责图像读取、控制输出
    """
    def __init__(self, winName=cfg['name_flash']):
        """
        窗口观测器初始化
        :param winName: 窗口名
        """
        self.winName = winName
        # 获取两个窗口的句柄
        self.winHwnd = get_window_view(winName)
        assert self.winHwnd, "\033[31m[ERROR] Cannot find the game program!\033[0m"
        self.ctrHwnd = get_window_ctr(self.winHwnd)
        assert self.ctrHwnd, "\033[31m[ERROR] Cannot find the control handler of game program!\033[0m"
        # 更改窗口大小
        win32gui.SetWindowPos(self.winHwnd, win32con.HWND_NOTOPMOST, 0, 0, 800, 600, win32con.SHOW_ICONWINDOW)
        
    def get_img(self, winSize=cfg['winSize']):
        """
        获取窗口图像
        :return: opencv图像
        """
        # 若窗口尺寸不对，则更改窗口大小
        nowWinSize = win32gui.GetWindowRect(self.winHwnd)
        w = nowWinSize[2] - nowWinSize[0]
        h = nowWinSize[3] - nowWinSize[1]
        if w != winSize[0] or h != winSize[1]:
            win32gui.SetWindowPos(self.winHwnd, win32con.HWND_NOTOPMOST, 0, 0, winSize[0], winSize[1], win32con.SHOW_ICONWINDOW)

        # 截图
        app = QApplication(sys.argv)
        screen = QApplication.primaryScreen()
        img = screen.grabWindow(self.winHwnd).toImage()

        # 转为np.array
        ptr = img.constBits()
        ptr.setsize(img.byteCount())
        img = np.array(ptr).reshape(img.height(), img.width(), 4)
        
        # 裁剪
        ROI = cfg['ROI']['image']

        return np.array(img[ROI[0][1]:ROI[1][1], ROI[0][0]:ROI[1][0], 0:3])
    
    def press_key(self, key):
        """
        按下按钮
        :param key: 按键名, 如win32con.VK_UP, win32con.VK_DOWN
        """
        win32api.PostMessage(self.ctrHwnd, win32con.WM_KEYDOWN, key, 0)
        win32api.PostMessage(self.ctrHwnd, win32con.WM_KEYUP, key, 0)
        
    def check(self, xy, twiceChick=False):
        """
        鼠标点击
        :param xy: (x, y) 点击的位置
        :param twiceChick: 是否双击
        """
        x, y = xy
        action = win32api.MAKELONG(x, y)
        
        win32api.SendMessage(self.ctrHwnd, win32con.WM_LBUTTONDOWN, win32con.MK_LBUTTON, action)
        win32api.SendMessage(self.ctrHwnd, win32con.WM_LBUTTONUP, win32con.MK_LBUTTON, action)
        if twiceChick:  # 双击
            win32api.SendMessage(self.ctrHwnd, win32con.WM_LBUTTONDOWN, win32con.MK_LBUTTON, action)
            win32api.SendMessage(self.ctrHwnd, win32con.WM_LBUTTONUP, win32con.MK_LBUTTON, action)
        


if __name__ == "__main__":
    if 1:        
        win = Window("Adobe Flash Player")
    
        while True:
            img = win.get_img()
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
