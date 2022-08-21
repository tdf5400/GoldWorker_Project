import cv2
import numpy as np


# 全局变量
__img = 0
__camera = 0

def __refresh(x):
    global __img, __camera

    copyImg = __img
    __imgHSV = __img#v2.cvtColor(__img, cv2.COLOR_BGR2HSV)

    __h = cv2.getTrackbarPos('loDirr_H', 'Trackbar')
    __s = cv2.getTrackbarPos('loDirr_S', 'Trackbar')
    __v = cv2.getTrackbarPos('loDirr_V', 'Trackbar')
    loDirr = (__h, __s, __v)
    __h = cv2.getTrackbarPos('upDirr_H', 'Trackbar')
    __s = cv2.getTrackbarPos('upDirr_S', 'Trackbar')
    __v = cv2.getTrackbarPos('upDirr_V', 'Trackbar')
    upDirr = (__h, __s, __v)
    print(f"Parameter: {loDirr}, {upDirr}")


    threImg = cv2.inRange(__imgHSV, loDirr, upDirr)  # FloodFill计算
    if threImg is None:  # 取色失败则进入下一帧
        print("计算失败！")
        try:
            cv2.destroyWindow('Demo')
        except Exception:
            pass
    else:
        cv2.imshow('Demo', copyImg)
        cv2.imshow('Thre', threImg)


def __main():
    global __img, __camera
    print("启动颜色阈值调试程序！")
    if 0:
        print("使用内置图片！")
        src = cv2.imread('../dataset/85.jpg')
    else:
        import GetImg, win32gui, win32con
        print("使用游戏窗口截图")
        hwnd = GetImg.get_window_view("Adobe Flash")
        win32gui.SetWindowPos(hwnd, win32con.HWND_NOTOPMOST, 0, 0, 800, 600, win32con.SHOW_ICONWINDOW)
        src = GetImg.get_img(hwnd)

    if src is None:  # 判断图像存在性
        print("图像不存在！")
    else:
        __img = cv2.resize(src, (800, 600))  # 分辨率重定义
        __refresh(None)



if __name__ == "__main__":
    # 创建调节棒
    cv2.namedWindow('Trackbar')
    cv2.createTrackbar('loDirr_H', 'Trackbar', 100, 255, __refresh)
    cv2.createTrackbar('loDirr_S', 'Trackbar', 255, 255, __refresh)
    cv2.createTrackbar('loDirr_V', 'Trackbar', 255, 255, __refresh)
    cv2.createTrackbar('upDirr_H', 'Trackbar', 100, 255, __refresh)
    cv2.createTrackbar('upDirr_S', 'Trackbar', 255, 255, __refresh)
    cv2.createTrackbar('upDirr_V', 'Trackbar', 255, 255, __refresh)

    __main()


    while(True):
        # Esc退出
        keyAction = cv2.waitKey(1)  # 延时1ms
        if keyAction == 27:  # Esc
            cv2.destroyAllWindows()
            break
