"""
ROI获取

"""
import cv2 as cv
import numpy as np

# 全局变量
__img = 0
__output = None

def __refresh(x):
    global __img, __output

    copyImg = __img
    __imgHSV = __img#v2.cvtColor(__img, cv2.COLOR_BGR2HSV)

    __x0 = cv.getTrackbarPos('x0', 'Trackbar')
    __y0 = cv.getTrackbarPos('y0', 'Trackbar')
    pos0 = (__x0, __y0)
    __x1 = cv.getTrackbarPos('x1', 'Trackbar')
    __y1 = cv.getTrackbarPos('y1', 'Trackbar')
    pos1 = (__x1, __y1)
    print(f"Point: {pos0} to {pos1}")
    print(f"x, y, w, h: ({pos0[0]}, {pos0[1]}, {pos1[0]-pos0[0]}, {pos1[1]-pos0[1]})")

    mask = np.zeros(np.shape(copyImg)[:2], dtype=np.uint8)
    mask[pos0[1]:pos1[1], pos0[0]:pos1[0]] = 255
    output = cv.add(copyImg, np.zeros(np.shape(copyImg), dtype=np.uint8), mask=mask)

    __output = output[pos0[1]:pos1[1], pos0[0]:pos1[0]]
    cv.imshow('roi', output)


def __main():
    global __img
    print("启动颜色阈值调试程序！")
    if 0:
        print("使用内置图片！")
        src = cv2.imread('../pic1.jpg')
    else:
        import GetImg, object_detection, win32gui, win32con
        print("使用游戏窗口截图")
        hwnd = GetImg.get_window_view("Adobe Flash")
        win32gui.SetWindowPos(hwnd, win32con.HWND_NOTOPMOST, 0, 0, 800, 600, win32con.SHOW_ICONWINDOW)
        src = GetImg.get_img(hwnd)
        picRoi = object_detection.picRoi  # 获取有效图像
        img = src[picRoi[1]:picRoi[3], picRoi[0]:picRoi[2]]

    if src is None:  # 判断图像存在性
        print("图像不存在！")
    else:
        __img = cv.resize(src, (800, 600))  # 分辨率重定义
        __refresh(None)


picNum = 0
picSavePath = './'
if __name__ == "__main__":
    # 创建调节棒
    cv.namedWindow('Trackbar')
    cv.createTrackbar('x0', 'Trackbar', 314, 800, __refresh)
    cv.createTrackbar('y0', 'Trackbar', 375, 600, __refresh)
    cv.createTrackbar('x1', 'Trackbar', 793, 800, __refresh)
    cv.createTrackbar('y1', 'Trackbar', 600, 600, __refresh)

    __main()


    while(True):
        # Esc退出
        keyAction = cv.waitKey(1)  # 延时1ms
        if keyAction == 27:  # Esc
            cv.destroyAllWindows()
            break
        elif keyAction == ord('s'):
            cv.imwrite(picSavePath + f'num{picNum}.jpg', __output)
            print(f'Pic save to {picSavePath}num{picNum}.jpg with {np.shape(__output)[:2]}')
            picNum += 1
            continue