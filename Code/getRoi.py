"""
ROI获取

"""
import cv2 as cv
import numpy as np
import GetImg

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
        img = cv2.imread('../pic1.jpg')
    else:
        import GetImg, object_detection, win32gui, win32con
        print("使用游戏窗口截图")
        
        win = GetImg.Window()
        img = win.get_img()

    if img is None:  # 判断图像存在性
        print("图像不存在！")
    else:
        __img = img
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

    win = GetImg.Window()
    while(True):
        __img = win.get_img()
        __refresh(None)
        
        # Esc退出
        keyAction = cv.waitKey(50)  # 延时1ms
        if keyAction == 27:  # Esc
            cv.destroyAllWindows()
            break
        elif keyAction == ord('s'):
            cv.imwrite(picSavePath + f'pic{picNum}.jpg', __output)
            print(f'Pic save to {picSavePath}pic{picNum}.jpg with {np.shape(__output)[:2]}')
            picNum += 1
            continue