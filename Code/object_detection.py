"""
该文件存放对象识别的相关函数及参数
"""

import cv2 as cv
import numpy as np

gameItems = ['gold_S', 'gold_M', 'gold_L',
            'bone_S', 'bone_L',
            'stone_S', 'stone_L',
            'pack', 'bucket', 'diamond',
            'pig', 'diamondPig'
             ]

scoreBoard_height = 120
goldColorThre = ((0, 215, 248), (56, 255, 255))  # 阈值分割参数
goldAreaLimit = 200     # 最小像素面积
goldAreaS2M = 400       # 小金块与中金块区分阈值
goldAreaM2L = 900       # 中金块与大金块区分阈值

stoneColorThre = (120, 121, 120), (149, 153, 145)
stoneAreaLimit = 200
stoneAreaS2L = 1800   # 小石头与大石头区分阈值

pigColorThre = (49, 84, 149), (60, 98, 161)
pigAreaLim_min = 200
pigAreaLim_max = 1000

diamondColorThre = (209, 209, 109), (255, 255, 192)
diamondAreaLimit = 40

packColorThre = (0, 79, 222), (19, 121, 255)
packAreaLimit = 100

bucketColorThre = (141, 197, 229), (148, 206, 233)
packAreaLimit = 100

boneColorThre = (199, 206, 234), (255, 255, 255)
boneAreaLimit = 1000

diamondPigDistanceLim = 1000

# 数字模板类别
CLASS_MONEY  = 0
CLASS_TARGET = 1
CLASS_TIME   = 2
CLASS_LEVEL  = 3
# 数字模板目录
templateDir = ('../numTemplate/money/',
               '../numTemplate/target/',
               '../numTemplate/timeAndLevel/',
               '../numTemplate/timeAndLevel/')
# 数字模板ROI
templateROI = ((80, 2, 150, 45),      # x, y, w, h
               (130, 45, 150, 45),
               (720, 7, 75, 45),
               (690, 45, 70, 45))
# 数字模板阈值
templateThre = (0.9, 0.9, 0.9, 0.9)


class Locate:  # 坐标表
    ORGIN = (400, 78)                   # 钩子原点（出钩点）
    HOOKAREA = ((360, 60), (440, 120))  # 钩子区域(x1, y1), (x2, y2)


def getGold(img):
    """
    获得金块信息
    :param img: 输入图像（800x600）
    :return:小、中、大金块的(x,y,w,h)
    """
    assert (img.shape == (600, 800, 3)), "[error] resolution should be 800x600!"

    thre = cv.inRange(img, goldColorThre[0], goldColorThre[1])
    # 排除顶端计分板干扰
    thre[:][0:scoreBoard_height] = 0

    # 形态学运算
    kernel = np.ones((9, 9), dtype=np.uint8)
    morph = cv.dilate(thre, kernel=kernel)
    # cv.imshow('morph', morph)

    # 检测
    small = []
    middle = []
    large = []
    contours, hier = cv.findContours(morph, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x, y, w, h = cv.boundingRect(c)
        area = w*h  # 计算面积
        if area < goldAreaLimit:  # 像素值过小则跳过
            continue
        elif area < goldAreaS2M:    # 小
            small.append((x, y, w, h))
        elif area < goldAreaM2L:    # 中
            middle.append((x, y, w, h))
        else:                       # 大
            large.append((x, y, w, h))

    # 显示
    # for x, y, w, h in small:
    #     cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #     cv.putText(img, 'gold_S', (x - int(w / 2), y - int(h / 2)), cv.FONT_HERSHEY_PLAIN,
    #                fontScale=1, color=(0, 255, 0), thickness=2)
    # for x, y, w, h in middle:
    #     cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #     cv.putText(img, 'gold_M', (x - int(w / 2), y - int(h / 2)), cv.FONT_HERSHEY_PLAIN,
    #                fontScale=1, color=(0, 255, 0), thickness=2)
    # for x, y, w, h in large:
    #     cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #     cv.putText(img, 'gold_L', (x - int(w / 2) +25, y - int(h / 2)+20), cv.FONT_HERSHEY_PLAIN,
    #                fontScale=1, color=(0, 255, 0), thickness=2)
    # cv.imshow('img', img)

    return small, middle, large


def getStone(img):
    """
    获得石头信息
    :param img: 输入图像（800x600）
    :return: 小、大石头的(x,y,w,h)
    """
    assert (img.shape == (600, 800, 3)), "[error] resolution should be 800x600!"

    thre = cv.inRange(img, stoneColorThre[0], stoneColorThre[1])
    # 排除顶端计分板干扰
    thre[:][0:scoreBoard_height] = 0

    kernel = np.ones((7, 7), dtype=np.uint8)
    morph = cv.morphologyEx(thre, cv.MORPH_CLOSE, kernel=kernel)
    kernel = np.ones((3, 3), dtype=np.uint8)
    morph = cv.morphologyEx(morph, cv.MORPH_OPEN, kernel=kernel)
    # cv.imshow('morph', morph)

    # 检测
    small = []
    large = []
    contours, hier = cv.findContours(morph, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x, y, w, h = cv.boundingRect(c)
        area = w * h  # 计算面积
        if area < stoneAreaLimit:  # 像素值过小则跳过
            continue
        elif area < stoneAreaS2L:  # 小
            small.append((x, y, w, h))
        else:  # 大
            large.append((x, y, w, h))

    # for x, y, w, h in small:
    #     cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #     cv.putText(img, 'stone_S', (x - int(w / 2), y - int(h / 2)), cv.FONT_HERSHEY_PLAIN,
    #                fontScale=1, color=(0, 255, 0), thickness=2)
    # for x, y, w, h in large:
    #     cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #     cv.putText(img, 'stone_L', (x - int(w / 2) +25, y - int(h / 2)+20), cv.FONT_HERSHEY_PLAIN,
    #                fontScale=1, color=(0, 255, 0), thickness=2)
    # cv.imshow('img', img)

    return small, large


def getPig(img):
    """
    获得猪信息
    :param img: 输入图像（800x600）
    :return: 猪的(x,y,w,h)
    """
    assert (img.shape == (600, 800, 3)), "[error] resolution should be 800x600!"

    thre = cv.inRange(img, pigColorThre[0], pigColorThre[1])
    thre[:][0:scoreBoard_height] = 0    # 排除顶端计分板干扰

    kernel = np.ones((7, 7), dtype=np.uint8)
    morph = cv.morphologyEx(thre, cv.MORPH_CLOSE, kernel=kernel)
    kernel = np.ones((13, 13), dtype=np.uint8)
    morph = cv.morphologyEx(morph, cv.MORPH_OPEN, kernel=kernel)
    # cv.imshow('morph', morph)

    # 检测
    pig = []
    contours, hier = cv.findContours(morph, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x, y, w, h = cv.boundingRect(c)
        area = w * h  # 计算面积
        if pigAreaLim_min < area < pigAreaLim_max:  # 像素值不符合则跳过
            pig.append((x, y, w, h))
        else:
            continue

    # for x, y, w, h in pig:
    #     cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #     cv.putText(img, 'pig', (x - int(w / 2), y - int(h / 2)), cv.FONT_HERSHEY_PLAIN,
    #                fontScale=1, color=(0, 255, 0), thickness=2)
    # cv.imshow('img', img)

    return pig


def getDiamond(img):
    """
    获得钻石信息
    :param img: 输入图像（800x600）
    :return: 钻石的(x,y,w,h)
    """
    assert (img.shape == (600, 800, 3)), "[error] resolution should be 800x600!"

    thre = cv.inRange(img, diamondColorThre[0], diamondColorThre[1])
    thre[:][0:scoreBoard_height] = 0  # 排除顶端计分板干扰

    kernel = np.ones((7, 7), dtype=np.uint8)
    morph = cv.morphologyEx(thre, cv.MORPH_CLOSE, kernel=kernel)
    kernel = np.ones((7, 7), dtype=np.uint8)
    morph = cv.morphologyEx(morph, cv.MORPH_DILATE, kernel=kernel)
    # cv.imshow('morph', morph)

    # 检测
    diamond = []
    contours, hier = cv.findContours(morph, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x, y, w, h = cv.boundingRect(c)
        area = w * h  # 计算面积
        if area < diamondAreaLimit:  # 像素值过小则跳过
            continue
        else:
            diamond.append((x, y, w, h))

    # for x, y, w, h in diamond:
    #     cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #     cv.putText(img, 'diamond', (x - int(w / 2), y - int(h / 2)), cv.FONT_HERSHEY_PLAIN,
    #                fontScale=1, color=(0, 255, 0), thickness=2)
    # cv.imshow('img', img)
    return diamond


def getPack(img):
    """
    获得问号包信息
    :param img: 输入图像（800x600）
    :return: 问号包的(x,y,w,h)
    """
    assert (img.shape == (600, 800, 3)), "[error] resolution should be 800x600!"

    thre = cv.inRange(img, packColorThre[0], packColorThre[1])
    thre[:][0:scoreBoard_height] = 0  # 排除顶端计分板干扰

    kernel = np.ones((7, 7), dtype=np.uint8)
    morph = cv.morphologyEx(thre, cv.MORPH_CLOSE, kernel=kernel)
    # cv.imshow('morph', morph)

    # 检测
    pack = []
    contours, hier = cv.findContours(morph, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x, y, w, h = cv.boundingRect(c)
        area = w * h  # 计算面积
        if area < packAreaLimit:  # 像素值过小则跳过
            continue
        else:
            pack.append((x, y, w, h))

    # for x, y, w, h in pack:
    #     cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #     cv.putText(img, 'pack', (x - int(w / 2), y - int(h / 2)), cv.FONT_HERSHEY_PLAIN,
    #                fontScale=1, color=(0, 255, 0), thickness=2)
    # cv.imshow('img', img)

    return pack


def getBucket(img):
    """
    获得爆炸桶信息
    :param img: 输入图像（800x600）
    :return: 爆炸桶的(x,y,w,h)
    """
    assert (img.shape == (600, 800, 3)), "[error] resolution should be 800x600!"

    thre = cv.inRange(img, bucketColorThre[0], bucketColorThre[1])
    thre[:][0:scoreBoard_height] = 0  # 排除顶端计分板干扰

    kernel = np.ones((7, 7), dtype=np.uint8)
    morph = cv.morphologyEx(thre, cv.MORPH_CLOSE, kernel=kernel)
    kernel = np.ones((5, 5), dtype=np.uint8)
    morph = cv.morphologyEx(morph, cv.MORPH_OPEN, kernel=kernel)
    # cv.imshow('morph', morph)

    # 检测
    bucket = []
    contours, hier = cv.findContours(morph, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x, y, w, h = cv.boundingRect(c)
        area = w * h  # 计算面积
        if area < packAreaLimit:  # 像素值过小则跳过
            continue
        else:
            bucket.append((x, y, w, h))

    # for x, y, w, h in bucket:
    #     cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #     cv.putText(img, 'bucket', (x - int(w / 2), y - int(h / 2)), cv.FONT_HERSHEY_PLAIN,
    #                fontScale=1, color=(0, 255, 0), thickness=2)
    # cv.imshow('img', img)

    return bucket


def getBone(img):
    """
    获得骨头信息
    :param img: 输入图像（800x600）
    :return: 骨头的(x,y,w,h)
    """
    assert (img.shape == (600, 800, 3)), "[error] resolution should be 800x600!"

    thre = cv.inRange(img, boneColorThre[0], boneColorThre[1])
    thre[:][0:scoreBoard_height] = 0  # 排除顶端计分板干扰

    kernel = np.ones((11, 11), dtype=np.uint8)
    morph = cv.morphologyEx(thre, cv.MORPH_DILATE, kernel=kernel)
    # cv.imshow('morph', morph)

    # 检测
    bone_S = [] # 棒状骨头
    bone_L = [] # 头盖骨
    contours, hier = cv.findContours(morph, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x, y, w, h = cv.boundingRect(c)
        area = w * h  # 计算面积
        if area < boneAreaLimit:  # 像素值过小则跳过
            continue
        else:
            # 滤去问号和炸药桶
            try:
                thre_2 = cv.inRange(img[y-int(h/2):y+int(h/2), x-int(w/2):x+int(w/2)],
                                    (0, 0, 181), (100, 113, 255))
            except Exception:
                break
            contours_2 = cv.findContours(thre_2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
            if not len(contours_2):     # 无内鬼,区分骨头类型（通过黑色数量）
                if np.unique(thre[y-int(h/2):y+int(h/2), x-int(w/2):x+int(w/2)], return_counts=True)[1][1] < 70:
                    bone_S.append((x, y, w, h))
                else:
                    bone_L.append((x, y, w, h))

    # for x, y, w, h in bone_S:
    #     cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #     cv.putText(img, 'bone_S', (x - int(w / 2), y - int(h / 2)), cv.FONT_HERSHEY_PLAIN,
    #                fontScale=1, color=(0, 255, 0), thickness=2)
    # for x, y, w, h in bone_L:
    #     cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #     cv.putText(img, 'bone_L', (x - int(w / 2), y - int(h / 2)), cv.FONT_HERSHEY_PLAIN,
    #                fontScale=1, color=(0, 255, 0), thickness=2)
    # cv.imshow('img', img)

    return bone_S, bone_L


def getItems(img):
    gold_S, gold_M, gold_L = getGold(img)
    bone_S, bone_L = getBone(img)
    stone_S, stone_L = getStone(img)
    pack = getPack(img)
    bucket = getBucket(img)

    # 钻石猪判断
    pig = getPig(img)
    diamond = getDiamond(img)
    diamondPig = []
    del_list = {'pig': [], 'diamond': []}
    if len(pig) and len(diamond):
        for i in range(len(pig)):
            for j in range(len(diamond)):
                if (pig[i][0]-diamond[j][0])**2 + (pig[i][1]-diamond[j][1])**2 < diamondPigDistanceLim:
                    diamondPig.append(pig[i])
                    del_list['pig'].append(i)
                    del_list['diamond'].append(j)

        del_list['pig'] = np.unique(del_list['pig'])[::-1]
        del_list['diamond'] = np.unique(del_list['diamond'])[::-1]
        for i in del_list['pig']:
            del pig[i]
        for i in del_list['diamond']:
            del diamond[i]


    return {'gold_S': gold_S, 'gold_M': gold_M, 'gold_L': gold_L,
            'bone_S': bone_S, 'bone_L': bone_L,
            'stone_S': stone_S, 'stone_L': stone_L,
            'pack': pack, 'bucket': bucket, 'diamond': diamond,
            'pig': pig, 'diamondPig': diamondPig}


def getNumber(img, numClass):
    """
    模板匹配数字
    :param img: 图片
    :param numClass: 类别选择，如CLASS_MONEY
    :return:
    """
    global tempThre, templateDir, templateThre, xxxx
    global CLASS_MONEY, CLASS_TARGET, CLASS_TIME, CLASS_LEVEL

    DEBUG = 0  # 是否框选(DEBUG)
    if DEBUG:
        copy = img.copy()

    assert CLASS_MONEY <= numClass <= CLASS_LEVEL, 'Invalid numClass!'

    # ROI - 减少计算量
    if not DEBUG:
        ROI_x, ROI_y, ROI_w, ROI_h = templateROI[numClass]
        img = img[ROI_y:ROI_y+ROI_h, ROI_x:ROI_x+ROI_w]
    else:
        ROI_x, ROI_y = 0, 0

    # 录入数字模板
    tempDir = templateDir[numClass]  # 数字模板目录
    scoreTemplate = []
    for i in range(0, 10):
        temp = cv.imread(tempDir + str(i) + '.jpg')
        assert not temp is None, "No image template {}".format(tempDir + str(i) + '.jpg')
        scoreTemplate.append(temp)

    # 模板匹配
    infos = []
    shapeInfo = {'x': img.shape[1], 'y': img.shape[0], 'w': 0, 'h': 0}
    for num in range(0, 10):
        ROI_h, ROI_w = scoreTemplate[num].shape[:2]
        output_temp = cv.matchTemplate(img, scoreTemplate[num], method=cv.TM_CCOEFF_NORMED)
        locates = np.where(output_temp >= templateThre[numClass])
        locates = zip(*locates[::-1])
        for i in locates:
            infos.append((i[0], i[1], num))  # (x轴，y轴，数值)
            shapeInfo['w'] += ROI_w
            if shapeInfo['x'] > i[0]:
                shapeInfo['x'] = i[0]
            if shapeInfo['y'] > i[1]:
                shapeInfo['y'] = i[1]
            if shapeInfo['h'] < ROI_h:
                shapeInfo['h'] = ROI_h
            if DEBUG:
                cv.rectangle(copy, templateROI[numClass][:2],
                             (templateROI[numClass][0] + templateROI[numClass][2],
                              templateROI[numClass][1] + templateROI[numClass][3]), (0, 255, 0))
                cv.rectangle(copy, i, (i[0] + ROI_w, i[1] + ROI_h), (0, 0, 255))
                cv.putText(copy, str(num), (i[0], i[1] + 2*ROI_h), cv.FONT_HERSHEY_SIMPLEX,
                           fontScale=0.5, color=(0, 0, 255), thickness=1)
    if DEBUG:
        cv.imshow('debug', copy)
        print(infos)
    # 数字解析
    for i in range(len(infos)):  # 冒泡排序
        minNum = i
        for j in range(i + 1, len(infos)):
            if infos[j][0] < infos[minNum][0]:
                minNum = j
        else:
            infos[i], infos[minNum] = infos[minNum], infos[i]
    value = 0
    for i in infos:
        value = value * 10 + i[2]

    return value, (shapeInfo['x'] + ROI_x, shapeInfo['y'] + ROI_y, shapeInfo['w'], shapeInfo['h'])


def drawNumbers(img):
    """
    画出所有的数字
    :param img:要画的图像
    :return: output - 画之后的图像
    """
    for i in range(CLASS_MONEY, CLASS_LEVEL+1):
        info = getNumber(img, i)
        # print(info)
        x, y, w, h = info[1]
        cv.rectangle(img, (x,y), (x+w, y+h), color=(0, 0, 255), thickness=1)
        cv.putText(img, str(info[0]), (x, y-2), cv.FONT_HERSHEY_SIMPLEX,
                   fontScale=0.5, color=(0, 0, 255), thickness=2)

    return img


def getHookAngel(raw):
    """
    获取钩子角度
    :param img:
    :return: 钩子角度
    """
    assert (raw.shape == (600, 800, 3)), "[error] resolution should be 800x600!"
    DEBUG = 0   # 调试模式

    hookPoint = Locate.ORGIN  # 出钩点
    HOOKAREA = Locate.HOOKAREA  # 钩子区域(x1, y1), (x2, y2)
    img = raw[HOOKAREA[0][1]:HOOKAREA[1][1], HOOKAREA[0][0]:HOOKAREA[1][0], :]  # 截取部分
    imgHSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)     # HSV色彩
    thre = cv.inRange(imgHSV, (0, 0, 120), (10, 10, 155))

    # 形态学运算
    morph = thre
    kernel = np.ones((5, 5), dtype=np.uint8)
    morph = cv.dilate(morph, kernel)

    # kernel = np.ones((3, 3), dtype=np.uint8)
    # morph = cv.morphologyEx(morph, cv.MORPH_CLOSE, kernel)
    # kernel = np.ones((5, 5), dtype=np.uint8)
    # morph = cv.morphologyEx(morph, cv.MORPH_OPEN, kernel)
    # kernel = np.ones((17, 17), dtype=np.uint8)
    # morph = cv.morphologyEx(morph, cv.MORPH_CLOSE, kernel)

    # 边缘检测
    contours, hier = cv.findContours(morph, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    maxArea = None  # 取最大面积的区域
    for c in contours:
        # 匹配最小垂直矩形
        if maxArea is None:  # 初值
            maxArea = c
            continue
        w0, h0 = cv.boundingRect(maxArea)[2:4]
        x, y, w, h = cv.boundingRect(c)
        if w0 * h0 < w * h:
            maxArea = c
    if maxArea is None:  # 无轮廓
        return None
    # x, y, w, h = cv.boundingRect(maxArea)               # 转换竖直矩形参数
    # cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0))  # 画钩子竖直矩形
    rect = cv.minAreaRect(maxArea)  # 拟合最小矩形
    if rect[1][0] > rect[1][1]:  # 角度计算
        angel = rect[2] - 90
    else:
        angel = rect[2]
    angel = angel + 180  # 角度反转

    if DEBUG:
        box = np.int0(cv.boxPoints(rect))  # 计算最小矩形离散点
        cv.drawContours(img, [box], 0, (0, 255, 0), 1)  # 画最小矩形

        cv.line(img, (hookPoint[0] - HOOKAREA[0][0], hookPoint[1] - HOOKAREA[0][1]), (int(rect[0][0]), int(rect[0][1])),
                (0, 0, 255), thickness=2)
        # 拼接回原图像
        raw[HOOKAREA[0][1]:HOOKAREA[1][1], HOOKAREA[0][0]:HOOKAREA[1][0], :] = img
        cv.imshow('img', raw)
        cv.imshow('hookArea', img)
        cv.imshow('thre', thre)
        cv.imshow('morph', morph)
    return -angel   # 返回极坐标角度


if __name__ == '__main__':
    import GetImg
    import win32gui, win32con
    hwnd = GetImg.get_window_view("game.swf")
    win32gui.SetWindowPos(hwnd, win32con.HWND_NOTOPMOST, 0, 0, 800, 600, win32con.SHOW_ICONWINDOW)
    while 1:
        # 获取物体
        img = GetImg.get_img(hwnd)
        # img = cv.imread('../dataset/84.jpg')
        img = cv.resize(img, (800, 600))
        Items = getItems(img)
        for i in gameItems:
            num = 0
            if Items[i]:
                for x, y, w, h in Items[i]:
                    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv.putText(img, i+"-"+str(num), (x - int(w / 2), y - int(h / 2)), cv.FONT_HERSHEY_SIMPLEX,
                               fontScale=0.5, color=(0, 255, 0), thickness=2)
                    num += 1

        # 获取数字
        img = drawNumbers(img)

        cv.imshow('demo', img)

        key = cv.waitKey(50)
        if key == 27:  # Esc退出
            cv.destroyAllWindows()
            break

    # img = cv.imread('../dataset/85.jpg')
    # img = cv.resize(img, (800, 600))
    # cv.imshow('img', img)
    # getGold(img)
    # getStone(img)
    # getPig(img)
    # getDiamond(img)
    # getPack(img)
    # getBucket(img)
    # getBone(img)
    # print(getNumber(img, numClass=CLASS_TIME))
    # cv.waitKey(0)

