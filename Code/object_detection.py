"""
该文件存放对象识别的相关函数及参数
"""

import cv2 as cv
import numpy as np

picRoi = (6, 0, 777, 561)  # x0,y0, x1,y1

gameItems = ['gold_S', 'gold_M', 'gold_L',
             'bone_S', 'bone_L',
             'stone_S', 'stone_L',
             'pack', 'bucket', 'diamond',
             'pig', 'diamondPig'
             ]

scoreBoard_height = 100

goldColorThre = ((0, 215, 248), (56, 255, 255))  # 阈值分割参数
goldAreaLimit = 200  # 最小像素面积
goldAreaS2M = 400  # 小金块与中金块区分阈值
goldAreaM2L = 900  # 中金块与大金块区分阈值
goldSize = ((20, 20), (35, 35), (90, 95))  # 金块的长宽，S to L，(w, h)

stoneColorThre = (120, 121, 120), (149, 153, 145)
stoneAreaLimit = 200
stoneAreaS2L = 1800  # 小石头与大石头区分阈值
stoneSize = ((35, 30), (55, 50))  # 石头的长宽，S to L，(w, h)

pigColorThre = (49, 84, 149), (60, 98, 161)
pigAreaLim_min = 200
pigAreaLim_max = 1000

diamondColorThre = (209, 209, 109), (255, 255, 192)
diamondAreaLimit = 40

packColorThre = (0, 79, 222), (19, 121, 255)
packAreaLimit = 100
packSize = (-13, -15, 42, 47)  # 礼包的长宽校正，(x, y, w, h)

bucketColorThre = (141, 197, 229), (148, 206, 233)
packAreaLimit = 100

boneColorThre = (199, 206, 234), (255, 255, 255)
boneAreaLimit = 1000

diamondPigDistanceLim = 1000

# 数字模板类别
CLASS_MONEY = 0
CLASS_TARGET = 1
CLASS_TIME = 2
CLASS_LEVEL = 3
# 数字模板目录
templateDir = ('../numTemplate/money/',
               '../numTemplate/target/',
               '../numTemplate/timeAndLevel/',
               '../numTemplate/timeAndLevel/')
# 数字模板ROI
templateROI = ((75, 8, 117, 37),  # x, y, w, h
               (128, 48, 120, 34),
               (733, 11, 51, 34),
               (698, 48, 52, 34))
# 数字模板阈值
templateThre = (0.68, 0.7, 0.73, 0.75)  # (0.129, 0.09, 0.7, 0.7)


class Locate:  # 坐标表
    ORGIN = (400, 73)  # 钩子原点（出钩点）
    HOOKAREA = ((360, 60), (440, 120))  # 钩子区域(x1, y1), (x2, y2)


class Sta:  # 状态表
    # UI
    UI_MAIN = 0  # 主菜单
    UI_SHOP = 1  # 商店
    UI_PLAY = 2  # 挖矿
    UI_END = 3  # 结算
    UI_OTHER = 4 # 其他界面


def figEnhance(img):
    """
    图像增强
    :param img:
    :return: 灰度图像
    """
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h = img_hsv[:, :, 0]
    s = img_hsv[:, :, 1]
    v = img_hsv[:, :, 2]
    # 画质增强算子
    kernel = np.array([[2, 1],
                       [-0.5, -2.4]])
    h = cv.filter2D(h, 8, kernel)
    s = cv.filter2D(s, 8, kernel)
    v = cv.filter2D(v, 8, kernel)
    diff = cv.absdiff(s, v)
    add = cv.add(diff, h)
    # cv.imshow('diff', diff)
    # cv.imshow('add', add)
    return add


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
        area = w * h  # 计算面积
        if area < goldAreaLimit:  # 像素值过小则跳过
            continue
        elif area < goldAreaS2M:  # 小
            small.append((x, y, goldSize[0][0], goldSize[0][1]))
        elif area < goldAreaM2L:  # 中
            middle.append((x, y, goldSize[1][0], goldSize[1][1]))
        else:  # 大
            large.append((x, y, goldSize[2][0], goldSize[2][1]))

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
            small.append((x, y, stoneSize[0][0], stoneSize[0][1]))
        else:  # 大
            large.append((x, y, stoneSize[1][0], stoneSize[1][1]))

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
    thre[:][0:scoreBoard_height] = 0  # 排除顶端计分板干扰

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
            pack.append((x + packSize[0], y + packSize[1], packSize[2], packSize[3]))

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
    bone_S = []  # 棒状骨头
    bone_L = []  # 头盖骨
    contours, hier = cv.findContours(morph, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x, y, w, h = cv.boundingRect(c)
        area = w * h  # 计算面积
        if area < boneAreaLimit:  # 像素值过小则跳过
            continue
        else:
            # 滤去问号和炸药桶
            try:
                thre_2 = cv.inRange(img[y - int(h / 2):y + int(h / 2), x - int(w / 2):x + int(w / 2)],
                                    (0, 0, 181), (100, 113, 255))
            except Exception:
                break
            contours_2 = cv.findContours(thre_2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
            if not len(contours_2):  # 无内鬼,区分骨头类型（通过黑色数量）
                if np.unique(thre[y - int(h / 2):y + int(h / 2), x - int(w / 2):x + int(w / 2)], return_counts=True)[1][
                    1] < 70:
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
                if (pig[i][0] - diamond[j][0]) ** 2 + (pig[i][1] - diamond[j][1]) ** 2 < diamondPigDistanceLim:
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
    global tempThre, templateDir, templateThre
    global CLASS_MONEY, CLASS_TARGET, CLASS_TIME, CLASS_LEVEL

    DEBUG = 0  # 是否框选(DEBUG)
    if DEBUG:
        copy = img.copy()

    assert CLASS_MONEY <= numClass <= CLASS_LEVEL, 'Invalid numClass!'

    # ROI - 减少计算量
    if not DEBUG:
        ROI_x, ROI_y, ROI_w, ROI_h = templateROI[numClass]
        img = img[ROI_y:ROI_y + ROI_h, ROI_x:ROI_x + ROI_w]
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
        # if(numClass == CLASS_TIME or numClass == CLASS_LEVEL):
        #     output_temp = cv.matchTemplate(img, scoreTemplate[num], method=cv.TM_CCOEFF_NORMED)
        #     locates = np.where(output_temp >= templateThre[numClass])
        #     locates = zip(*locates[::-1])

        for i in locates:
            infos.append((i[0], i[1], num, output_temp[i[1]][i[0]]))  # (x轴，y轴，数值, 匹配值)

    # 数字解析
    for i in range(len(infos)):  # 冒泡排序
        minNum = i
        for j in range(i + 1, len(infos)):
            if infos[j][0] < infos[minNum][0]:
                minNum = j
        else:
            infos[i], infos[minNum] = infos[minNum], infos[i]

    # 滤去对同个位置的重复识别（选取匹配优的）
    delIndexList = []
    for i in range(len(infos)):
        p0 = infos[i][:2]  # 坐标
        pv = infos[i][2]  # 数字
        for j in range(i + 1, len(infos)):
            distance = abs(p0[0] - infos[j][0])
            if distance < 5:  # 滤波阈值
                if infos[j][3] >= pv:
                    delIndexList.append(j)
                else:
                    delIndexList.append(i)
    delIndexList = list(set(delIndexList))  # 去除重复值，防止误删
    delIndexList.sort()
    for i in delIndexList[::-1]:
        infos.pop(i)

    # 组合结果
    value = 0
    for i in infos:
        # 数值信息
        value = value * 10 + i[2]
        # 选框大小信息
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
            cv.putText(copy, str(num), (i[0], i[1] + 2 * ROI_h), cv.FONT_HERSHEY_SIMPLEX,
                       fontScale=0.5, color=(0, 0, 255), thickness=1)
    if DEBUG:
        cv.imshow('debug', copy)
        print('[DEBUG] Infos:', infos)

    return value, (shapeInfo['x'] + ROI_x, shapeInfo['y'] + ROI_y, shapeInfo['w'], shapeInfo['h'])


def drawNumbers(img):
    """
    画出所有的数字
    :param img:要画的图像
    :return: output - 画之后的图像
    """
    for i in range(CLASS_MONEY, CLASS_LEVEL + 1):
        info = getNumber(img, i)
        # print(info)
        x, y, w, h = info[1]
        cv.rectangle(img, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=1)
        cv.putText(img, str(info[0]), (x, y - 2), cv.FONT_HERSHEY_SIMPLEX,
                   fontScale=0.5, color=(0, 0, 255), thickness=2)

    return img


def __getHookAngle_featurePoint(src, hook_point=np.array((40., 13.))):
    """
    函数“获取钩子角度”中获取特征点的子函数
    :param src: 二值化图像
    :param hook_point: 钩子原点（出钩点）
    :return: 钩子的角度及三个特征点，[角度, [钩子中心， 钩子边1， 钩子边2]]
    角度从极轴顺时针增加
    """
    # 特征点获取
    featurePoints = cv.goodFeaturesToTrack(src, 0, 0.05, 3)

    # 特征点过滤
    if featurePoints is not None:
        # print('featurePoints', featurePoints)
        outputPoints = []
        dlen = featurePoints.shape[0]
        featurePoints = np.reshape(featurePoints, (dlen, 2))
        for i in range(dlen):
            # 离出勾原点过远的点不予考虑
            if abs(np.linalg.norm(hook_point - featurePoints[i]) - 12) > 4:
                continue

            # print('i', i)
            candidate_points = []
            p = featurePoints[i]
            for j in range(dlen):
                if j == i:
                    continue
                distance = np.linalg.norm(p - featurePoints[j])
                # print('distance', distance)
                if abs(distance - 18) < 5:
                    candidate_points.append(featurePoints[j])

            candidate_points = np.array(candidate_points)
            clen = candidate_points.shape[0]
            # print('clen', clen)
            if clen < 2:  # 相符特征点不足2个
                continue
            # print('candidate_points', candidate_points)

            for j in range(0, clen - 1):
                for k in range(j + 1, clen):
                    # 判断中心点是否在上方
                    if candidate_points[j][0] < candidate_points[k][0]:  # 角点分方向，定点为A，左点为B，右点为C
                        p_l = candidate_points[j]
                        p_r = candidate_points[k]
                    else:
                        p_l = candidate_points[k]
                        p_r = candidate_points[k]
                    v_BC = np.transpose(p_r - p_l)
                    v_BA = np.transpose(p - p_l)
                    if np.cross(v_BC, v_BA) >= 0:
                        continue

                    a = p[0] - hook_point[0]
                    b = np.linalg.norm(p - hook_point)
                    angle = np.arccos(a / b)

                    # 中点有点偏差，进行旋转补偿
                    compensate_angle = 9 / 180 * np.pi
                    compensate_cos = np.cos(compensate_angle)
                    compensate_sin = np.sin(compensate_angle)
                    compensate_mat = np.array(((compensate_cos, -compensate_sin),
                                               (compensate_sin, compensate_cos)))
                    p = np.dot(compensate_mat, np.reshape((p - hook_point), (2, 1)))
                    p = np.transpose(p)
                    p = np.reshape(p, (2,))
                    p = p + hook_point
                    angle = angle + compensate_angle

                    output = np.array((angle, np.array((p, p_l, p_r), dtype=np.uint8)), dtype=object)
                    outputPoints.append(output)

        # 有多个结果时，选择最佳结果
        olen = len(outputPoints)
        if olen == 1:
            return outputPoints[0]
        elif olen > 1:
            maxValue = [0, 65535]  # [序号, 评分]
            for i in range(olen):
                # 评分
                p, p_l, p_r = outputPoints[i][1]

                # 求三角形边长
                a = np.linalg.norm(p_r - p_l)
                b = np.linalg.norm(p - p_r)
                c = np.linalg.norm(p - p_l)
                if a < max(b, c):  # a应为最长边
                    continue
                if abs(a - 27) > 3 or abs(b - 18) > 3 or abs(c - 18) > 3:
                    continue

                # 以两边长度相同程度判定
                value = abs(np.linalg.norm(p - p_r) - np.linalg.norm(p - p_l))
                if value < maxValue[1]:
                    maxValue = [i, value]
            return outputPoints[maxValue[0]]
        else:
            return None
    return None


def getHookAngle(raw):
    """
    获取钩子角度
    :param raw: 画面图像
    :return: 钩子角度, None - 失败
    """
    assert (raw.shape == (600, 800, 3)), "[error] resolution should be 800x600!"
    DEBUG = 0  # 调试模式

    hookPoint = np.array(Locate.ORGIN)  # 出钩点
    HOOKAREA = np.array(Locate.HOOKAREA)  # 钩子区域(x1, y1), (x2, y2)
    img = raw[HOOKAREA[0][1]:HOOKAREA[1][1], HOOKAREA[0][0]:HOOKAREA[1][0], :]  # 截取部分
    imgHSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)  # 转HSV

    # 图像分割
    SV = cv.addWeighted(imgHSV[:, :, 1], -1, imgHSV[:, :, 2], 0.95, 0)
    thre = cv.inRange(SV, 100, 140)

    # 面积过少则判定钩子不在区域内
    if sum(thre.flat == 255) < 200:
        return None

    if DEBUG:
        thre = cv.resize(thre, ((HOOKAREA[1][0] - HOOKAREA[0][0]) * 2, (HOOKAREA[1][1] - HOOKAREA[0][1]) * 2))
        cv.imshow('hook_thre', thre)
        thre = cv.resize(thre, ((HOOKAREA[1][0] - HOOKAREA[0][0]), (HOOKAREA[1][1] - HOOKAREA[0][1])))

    # 获得特征点
    angle = None
    hookFeature = __getHookAngle_featurePoint(thre, hook_point=hookPoint - HOOKAREA[0])
    if hookFeature is not None:
        angle = hookFeature[0]

        if DEBUG:
            raw_debug0 = raw.copy()
            startPoint = hookPoint
            endPoint = hookFeature[1][0] + HOOKAREA[0]
            v = endPoint - startPoint
            v = v * 50
            endPoint = v + startPoint

            cv.line(raw_debug0, startPoint, endPoint, color=(0, 0, 255), thickness=5)
            cv.imshow('hook_feature', raw_debug0)

    if DEBUG:
        hookFeature_all = cv.goodFeaturesToTrack(thre, 0, 0.35, 15)
        if hookFeature_all is not None:
            img_debug1 = img.copy()
            for i in hookFeature_all:
                i = np.array(i[0], dtype=np.uint8)
                cv.circle(img_debug1, i, 3, color=(0, 0, 255))
            img_debug1 = cv.resize(img_debug1, (img_debug1.shape[1] * 3, img_debug1.shape[0] * 3))
            cv.imshow('hookFeature_all', img_debug1)
    return angle  # 返回极坐标角度


def getUI(src):
    """
    获取UI状态
    :param img: 图片
    :return: 响应的UI类别
    """
    if src[314][375][0] == 58 and \
            src[314][375][1] == 136 and \
            src[314][375][2] == 214:    # 主界面
        return Sta.UI_MAIN
    elif src[314][375][0] == 1 and \
            src[314][375][1] == 220 and \
            src[314][375][2] == 255:    # 结算
        return Sta.UI_END
    elif src[10][10][0] == 41 and \
            src[10][10][1] == 70 and \
            src[10][10][2] == 118:      # 商店
        return Sta.UI_SHOP
    elif src[10][10][0] == 52 and \
            src[10][10][1] == 208 and \
            src[10][10][2] == 255:      # 挖矿
        return Sta.UI_PLAY
    return Sta.UI_OTHER     # 其他界面


if __name__ == '__main__':
    import GetImg
    import win32gui, win32con

    hwnd = GetImg.get_window_view("Adobe")
    # hwnd = GetImg.get_window_view("game.swf")
    win32gui.SetWindowPos(hwnd, win32con.HWND_NOTOPMOST, 0, 0, 800, 600, win32con.SHOW_ICONWINDOW)
    while 1:
        # 画面预处理
        img = GetImg.get_img(hwnd)
        # img = cv.imread('../dataset/84.jpg')
        # img = cv.imread('../pic1.jpg')
        img = cv.resize(img, (800, 600))
        img = img[picRoi[1]:picRoi[3], picRoi[0]:picRoi[2]]
        img = cv.resize(img, (800, 600))

        # 获取界面状态
        getUI(img)

        # angle = getHookAngle(img)
        # if angle is not None:
        #     hookPoint = np.array(Locate.ORGIN)  # 出钩点
        #     HOOKAREA = np.array(Locate.HOOKAREA)  # 钩子区域(x1, y1), (x2, y2)
        #     startPoint = hookPoint
        #     v = np.array((np.cos(angle), np.sin(angle)))
        #     v = v * 50
        #     v = np.array(v, dtype=np.int16)
        #     endPoint = v + startPoint
        #     # print(startPoint, endPoint, v)
        #     cv.line(img, startPoint, endPoint, color=(0, 0, 255), thickness=5)
        # print(f'Hook angle:{angle}')
        #
        # # 获取物体
        # Items = getItems(img)
        # for i in gameItems:
        #     num = 0
        #     if Items[i]:
        #         for x, y, w, h in Items[i]:
        #             cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #             cv.putText(img, i+"-"+str(num), (x - int(w / 2), y - int(h / 2)), cv.FONT_HERSHEY_SIMPLEX,
        #                        fontScale=0.5, color=(0, 255, 0), thickness=2)
        #             num += 1
        #
        # # 获取数字
        # img = drawNumbers(img)

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
