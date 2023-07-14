"""
该文件存放对象识别的相关函数及参数
"""

import cv2 as cv
import numpy as np
import os
from config import cfg, sta
cfg_smp = cfg['sample']
import sys
from threading import Thread
import GetImg

from paddleocr import PaddleOCR
import logging
logging.disable(logging.DEBUG)
logging.disable(logging.WARNING)

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

def getNumber_ocr(ocrDetector, img):
    """
    数字识别_OCR
    :param img: 图片
    :return:
    """    
    results = []
    for x, y, w, h in cfg['ROI']['number']:
        ROI = img[y:y+h, x:x+w, :]
        # ROI = cv.resize(ROI, (int(w/2), int(h/2)))
        result = ocrDetector.ocr(ROI, cls=False)
        results += result
    # print("OCR results: ", results)
        
    # 解析结果
    infos = [None for _ in range(len(cfg_smp['number']['class']))]
    try:
        for i in results:
            firstWord = i[1][0][0] # 第一个字
            if firstWord == '金':    # 钱
                infos[sta.NUM_MONEY] = int(i[1][0][4:])
            elif firstWord == '$':    # 目标
                infos[sta.NUM_TARGET] = int(i[1][0][1:])
            elif firstWord == '时':  # 剩余时间
                infos[sta.NUM_TIME] = int(i[1][0][3:])
            elif firstWord == '第':   # 关卡号
                infos[sta.NUM_LEVEL] = int(i[1][0][1:-1])
    except Exception:
        pass
    return infos  


def drawNumbers_ocr(img, results):
    """
    画出所有的数字
    :param img:要画的图像
    :return: output - 画之后的图像
    """
    for i in range(len(cfg['draw']['number'])):
        if i >= len(results) or results[i] is None:
            break
        x, y = cfg['draw']['number'][i]
        cv.putText(img, cfg_smp['number']['class'][i]+": "+str(results[i]), (x, y), cv.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 255), thickness=2)

    return img


def getHookAngle(img):
    """
    获取钩子角度
    :param img: 画面图像
    :return: 钩子角度, None - 失败
    """
    DEBUG = 0  # 调试模式
    
    img_bg = cv.imread(os.path.join(cfg['path_root'], cfg_smp['hook']['background']))[:, :, :]
    x, y, w, h = cfg['ROI']['hook']
    ROI = img[y:y+h, x:x+w, :]

    difference = cv.absdiff(ROI, img_bg)
    thre = cv.threshold(cv.cvtColor(difference, cv.COLOR_BGR2GRAY), 30, 255, cv.THRESH_BINARY)[1]
    
    thre = cv.dilate(thre, np.ones((2,2), dtype=np.uint8), iterations=1)
    # thre = cv.erode(thre, np.ones((3,3), dtype=np.uint8), iterations=1)
    
    # 寻找轮廓
    contours, hierarchy = cv.findContours(thre, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    contours.sort(key=lambda c: cv.contourArea(c), reverse=True) # 按面积排序
    # 计算最大面积
    maxS = cv.contourArea(contours[0])
    
    # 无轮廓
    if contours is None or maxS > 400:# or maxS < 170:
        return None
    
    # 找外接圆
    (cx, cy), radius = cv.minEnclosingCircle(contours[0])
    # 把圆画出来
    # thre = cv.circle(cv.cvtColor(thre, cv.COLOR_GRAY2BGR), (int(cx), int(cy)), int(radius), (0, 0, 255), 3)
    # cv.imshow('thre', thre)
    
    # 计算角度
    hook_center = cfg_smp['hook']['centerLoc']
    angle = cv.fastAtan2((cy+y) - hook_center[1], (cx+x) - hook_center[0])
    angle = angle * np.pi / 180
    
    return angle


def getUI(HSV):
    """
    获取UI状态
    :param img: 图片
    :return: 响应的UI类别
    """
    loc = cfg_smp['UI']['locate']
    hue_sample = HSV[loc[1]][loc[0]]    # 采样点的色相值    
    hues = cfg_smp['UI']['hue']         # 各个状态对应的色相值
    state_amount = len(cfg_smp['UI']['status']) # 状态数
    for sta in range(state_amount-1):
        if hue_sample[0] == hues[sta][0] and \
           hue_sample[2] == hues[sta][2]:
            return sta
    return state_amount-1     # 其他界面(最后一个状态)



def main():
    # 若游戏未启动，则启动游戏
    if GetImg.get_window_view(cfg['name_flash']) is None:
        game = os.path.join(cfg['path_flash_game'], 'game.swf')
        assert os.path.isfile(game)==True, "\033[31m[ERROR] Game is not exist!\033[0m"
        engine = os.path.join(cfg['path_flash_engine'], 'flashplayer' if sys.platform == 'linux' else 'flashplayer.exe')
        assert os.path.isfile(game)==engine, "\033[31m[ERROR] Flashplayer is not exist!\033[0m"
        Thread(target=os.popen, args=(f'{engine} \"{game}\"', )).start()
        
        # 等待游戏启动
        import time
        time.sleep(0.1)
        
    ocrDetector = PaddleOCR(use_angle_cls=False, lang="ch") # ocr
    win = GetImg.Window() # 游戏窗口
    while 1:
        # 获取图片
        img = win.get_img()
        img = np.array(img)
        img_HSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)

        # # 获取界面状态
        # print('UI state: ', cfg_smp['UI']['status'][getUI(img_HSV)])

        # # 获取钩子信息
        # angle = getHookAngle(img)
        # if angle is not None:
        #     startPoint = np.array(cfg_smp['hook']['centerLoc'], dtype=np.int16)
        #     v = np.array((np.cos(angle), np.sin(angle)))
        #     v = v * 50
        #     v = np.array(v, dtype=np.int16)
        #     endPoint = v + startPoint
        #     # print(startPoint, endPoint, v)
        #     cv.line(img, tuple(startPoint), tuple(endPoint), color=(0, 0, 255), thickness=5)
        #     print(f'Hook angle:{angle}')
        
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
        
        # # 获取数字
        # numbers = getNumber_ocr(ocrDetector, img)
        # img = drawNumbers_ocr(img, numbers)

        # cv.imshow('demo', img)
        # key = cv.waitKey(50)
        # if key == 27:  # Esc退出
        #     cv.destroyAllWindows()
        #     break

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




if __name__ == '__main__':
    main()
