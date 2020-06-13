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
goldAreaM2L = 700       # 中金块与大金块区分阈值

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
    cv.imshow('morph', morph)

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

tmp = '73.jpg'
moneyColorThre = (0, 19, 0), (44, 157, 46)
moneyAreaLimit = (500, 2000)
moneyFontWidth = 13
moneyFontArea = (0, 0, 0, 232,
                    0, 0, 0,
                    0, 0, 0)      # 字体像素值 0-9
# moneyFontThre = 5   # 面积差为moneyFontThre以内则判断为该值
def getMoney(img):
    """
    显示金钱
    :param img:
    :return: 金钱数
    """
    assert (img.shape == (600, 800, 3)), "[error] resolution should be 800x600!"

    thre = cv.inRange(img, moneyColorThre[0], moneyColorThre[1])
    thre[:][scoreBoard_height:] = 0  # ROI

    kernel = np.ones((8, 8), dtype=np.uint8)
    morph = cv.morphologyEx(thre, cv.MORPH_CLOSE, kernel=kernel)
    # cv.imshow('morph', morph)

    # 检测
    money = 0
    contours, hier = cv.findContours(morph, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    for c in contours:
        x, y, w, h = cv.boundingRect(c)
        y -= 5  # 往上挪一点
        h += 7
        area = w * h

        if moneyAreaLimit[0] < area < moneyAreaLimit[1]:
            x0, x1 = x, x+w
            fonts = []
            fontMorph = cv.morphologyEx(thre, cv.MORPH_ERODE, np.ones((2,2),np.uint8))
            fontImg = fontMorph[y:y + h, x0:x1]

            cv.imshow('morph', fontImg)
            while x1 > x0:
                minArea = area  # 取最小差值的数
                minNum = None
                fontArea = 0#np.unique(fontImg, return_counts=True)[1][1]  # 白色像素数量
                for i in np.matrix.tolist(fontImg):
                    for j in i:
                        if j == 255:
                            fontArea += 1
                print(fontArea)#np.unique(fontImg, return_counts=True))
                img[y:y+h, x1-100-moneyFontWidth:x1] = (255, 255, 255)
                cv.imshow('img', img)

                for i in range(len(moneyFontArea)):
                    duce = abs(fontArea - moneyFontArea[i])
                    if duce < minArea:
                        minArea = duce
                        minNum = i
                fonts.append(minNum)
                x1 -= moneyFontWidth
                print(minNum, fontArea)


            if not len(fonts):
                print('fonts is []!')
                return 0

            for i in range(len(fonts)):
                money += fonts[i] * (10**i)
            print(f'money:{money}')

            return money

            # if i is not 3:
            #     i += 1
            #     continue
            cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv.putText(img, str(area), (x - int(w / 2), y + 20*int(h / 2)), cv.FONT_HERSHEY_PLAIN,
                       fontScale=1, color=(0, 255, 0), thickness=2)
            cv.imshow('img', img)
            # return 0
    cv.imshow('img', img)



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
            'pig': pig, 'diamondPig': diamondPig
            }




if __name__ == '__main__':
    # import GetImg
    # import win32gui, win32con
    # hwnd = GetImg.get_window("game.swf")
    # win32gui.SetWindowPos(hwnd, win32con.HWND_NOTOPMOST, 0, 0, 800, 600, win32con.SHOW_ICONWINDOW)
    # while 1:
    #     img = GetImg.get_img(hwnd)
    #     # img = cv.imread('../dataset/84.jpg')
    #     img = cv.resize(img, (800, 600))
    #     print(img.shape)
    #     Items = getItems(img)
    #     for i in gameItems:
    #         num = 0
    #         if Items[i]:
    #             for x, y, w, h in Items[i]:
    #                 cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #                 cv.putText(img, i+"-"+str(num), (x - int(w / 2), y - int(h / 2)), cv.FONT_HERSHEY_SIMPLEX,
    #                            fontScale=0.5, color=(0, 255, 0), thickness=2)
    #                 num += 1
    #     cv.imshow('demo', img)
    #
    #     key = cv.waitKey(50)
    #     if key == 27:  # Esc退出
    #         cv.destroyAllWindows()
    #         break

    img = cv.imread('../dataset/'+tmp)
    img = cv.resize(img, (800, 600))
    cv.imshow('img', img)
    # getGold(img)
    # getStone(img)
    # getPig(img)
    # getDiamond(img)
    # getPack(img)
    # getBucket(img)
    # getBone(img)
    getMoney(img)
    cv.waitKey(0)

