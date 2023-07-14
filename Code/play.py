"""
该文件负责游戏游玩相关操作，调用object_detection、Filter和GetImg脚本
"""
import cv2 as cv
import numpy as np
from config import cfg, sta

import GetImg
import object_detection, Filter
import time
import win32api, win32con, win32gui
import multiprocessing, threading
import time
import os, sys


DEBUG = True   # 调试模式

hookAllowErr = 5 * np.pi / 180    # 自动钩子的误差允许值，角度到达允许值时自动出钩
# 全局变量

def DEBUG_log(msg):
    """
    打印调试信息
    """
    if DEBUG:
        print("\033[34m" + str(msg) + "\033[0m")
        
        
def task_getImage(win, freq, q_img1, q_img2):
    """
    图片获取任务
    """
    deltaT_s = 1.0 / freq  # 周期时间
    while 1:
        t0 = time.time()
        
        img = win.get_img() # 获取图片
        # 清空队列
        while not q_img1.empty():
            q_img1.get()
        while not q_img2.empty():
            q_img2.get()
        # 图片放入队列
        q_img1.put(img) 
        q_img2.put(img) 
        
        # 频率控制
        t1 = time.time()
        spendTime = t1 - t0
        if spendTime < deltaT_s:
            time.sleep(deltaT_s - spendTime)
            
def task_number(freq_ocr, q_img2, q_number, UIsta):    # 数字更新任务
    import time
    from config import sta
    import logging
    logging.disable(logging.DEBUG)
    logging.disable(logging.WARNING)
    from paddleocr import PaddleOCR
    
    ocrDetector = PaddleOCR(use_angle_cls=False, lang="ch") # ocr
    deltaT_s = 1.0 / freq_ocr   # 周期时间
    
    while 1:
        t0 = time.time()
        # 界面状态分析
        # 1. 游戏中
        if UIsta.value == sta.UI_PLAY:
            img = q_img2.get() # 获取图像
            t0 = time.time()
            numbers = object_detection.getNumber_ocr(ocrDetector, img)
            
            # 时间补偿
            if not None in numbers:
                T = round(numbers[sta.NUM_TIME] - (time.time()-t0)*2/3)
                numbers[sta.NUM_TIME] = T if T > 0 else 0
            
            # 清空队列
            while not q_number.empty():
                q_number.get()
            
            # 放入队列
            q_number.put(numbers)
            
        # 频率控制
        t1 = time.time()
        spendTime = t1 - t0
        if spendTime < deltaT_s:
            time.sleep(deltaT_s - spendTime)


class GamePlay:
    """
    游戏游玩的类
    """
    win = None # 游戏窗口控制器
    winWait = 0     # 每一帧之间的等待时间(ms)
    showImg = None  # 显示窗口的名称
    
    # 游戏状态观测器
    class Status:
        UI = multiprocessing.Value("d", len(cfg['sample']['UI']['status']))   # 界面状态
        hook_angle = -255.0 # 钩子当前角度
        hook_clockwise = True  # 钩子方向，True-顺时针, False逆时针
        hook_available = True  # 钩子是否可控制（即是否在区域内未发射），True-可使用，False-不可使用
        hook_predict_angle = -255.0 # 预测的钩子下一时刻角度
        money_now = 0
        money_target = 0
        time = 0
        level = 0
        items = {} # 物品信息
        
        # 读取函数
        def get_UI(self):
            return self.UI.value
        def get_hook(self):
            return [self.hook_angle, self.hook_clockwise, self.hook_available]
        def get_hook_angle(self):
            return self.hook_angle
        def get_hook_angle_predict(self):
            return self.hook_predict_angle
        def get_money(self):
            return self.money_now
        def get_target(self):
            return self.money_target
        def get_time(self):
            return self.time
        def get_level(self):
            return self.level
        def get_item(self):
            return self.items
        
        def __init__(self, win:GetImg.Window, freq_fresh=cfg['sample']['freq']['default'], freq_ocr=cfg['sample']['freq']['ocr']) -> None:
            # 窗口句柄
            self.win = win     
            
            # 更新频率
            self.freq_fresh = freq_fresh
            self.freq_ocr = freq_ocr
            
            # 创建任务       
            self.q_img1 = multiprocessing.Queue() # 图片消息队列(传统处理)
            self.q_img2 = multiprocessing.Queue() # 图片消息队列(OCR)
            self.q_number = multiprocessing.Queue() # 数字识别消息队列(OCR)
            # 图片装载进程
            self.task_getImg = multiprocessing.Process(target=task_getImage, args=(self.win, self.freq_fresh, self.q_img1, self.q_img2)) 
            # 数字识别进程
            self.task_number = multiprocessing.Process(target=task_number, args=(self.freq_ocr, self.q_img2, self.q_number, self.UI)) 
            # 状态更新线程 
            self.task_fresh = threading.Thread(target=GamePlay.Status.__task_fresh, args=(self,))   
            
            self.task_fresh.setDaemon(True)
            self.task_fresh.start()
            
        def __del__(self):
            # 在析构时结束各程序
            self.task_number.terminate()
            self.task_getImg.terminate()
            self.task_number.join()
            self.task_getImg.join()

            
        def __task_fresh(self):     # 参数更新任务
            UI_last = sta.UI_OTHER # 上次的UI状态
            deltaT_s = 1.0 / self.freq_fresh  # 周期时间
            hook_angle_last = -255.0 # 上次的钩子角度
            hook_a_last = 0 # 上次的钩子加速度
            hook_gama = 0   # 修正因子
            hook_learningRate = 0.92 # 修正因子的学习率

            # 启动数字识别任务
            self.task_getImg.start()
            self.task_number.start()
                
            while 1:
                t0 = time.time()
                
                # 获取图片
                img = self.q_img1.get()
                if img.any():
                    img_HSV = cv.cvtColor(img, cv.COLOR_BGR2HSV) # 获取HSV图片
                    self.UI.value = object_detection.getUI(img_HSV) # UI判断
                    
                    # 界面状态分析
                    # 1. 游戏中
                    if self.UI.value == sta.UI_PLAY:
                        # 获取各部分信息
                        # 1.1 物体信息
                        self.items = object_detection.getItems(img)
                        # 1.2 钩子信息
                        newState = object_detection.getHookAngle(img)
                        # a. 游戏刚开始/钩子不在区域内， 则初始化钩子状态
                        if newState is None or UI_last != self.UI.value:
                            self.hook_available = False
                            self.hook_angle == -255
                            self.hook_clockwise = True
                            UI_last = self.UI.value
                        # b. 正常情况
                        else:
                            self.hook_available = True
                            if self.hook_angle == -255:     # 初状态时初始化
                                self.hook_angle = newState
                                hook_angle_last = newState
                            elif self.hook_angle != newState:
                                # 预测通过加速度预测
                                v = newState - hook_angle_last
                                hook_gama = hook_gama + hook_learningRate*(newState - self.hook_predict_angle) # 计算新的修正因子
                                v_predit = v + hook_a_last + 2*hook_gama # 预测速度
                                self.hook_predict_angle = newState + v_predit # 预测角度
                                
                                hook_a_last = (v + v_predit) / 2 # 预测加速度
                                hook_angle_last = self.hook_angle
                                self.hook_angle = newState
                                
                                # # 分段权值滤波
                                # if abs((np.pi / 2) - newState) < (20 * np.pi / 180):
                                #     weight = 1.5
                                # else:
                                #     weight = 0.7
                                # hook_angle_last = self.hook_angle # 上次观测的钩子角度
                                # self.hook_angle = (newState * weight + hook_angle_last * (1 - weight)) / 10
                                # 顺/逆时针判断(摆动小于1°则不改变)
                                if abs(hook_angle_last - self.hook_angle) > (np.pi/180):
                                    if hook_angle_last < self.hook_angle:
                                        self.hook_clockwise = 0
                                    else:
                                        self.hook_clockwise = 1
                        # 1.3 数字信息
                        if not self.q_number.empty():
                            nums = self.q_number.get()
                            if not None in nums:
                                self.money_now = nums[sta.NUM_MONEY]
                                self.money_target = nums[sta.NUM_TARGET]
                                self.time = nums[sta.NUM_TIME]
                                self.level = nums[sta.NUM_LEVEL]
                # 频率控制
                t1 = time.time()
                spendTime = t1 - t0
                if spendTime < deltaT_s:
                    time.sleep(deltaT_s - spendTime)

    class Tasks:
        hook_thread = None   # 自动出勾线程

    def __init__(self, winName=cfg['name_flash'], ctrFreq=cfg['sample']['freq']['ctrl'], showImg='detection'):
        """
        :param winName: 游戏窗口名称
        :param frequency: 更新频率
        :param showFlag: 显示画面的名称，若为None则不显示
        """
        DEBUG_log("[Player] Starting...")
        
        # 若游戏未启动，则先启动游戏
        if GetImg.get_window_view(winName) is None:
            print("\033[31m[ERROR] Game is not opened! Trying to open it...\033[0m")
            
            game = os.path.join(cfg['path_flash_game'], 'game.swf')
            assert os.path.isfile(game)==True, "\033[31m[ERROR] Game is not exist! Path: " + game + "\033[0m"
            engine = os.path.join(cfg['path_flash_engine'], 'flashplayer' if sys.platform == 'linux' else 'flashplayer.exe')
            assert os.path.isfile(engine)==True, "\033[31m[ERROR] Flashplayer is not exist! Path: " + engine + "\033[0m"
            multiprocessing.Process(target=os.popen, args=(f'{engine} \"{game}\"', )).start()
            
            # 等待游戏启动
            waitTime = time.time() + 10.0 # 启动等待最多10秒
            while time.time() < waitTime:
                if GetImg.get_window_view(winName):
                    print("\033[31m[ERROR/FIX] Game is opened successfully! \033[0m")
                    break
                time.sleep(0.1)
            else:
                assert GetImg.get_window_view(winName) is not None, "\033[31m[ERROR] Filed to open game!\033[0m"

        self.winName = winName
        self.win = GetImg.Window(winName)# 游戏窗口控制器
        self.deltaT_s = 1.0 / ctrFreq # 控制频率

        # 录入结果显示窗口名称
        if type(showImg) is str:
            self.showImg = showImg         
            
        # 创建状态观测器
        self.sta = self.Status(self.win)        

    def show(self):
        """
        状态更新任务
        :param angle: 当前角度
        :param clockwise: 是否顺时针
        :param hook_extern: 钩子是否存在于区域内
        :param ui_sta: UI状态
        """
        if self.sta.get_UI() == sta.UI_PLAY:
            # 结果显示
            if type(self.showImg) is str:
                # 获取图像
                img = self.win.get_img()
                
                # 图片获取失败
                if not img.any():
                    return
                
                # 显示识别结果-钩子绘制
                if self.sta.get_hook()[0] is not None:
                    startPoint = np.array(cfg['sample']['hook']['centerLoc'], dtype=np.int16) # 出钩点
                    v = np.array((np.cos(self.sta.get_hook_angle()), np.sin(self.sta.get_hook_angle())))
                    v = v * 50
                    v = np.array(v, dtype=np.int16)
                    endPoint = v + startPoint
                    cv.line(img, tuple(startPoint), tuple(endPoint), color=(0, 0, 255), thickness=5)
                    
                # 显示识别结果-获取物体
                Items = object_detection.getItems(img)
                for i in object_detection.gameItems:
                    num = 0
                    if Items[i]:
                        for x, y, w, h in Items[i]:
                            cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            cv.putText(img, i + "-" + str(num), (x - int(w / 2), y - int(h / 2)),
                                        cv.FONT_HERSHEY_SIMPLEX,
                                        fontScale=0.5, color=(0, 255, 0), thickness=2)
                            num += 1
                # 显示识别结果-获取数字
                x, y = cfg['draw']['number'][sta.NUM_MONEY]
                cv.putText(img, "money: "+str(self.sta.get_money()), (x, y), cv.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 255), thickness=2)
                x, y = cfg['draw']['number'][sta.NUM_TARGET]
                cv.putText(img, "target: "+str(self.sta.get_target()), (x, y), cv.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 255), thickness=2)
                x, y = cfg['draw']['number'][sta.NUM_TIME]
                cv.putText(img, "time: "+str(self.sta.get_time()), (x, y), cv.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 255), thickness=2)
                x, y = cfg['draw']['number'][sta.NUM_LEVEL]
                cv.putText(img, "level: "+str(self.sta.get_level()), (x, y), cv.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 255), thickness=2)
                # 显示识别结果-钩子角度
                x, y = cfg['draw']['hook']
                cv.putText(img, "Angel:%03.1f" % (self.sta.get_hook_angle()*180/np.pi), (x, y), cv.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 255), thickness=2)

                # 窗口显示
                cv.imshow(self.showImg, img)
                
                # 开启画面，打开画面以及opencv的鼠标点击事件
                cv.setMouseCallback(self.showImg, self.__mouse_click_callback)

    def put_explosive(self):
        """
        丢炸药
        """
        self.win.press_key(win32con.VK_UP)

    def put_hook(self):
        """
        出钩
        """
        self.win.press_key(win32con.VK_DOWN)

    def __task_put_hook_auto(self, target):
        """
        出勾任务
        :param target: 目标角度
        :return True - 成功, False - 失败
        """
        stopTime = time.time() + 10.0
        
        while time.time() < stopTime:
            # 满足出勾条件(快到角度的时候出钩)
            if abs(self.sta.get_hook_angle_predict() - target) < hookAllowErr:
                self.put_hook()
                DEBUG_log("[DEBUG/GamePlay]\tAuto put hook succeed!")
                return True
            time.sleep(1/self.deltaT_s)
        else:   # 超时
            DEBUG_log("[DEBUG/GamePlay]\tAuto put hook timeout!")
            return False

    def put_hook_auto(self, angle):
        """
        自动出钩
        :param angle: 出钩的角度
        :return True - 任务添加成功, False - 已有任务
        """       
        if self.Tasks.hook_thread is None: 
            self.Tasks.hook_thread = threading.Thread(target=self.__task_put_hook_auto, args=(angle,))
            
            self.Tasks.hook_thread.setDaemon(True)
            self.Tasks.hook_thread.start()
            return True
        else:
            return False

    def __mouse_click_callback(self, event, x, y, flags, param):
        """
        鼠标回调函数（用于用户操控）
        现象：点击某处，则向某处出勾
        """
        if event == cv.EVENT_LBUTTONDOWN:   # 左键点击
            p_zero = cfg['sample']['hook']['centerLoc']    # 钩子原点
            p_cursor = (x, y)
            p_zero = np.array(p_zero)
            p_cursor = np.array(p_cursor)

            angle = np.arccos((p_cursor[0] - p_zero[0]) / (np.linalg.norm(p_cursor - p_zero)))

            flag = self.put_hook_auto(angle)
            if flag:
                DEBUG_log("[DEBUG/GamePlay]\tPut hook to the locate of cursor!")
            else:
                DEBUG_log("[DEBUG/GamePlay]\tFailed to put hook to the locate of cursor!")
            

    def ui_click(self):
        """
        页面点击
        :return : True - 成功， False - 失败
        :ps :   主界面 → 挖矿界面
                商店 → 下一步
                结算 → 主界面
        """
        flag_twice = False  # 双击
        ui_sta = self.sta.get_UI()
        if ui_sta == sta.UI_TITLE:   # 主界面
            x, y = (153, 164)
            flag_twice = True
        elif ui_sta == sta.UI_SHOP: # 商店
            x, y = (644, 115)
            flag_twice = True
        elif ui_sta == sta.UI_END:  # 结算
            x, y = (111, 475)
            flag_twice = True
        else:
            return False

        DEBUG_log("[DEBUG/GamePlay]\tClick to the next UI!")
            
        # 点击事件
        self.win.check((x, y), flag_twice)
        return True



if __name__ == "__main__":
    player = GamePlay()
    while 1:
        player.show()
        player.ui_click()
        # player.put_hook_auto(np.pi / 2)

        key = cv.waitKey(100)
        if key == 27:  # Esc退出
            cv.destroyAllWindows()
            break
    cv.destroyAllWindows()
    
