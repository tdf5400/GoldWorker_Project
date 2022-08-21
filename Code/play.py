"""
该文件负责游戏游玩相关操作，调用object_detection、Filter和GetImg脚本
"""
import cv2 as cv
import numpy as np

import GetImg
import object_detection, Filter
import time, threading, multiprocessing
import win32api, win32con, win32gui


DEBUG = 1   # 调试模式

hookAllowErr = 4.5 * np.pi / 180    # 自动钩子的误差允许值，角度到达允许值时自动出钩
# 全局变量
global player   # 用户操作对象，用于鼠标事件


class GamePlay:
    """
    游戏游玩的类
    """
    winHwnd = None  # 游戏窗口句柄
    ctrHwnd = None  # 操控窗口句柄
    winWait = 0     # 每一帧之间的等待时间(ms)
    showImg = None  # 显示窗口名
    UISta = multiprocessing.Value("d", object_detection.Sta.UI_OTHER)   # 界面状态

    class Hook:
        angle_last = multiprocessing.Value("d", -255.0)      # 上次观测的钩子角度
        angle = multiprocessing.Value("d", -255.0)           # 当前的钩子角度
        clockwise = multiprocessing.Value("i", 1)       # 当前是否顺时针, 1-顺时针, 0逆时针
        extern = multiprocessing.Value("i", 1)  # 是否在区域内

    class Tasks:
        put_hook = None     # 自动出勾任务，非None时，状态更新后会进行一次出勾检测
        put_hook_timer = 0  # 自动出勾任务_超时计时器
        sta_thread = None   # 状态更新线程

    def __init__(self, windowsName="Adobe", frequency=100, showImg=None):
        """, 100, "output"
        :param windowsName: 游戏窗口名称
        :param frequency: 更新频率
        :param showFlag: 显示画面的句柄，若为None则不显示
        """
        self.name = windowsName + "_GamePlay"
        self.winHwnd = GetImg.get_window_view(windowsName)  # 游戏窗口句柄
        self.ctrHwnd = GetImg.get_window_ctr(self.winHwnd)  # 控制句柄
        showImg = "output"
        # 窗口resize
        win32gui.SetWindowPos(self.winHwnd, win32con.HWND_NOTOPMOST, 0, 0, 800, 600, win32con.SHOW_ICONWINDOW)

        if frequency:
            self.winWait = int(1 / frequency * 1000)
        else:
            self.winWait = 0

        if type(showImg) is str:
            self.showImg = showImg

        # 创建状态更新进程
        self.Tasks.sta_thread = multiprocessing.Process(target=self.task_state, args=(self.Hook.angle,
                                                                                      self.Hook.clockwise,
                                                                                      self.Hook.extern,
                                                                                      self.UISta), daemon=True)
        self.Tasks.sta_thread.start()

    def task_state(self, angle, clockwise, hook_extern, ui_sta):
        """
        状态更新任务
        :param angle: 当前角度
        :param clockwise: 是否顺时针
        :param hook_extern: 钩子是否存在于区域内
        :param ui_sta: UI状态
        """
        angle_last = angle.value
        ui_sta_last = ui_sta.value

        # 开启画面，打开画面以及opencv的鼠标点击事件
        if type(self.showImg) is str:
            cv.imshow(self.showImg, 0)
            cv.resizeWindow(self.showImg, 800, 600)
            cv.setMouseCallback(self.showImg, self.__mouse_click_callback)

        while 1:
            img = GetImg.get_img(self.winHwnd)
            pic_roi = object_detection.picRoi    # 获取有效图像
            img = img[pic_roi[1]:pic_roi[3], pic_roi[0]:pic_roi[2]]
            img = cv.resize(img, (800, 600))

            # 界面状态更新
            ui_sta.value = object_detection.getUI(img)
            if ui_sta.value == object_detection.Sta.UI_PLAY:   # 游戏中
                # 钩子状态更新（角度发生变化时）
                newState = object_detection.getHookAngle(img)
                # 游戏刚开始/钩子不在区域内则初始化钩子状态
                if newState is None or ui_sta_last != ui_sta.value:
                    hook_extern = False
                    angle.value == -255
                    clockwise.value = 1
                else:
                    hook_extern = True
                    if angle.value == -255:     # 初状态时初始化
                        angle.value = newState
                    elif angle.value != newState:
                        # 分段权值滤波
                        if abs((np.pi / 2) - newState) < (20 * np.pi / 180):
                            weight = 9
                        else:
                            weight = 8
                        angle_last = angle.value
                        angle.value = (newState * weight + angle_last * (10 - weight)) / 10
                        # 顺/逆时针判断(摆动小于1°则不改变)
                        if abs(angle_last - angle.value) > (np.pi/180):
                            if angle_last < angle.value:
                                clockwise.value = 0
                            else:
                                clockwise.value = 1

                # 自动出勾任务
                if self.Tasks.put_hook is not None:
                    self.__task_put_hook_auto(self.Tasks.put_hook, angle.value)

                # 结果显示
                if type(self.showImg) is str:
                    # 显示识别结果
                    # 显示识别结果-钩子绘制
                    if angle.value is not None:
                        hookPoint = np.array(object_detection.Locate.ORGIN)  # 出钩点
                        HOOKAREA = np.array(object_detection.Locate.HOOKAREA)  # 钩子区域(x1, y1), (x2, y2)
                        startPoint = hookPoint
                        v = np.array((np.cos(angle.value), np.sin(angle.value)))
                        v = v * 50
                        v = np.array(v, dtype=np.int16)
                        endPoint = v + startPoint
                        cv.line(img, startPoint, endPoint, color=(0, 0, 255), thickness=5)
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
                    img = object_detection.drawNumbers(img)
                    # 显示识别结果-钩子角度
                    cv.putText(img, "Angel:{}".format(angle.value), (0, 50), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            ui_sta_last = ui_sta.value

            cv.imshow(self.showImg, img)
            cv.waitKey(self.winWait)

    def put_explosive(self):
        """
        丢炸药
        """
        win32api.PostMessage(self.ctrHwnd, win32con.WM_KEYDOWN, win32con.VK_UP, 0)
        win32api.PostMessage(self.ctrHwnd, win32con.WM_KEYUP, win32con.VK_UP, 0)

    def put_hook(self):
        """
        出钩
        """
        win32api.PostMessage(self.ctrHwnd, win32con.WM_KEYDOWN, win32con.VK_DOWN, 0)
        win32api.PostMessage(self.ctrHwnd, win32con.WM_KEYUP, win32con.VK_DOWN, 0)

    def __task_put_hook_auto(self, target, now):
        """
        出勾任务
        :param target: 目标角度
        :param now: 当前角度
        """
        # 满足出勾条件(快到角度的时候出钩)
        if abs(now - target) < hookAllowErr:
            self.put_hook()
            if DEBUG:
                print("[DEBUG/GamePlay]\tAuto put hook succeed!")
            self.Tasks.put_hook = None
            self.Tasks.put_hook_timer = 0
            return False
        # 不满足
        if time.time() - self.Tasks.put_hook_timer > 10:
            if DEBUG:
                print("[DEBUG/GamePlay]\tAuto put hook timeout!")
            self.Tasks.put_hook = None
            self.Tasks.put_hook_timer = 0
            return True

    def put_hook_auto(self, angle):
        """
        自动出钩
        :param angle: 出钩的角度
        :return True - 失败, False - 成功
        """
        if self.Tasks.put_hook is None:
            self.Tasks.put_hook = angle     # 把变量改为非None
            self.Tasks.put_hook_timer = time.time()     # 启动超时检测

    def __mouse_click_callback(self, event, x, y, flags, param):
        """
        鼠标回调函数（用于用户操控）
        现象：点击某处，则向某处出勾
        """
        if event == cv.EVENT_LBUTTONDOWN:   # 左键点击
            p_zero = object_detection.Locate.ORGIN    # 钩子原点
            p_cursor = (x, y)
            p_zero = np.array(p_zero)
            p_cursor = np.array(p_cursor)

            angle = np.arccos((p_cursor[0] - p_zero[0]) / (np.linalg.norm(p_cursor - p_zero)))

            if DEBUG:
                print("[DEBUG/GamePlay]\tPut hook to the locate of cursor!")
            self.put_hook_auto(angle)

    def get_hook_sta(self):
        """
        获取钩子状态
        :return : True - 在区域内， False - 在区域外
        """
        return self.Hook.extern.value

    def ui_get_sta(self):
        """
        获取UI状态
        :return @objection_detection.Sta: 状态值
        """
        return self.UISta.value

    def ui_click(self):
        """
        页面点击
        :return : True - 成功， False - 失败
        :ps :   主界面 → 挖矿界面
                商店 → 下一步
                结算 → 主界面
        """
        flag_twice = False  # 双击
        ui_sta = self.ui_get_sta()
        if ui_sta == object_detection.Sta.UI_MAIN:   # 主界面
            x, y = (153, 164)
            flag_twice = True
        elif ui_sta == object_detection.Sta.UI_SHOP: # 商店
            x, y = (644, 115)
            flag_twice = True
        elif ui_sta == object_detection.Sta.UI_END:  # 结算
            x, y = (111, 475)
            flag_twice = True
        else:
            return False

        if DEBUG:
            print("[DEBUG/GamePlay]\tClick to the next UI!")

        action = win32api.MAKELONG(x, y)
        win32api.SendMessage(self.ctrHwnd, win32con.WM_LBUTTONDOWN, win32con.MK_LBUTTON, action)
        win32api.SendMessage(self.ctrHwnd, win32con.WM_LBUTTONUP, win32con.MK_LBUTTON, action)
        if flag_twice:  # 双击
            win32api.SendMessage(self.ctrHwnd, win32con.WM_LBUTTONDOWN, win32con.MK_LBUTTON, action)
            win32api.SendMessage(self.ctrHwnd, win32con.WM_LBUTTONUP, win32con.MK_LBUTTON, action)
        return True


class MyManager(multiprocessing.managers.BaseManager):
    pass


MyManager.register('Manage_GamePlay', GamePlay)


if __name__ == "__main__":
    global player
    with MyManager as manager:
        # player = GamePlay("Adobe", 20)
        player = manager.Manage_GamePlay()
        # player = manager.Manage_GamePlay("Adobe", 100, "output")
        while 1:
            time.sleep(1)
            player.ui_click()
            # player.put_hook_auto(np.pi / 2)
            pass

        cv.waitKey(0)
        cv.destroyAllWindows()
