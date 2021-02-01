"""
该文件负责游戏游玩相关操作，调用object——detection和GetImg文件
"""
import cv2 as cv
import GetImg
import object_detection
import threading
import time
import win32api, win32con, win32gui

DEBUG = 1   # 调试模式


hookAllowErr = 5    # 自动钩子的误差允许值，角度到达允许值时自动出钩


class GamePlay:
    """
    游戏游玩的类
    """
    winHwnd = None  # 游戏窗口句柄
    ctrHwnd = None  # 操控窗口句柄
    winWait = 0     # 每一帧之间的等待时间(s)

    sta_thread = None # 状态更新线程

    class Hook:
        angel_last = None      # 钩子上一个角度
        angel = None           # 钩子当前角度
        clockwise = True    # 是否顺时针

    def __init__(self, windowsName, frequency):
        """
        :param windowsName: 游戏窗口名称
        :param frequency: 更新频率
        """
        self.winHwnd = GetImg.get_window_view(windowsName)
        self.ctrHwnd = GetImg.get_window_ctr(self.winHwnd)
        if frequency:
            self.winWait = int(1 / frequency)
        else:
            self.winWait = 0

        # 创建状态更新线程
        self.sta_thread = threading.Thread(target=self.task_state, name="state refresh")
        self.sta_thread.start()

    def task_state(self):
        """
        状态更新任务
        """
        img = GetImg.get_img(self.winHwnd)
        img = cv.resize(img, (800, 600))

        while 1:
            img = GetImg.get_img(self.winHwnd)
            img = cv.resize(img, (800, 600))

            # 钩子状态更新（角度发生变化时）
            newState = object_detection.getHookAngel(img)
            if newState is not None:
                if self.Hook.angel is None:     # 初状态时初始化
                    self.Hook.angel = newState
                elif self.Hook.angel != newState:
                    self.Hook.angel_last = self.Hook.angel
                    self.Hook.angel = newState
                    # 顺/逆时针判断
                    if self.Hook.angel_last < self.Hook.angel:
                        self.Hook.clockwise = False
                    else:
                        self.Hook.clockwise = True

            time.sleep(self.winWait)

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

    def put_hook_auto(self, angel):
        """
        自动出钩
        :param angel: 出钩的角度
        :return True - 失败, False - 成功
        """
        start_time = time.time()  # 获取当前时间，用于超时判断
        # 等待数据类型正确
        while self.Hook.angel is None:
            time.sleep(self.winWait)
            # 超时判断
            if (time.time() - start_time) > 10:
                if DEBUG:
                    print("[DEBUG/GamePlay]\tAuto put hook type err!")
                return True

        while abs(self.Hook.angel - angel) > hookAllowErr:
            time.sleep(self.winWait)
            # 超时判断
            if (time.time() - start_time) > 10:
                if DEBUG:
                    print("[DEBUG/GamePlay]\tAuto put hook timeout!")
                return True

        self.put_hook()
        if DEBUG:
            print("[DEBUG/GamePlay]\tAuto put hook succeed!")
        return False


if __name__ == "__main__":
    player = GamePlay("game.swf", 5)
    player.put_hook_auto(-90)

    cv.waitKey(0)
    cv.destroyAllWindows()
