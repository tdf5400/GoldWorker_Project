"""
该文件存放滤波器相关算法
"""
import numpy as np
import time


class RLS:
    """
    递推最小二乘估计
    :ps :只能估计确定性的常值向量
    """
    X_k = 0     # X^ k
    X_kp = 0    # X^ k+1
    P_k = 0     # P k
    P_kp = 0    # P k+1
    W = 0       # 加权阵

    def __init__(self, X0, W):
        """
        :param X0:  估计的初值
        :param W:   权重，若 W=R^-1，则为马尔科夫估计
        """
        self.P_k  = self.P_kp = 1
        self.X_k = self.X_kp = X0
        self.W = W

    def update(self, Z_kp):
        """
        状态更新
        :param Z_kp: 新观测的状态
        """
        Pk, W = self.P_kp, self.W
        Pkp = Pk - Pk * Pk / ((1/W) + Pk)
        Xk = self.X_kp
        Xkp = Xk + Pkp * W * (Z_kp - Xk)

        self.P_k = self.P_kp
        self.P_kp = Pkp
        self.X_k = self.X_kp
        self.X_kp = Xkp

    def value(self):
        """
        数值获取
        """
        return self.X_kp


if __name__ == "__main__":
    val_real = 5
    system = RLS(4.5, 1/2)
    while 1:
        val_get = val_real + np.random.normal(0, 2)     # 包含高斯噪声的观测值
        system.update(val_get)
        print("get: {}, estimate: {}".format(val_get, system.value()))
        time.sleep(1)