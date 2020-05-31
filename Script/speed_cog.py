# -*- coding: utf-8 -*-
import cv2
import numpy as np
import time
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

set_parameter = {
                 "input_video" : 0,
                 "th_binary_1" : 50,
                 "th_binary_2" : 255,
                 "th_binary_3" : cv2.THRESH_BINARY_INV,
                 "coefficient" : 1,
                 "areasize"    : 0.0008,
                 "height"      : 240,
                 "width"       : 320
                 }


class Horizontal_speed__estimation_by_com:
    # input_video : 入力映像. device number or file path.
    # th_binary_1 : 2値化閾値1.
    # th_binary_2 : 2値化閾値2.
    # th_binary_3 : 2値化処理方法.
    # coefficient : [pixels/frame(s)]を[m/s]に変換するための係数.
    # areasize    : 輪郭の採用範囲.
    # height      : 入力画像のリサイズ高さ.
    # width       : 入力画像のリサイズ幅.
    
    def __init__(self, input_video,
                       th_binary_1, th_binary_2, th_binary_3,
                       coefficient,
                       areasize,
                       height=None, width=None):
        self.th_binary_1  = th_binary_1
        self.th_binary_2  = th_binary_2
        self.th_binary_3  = th_binary_3
        self.coefficient  = coefficient
        self.height       = height
        self.width        = width
        
        self.q0 = 1.0
        self.q1 = 0.0
        self.q2 = 0.0
        self.q3 = 0.0
        
        self.org = cv2.VideoCapture(input_video)
        self.end_flag, self.frame_org = self.org.read()
        
        _height, _width, _ = self.frame_org.shape
        print("<Original>\n "
              "Height :", _height, "\n",
              "Width  :", _width)
        
        if height is None:
            self.height = _height
        if width is None:
            self.width = _width
        self.areasize     = height*width * areasize
        print("<Setting>\n "
              "Height      :", self.height,      "\n",
              "Width       :", self.width,       "\n",
              "th_binary_1 :", self.th_binary_1, "\n",
              "th_binary_2 :", self.th_binary_2, "\n",
              "th_binary_3 :", self.th_binary_3, "\n",
              "Areasize    :", self.areasize,    "\n")
        
        cv2.namedWindow("org")
        cv2.namedWindow("bin")
        
    def speed_estimation(self):
        com_sum  = []
        time_cs  = []
        cnt_cs   = []
        vel      = []
        time_vel = []
        cnt_vel  = []
        self.cnt = 1
        
        print("Speed estimation start.")
        time_0 = time.time()
        
        while self.end_flag == True:
            self.frame_org = cv2.resize(self.frame_org,
                                        dsize=(self.width, self.height))
            frame_bin = cv2.cvtColor(self.frame_org, cv2.COLOR_BGR2GRAY)
            _, frame_bin = cv2.threshold(frame_bin, self.th_binary_1,
                                                    self.th_binary_2,
                                                    self.th_binary_3)
            self.dt = time.time() - time_0
            # 重心座標算出
            contours, _ = cv2.findContours(frame_bin, 
                                           cv2.RETR_EXTERNAL, 
                                           cv2.CHAIN_APPROX_TC89_L1)
            #print(_)
            com_org = []
            for c in contours:
                _moment = cv2.moments(c)
                #print(_moment["m00"])
                if(_moment["m00"] != 0):
                    if(_moment["m00"] > self.areasize):
                        _com_x = int(_moment["m10"]/_moment["m00"])
                        _com_y = int(_moment["m01"]/_moment["m00"])
                        com_org.append((_com_x, _com_y))
            
            if com_org:                
                for com in com_org:  # com_org描写
                    self.frame_org = cv2.drawMarker(self.frame_org,
                                                    com,
                                                    color      = (0,0,255), 
                                                    markerType = cv2.MARKER_CROSS, 
                                                    markerSize = 20, 
                                                    thickness  = 1)
                _com_x = [i[0] for i in com_org]
                _com_y = [i[1] for i in com_org]
                _len_x = len(_com_x)
                _len_y = len(_com_y)
                com_sum_x = int(sum(_com_x)/_len_x)
                com_sum_y = int(sum(_com_y)/_len_y)
                com_sum.append((com_sum_x, com_sum_y))  # com_sum保存
                time_cs.append(self.dt)
                cnt_cs.append(self.cnt)
                # velocity算出.
                if self.cnt != 1:
                    self._vel_x = (com_sum[-2][0]-com_sum_x)*self.coefficient
                    self._vel_y = (com_sum[-2][1]-com_sum_y)*self.coefficient
                    vel.append((self._vel_x, self._vel_y))  # vel保存.
                    time_vel.append(self.dt)
                    cnt_vel.append(self.cnt)
                    print(self.cnt, ": Velocity :", vel[-1])
                    # com_sum軌跡描写
                    cv2.arrowedLine(self.frame_org,
                                    (com_sum[-2][0],com_sum[-2][1]),
                                    (com_sum_x, com_sum_y),
                                    color     = (255,0,0),
                                    thickness = 2,
                                    line_type = 8,
                                    shift     = 0,
                                    tipLength = 0.1)
                    
                    #self.quaternion_estimation()
                
                if(_len_x*_len_y != 0):  # com_sum描写
                    self.frame_org = cv2.drawMarker(self.frame_org,
                                                    (com_sum_x, com_sum_y),
                                                    color      = (0,0,255),
                                                    markerType = cv2.MARKER_CROSS,
                                                    markerSize = 20,
                                                    thickness  = 2)
                self.cnt += 1
            
            cv2.drawContours(self.frame_org, contours, -1, (0,255,0), 1)
            cv2.imshow("org", self.frame_org)
            cv2.imshow("bin", frame_bin)
            
            self.end_flag, self.frame_org = self.org.read()
            
            # ESCキーで終了.
            key = cv2.waitKey(1)  # 1ms
            if key == 27:
                break
        
        cv2.destroyAllWindows()
        self.org.release()
        
        # グラフ描写.
        fig_cs = plt.figure(figsize=(9,9))
        axes = fig_cs.add_subplot(111, projection="3d")
        axes.view_init(elev=0, azim=0)
        axes.set_title("com_sum locus",            size=20)
        axes.set_xlabel("frame[frame]",            size=15)
        axes.set_ylabel("com_sum_x[pixels/frame]", size=15)
        axes.set_zlabel("com_sum_y[pixels/frame]", size=15)
        axes.set_ylim(0, self.width)
        axes.set_zlim(self.height, 0)
        axes.plot(cnt_cs, [i[0] for i in com_sum], [i[1] for i in com_sum])
        plt.show()
        
        plt.figure(figsize=(8,7))
        plt.subplot(211)
        plt.plot(cnt_vel, [i[0] for i in vel])
        plt.xlabel("frame[frame]")
        plt.ylabel("velocity_x[pixels/frame]")
        plt.grid()
        plt.subplot(212)
        plt.plot(cnt_vel, [i[1] for i in vel])
        plt.xlabel("frame[frame]")
        plt.ylabel("velocity_y[pixels/frame]")
        plt.grid()
        plt.show()
        
    def quaternion_estimation(self):
        wx = self._vel_y * np.pi / 180.0
        wy = self._vel_x * np.pi / 180.0
        wz = 0.0
        
        qdot0 = 0.5 * (           - self.q1*wx - self.q2*wy - self.q3*wz)
        qdot1 = 0.5 * (self.q0*wx              + self.q2*wz - self.q3*wy)
        qdot2 = 0.5 * (self.q0*wy - self.q1*wz              + self.q3*wx)
        qdot3 = 0.5 * (self.q0*wz + self.q1*wy - self.q2*wx)
        
        self.q0 += qdot0 * self.dt
        self.q1 += qdot1 * self.dt
        self.q2 += qdot2 * self.dt
        self.q3 += qdot3 * self.dt
        
        norm = np.sqrt(self.q0**2 + self.q1**2 + self.q2**2 + self.q3**2)
        self.q0 /= norm
        self.q1 /= norm
        self.q2 /= norm
        self.q3 /= norm
        
        print(self.cnt, ": Quaternion :", self.q0, self.q1, self.q2, self.q3)

if __name__ == "__main__":
    hse_c = Horizontal_speed__estimation_by_com(**set_parameter)
    hse_c.speed_estimation()