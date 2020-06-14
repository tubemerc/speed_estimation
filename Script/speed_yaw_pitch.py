# -*- coding: utf-8 -*-

'''
ToDo:
    - 全て
'''

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
                 "width"       : 320,
                 "flag_rec"    : True
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
                       height=None, width=None,
                       flag_rec=False):
        self.th_binary_1  = th_binary_1
        self.th_binary_2  = th_binary_2
        self.th_binary_3  = th_binary_3
        self.coefficient  = coefficient
        self.height       = height
        self.width        = width
        self.flag_rec    = flag_rec
        
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
    
        if self.flag_rec == True:
            fmt = cv2.VideoWriter_fourcc("m", "p", "4", "v")
            self.writer = cv2.VideoWriter("./video_yp.mp4",
                                          fmt,
                                          30.0,
                                          (self.width ,self.height))
        
        
        self.frame_org = cv2.resize(self.frame_org,
                                    dsize=(self.width, self.height))
        gray_prev = cv2.cvtColor(self.frame_org, cv2.COLOR_BGR2GRAY)
        feature_params = dict( maxCorners = 100,
                               qualityLevel = 0.3,
                               minDistance = 7,
                               blockSize = 7 )
        self.lk_params = dict( winSize  = (15,15),
                          maxLevel = 2,
                          criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.feature_prev = cv2.goodFeaturesToTrack(gray_prev,
                                                    mask = None,
                                                    **feature_params)
        self.mask = np.zeros_like(self.frame_org)
        self.color = np.random.randint(0, 255, (100, 3))
    
    def speed_estimation(self):
        com_sum  = []
        time_cs  = []
        cnt_cs   = []
        vel      = []
        time_vel = []
        cnt_vel  = []
        self.cnt = 1
        
        # 最初
        self.frame_org = cv2.resize(self.frame_org,
                                    dsize=(self.width, self.height))
        frame_gray_old = cv2.cvtColor(self.frame_org, cv2.COLOR_BGR2GRAY)
        
        self.end_flag, self.frame_org = self.org.read()
    
        
        print("Speed estimation start.")
        time_0 = time.time()
        
        while self.end_flag == True:
            self.frame_org = cv2.resize(self.frame_org,
                                        dsize=(self.width, self.height))
            frame_gray = cv2.cvtColor(self.frame_org, cv2.COLOR_BGR2GRAY)
            
            feature_next, status, err = cv2.calcOpticalFlowPyrLK(frame_gray_old,
                                                                 frame_gray,
                                                                 self.feature_prev,
                                                                 None,
                                                                 **self.lk_params)
            self.dt = time.time() - time_0
            
            if self.feature_prev[status == 1].all() and feature_next[status == 1].all:
                good_prev = self.feature_prev[status == 1]
                good_next = feature_next[status == 1]
            
            print(self.feature_prev[status == 1])
            print("-----------------------------")
            
            for i, (next_point, prev_point) in enumerate(zip(good_next, good_prev)):
                prev_x, prev_y = prev_point.ravel()
                next_x, next_y = next_point.ravel()
                self.mask = cv2.line(self.mask,
                                     (next_x, next_y),
                                     (prev_x, prev_y),
                                     self.color[i].tolist(),
                                     2)
                self.frame_org = cv2.circle(self.frame_org,
                                            (next_x, next_y),
                                            5,
                                            self.color[i].tolist(),
                                            -1)
            
            cv2.imshow("org", self.frame_org)
            cv2.imshow("bin", frame_gray)
            
            if self.flag_rec == True:
                self.writer.write(self.frame_org)
            
            frame_gray_old = frame_gray.copy()
            self.feature_prev = good_next.reshape(-1, 1, 2)
            
            self.end_flag, self.frame_org = self.org.read()
            
            # ESCキーで終了.
            key = cv2.waitKey(1)  # 1ms
            if key == 27:
                break
        
        cv2.destroyAllWindows()
        self.org.release()
        self.writer.release()
        
        # グラフ描写.
#        fig_cs = plt.figure(figsize=(9,9))
#        axes = fig_cs.add_subplot(111, projection="3d")
#        axes.view_init(elev=0, azim=0)
#        axes.set_title("com_sum locus",            size=20)
#        axes.set_xlabel("frame[frame]",            size=15)
#        axes.set_ylabel("com_sum_x[pixels/frame]", size=15)
#        axes.set_zlabel("com_sum_y[pixels/frame]", size=15)
#        axes.set_ylim(0, self.width)
#        axes.set_zlim(self.height, 0)
#        axes.plot(cnt_cs, [i[0] for i in com_sum], [i[1] for i in com_sum])
#        plt.show()
#        
#        plt.figure(figsize=(8,7))
#        plt.subplot(211)
#        plt.plot(cnt_vel, [i[0] for i in vel])
#        plt.xlabel("frame[frame]")
#        plt.ylabel("velocity_x[pixels/frame]")
#        plt.grid()
#        plt.subplot(212)
#        plt.plot(cnt_vel, [i[1] for i in vel])
#        plt.xlabel("frame[frame]")
#        plt.ylabel("velocity_y[pixels/frame]")
#        plt.grid()
#        plt.show()
        
if __name__ == "__main__":
    hse_c = Horizontal_speed__estimation_by_com(**set_parameter)
    hse_c.speed_estimation()