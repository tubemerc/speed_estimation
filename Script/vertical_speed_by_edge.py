# -*- coding: utf-8 -*-
import cv2
import numpy as np
import statistics as stats
import time
from matplotlib import pyplot as plt

class Vertical_speed_estimation_by_edge:
    # input_video : 入力映像. device number or file path.
    # num_frame   : 何フレーム毎に速度推定を行うか.
    # num_block   : 入力画像を何分割するか.
    # th_canny_1  : Canny閾値1.
    # th_canny_2  : Canny閾値2.
    # th_edge_org : Edge閾値1. 0~1.
    # th_edge_ave : Edge閾値2. 0~1.
    # coefficient : [pixels/frame(s)]を[m/s]に変換するための係数.
    # height      : 入力画像のリサイズ高さ.
    # width       : 入力画像のリサイズ幅.
        
    def __init__(self, input_video,
                       num_frame, num_block,
                       th_canny_1, th_canny_2,
                       th_edge_org, th_edge_ave,
                       coefficient,
                       height=None, width=None):
        self.num_frame   = num_frame
        self.num_block   = num_block
        self.th_canny_1  = th_canny_1
        self.th_canny_2  = th_canny_2
        self.th_edge_org = th_edge_org
        self.th_edge_ave = th_edge_ave
        self.coefficient = coefficient
        self.height      = height
        self.width       = width
        
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
        print("<Setting>\n "
              "Height      :", self.height,      "\n",
              "Width       :", self.width,       "\n",
              "num_frame   :", self.num_frame,   "\n",
              "num_block   :", self.num_block,   "\n",
              "th_canny_1  :", self.th_canny_1,  "\n",
              "th_canny_2  :", self.th_canny_2,  "\n",
              "th_edge_org :", self.th_edge_org, "\n",
              "th_edge_ave :", self.th_edge_ave, "\n",
              "coefficient :", self.coefficient)
        
        cv2.namedWindow("org")
        cv2.namedWindow("edge")
    
    def speed_estimation(self):
        edge_org      = []
        time_edge_org = []
        cnt_edge_org  = []
        edge_ave      = []
        time_edge_ave = []
        cnt_edge_ave  = []
        vel           = []
        time_vel      = []
        cnt_vel       = []
        cnt           = 1
        
        print("Speed estimation start.")
        time_0 = time.time()
        
        while self.end_flag == True:
            frame_edge = cv2.cvtColor(self.frame_org, cv2.COLOR_BGR2GRAY)
            frame_edge = cv2.resize(frame_edge, dsize=(self.width,self.height))
            frame_edge = cv2.Canny(frame_edge,
                                   threshold1 = self.th_canny_1,
                                   threshold2 = self.th_canny_2)
            dt = time.time() - time_0
            
            # 画面分割して中央値を取得.
            _edge = []
            i = 0
            while i < self.num_block:
                j = 0
                while j < self.num_block:
                    a = int(    i*(self.height/self.num_block)  )
                    b = int((i+1)*(self.height/self.num_block)-1)
                    c = int(    j*( self.width/self.num_block)  )
                    d = int((j+1)*( self.width/self.num_block)-1)
                    _edge.append(np.sum(frame_edge[a:b,c:d]==255))
                    j += 1
                i += 1
            _edge_med = stats.median(_edge)
            # edge算出.
            if _edge_med != 0:
                _edge = [e for e in _edge
                         if e/_edge_med > self.th_edge_org or
                            e/_edge_med < 1+self.th_edge_org]
                edge_org.append(sum(_edge)/len(_edge))  # edge_org保存.
                time_edge_org.append(dt)
                cnt_edge_org.append(cnt)
            else:
                edge_org.append(0)
                time_edge_org.append(dt)
                cnt_edge_org.append(cnt)
            
            if cnt%self.num_frame == 0:
                edge_med = stats.median(edge_org[-self.num_frame:])
                # edge平均値算出.
                if edge_med != 0:
                    _edge_ave = [e for e in edge_org[-self.num_frame:]
                                 if e/edge_med > self.th_edge_ave or
                                    e/edge_med < 1+self.th_edge_ave]
                    edge_ave.append(stats.median(_edge_ave[-self.num_frame:]))  # edge_ave保存.
                    time_edge_ave.append(dt)
                    cnt_edge_ave.append(cnt)
                else:
                    edge_ave.append(0)
                    time_edge_ave.append(dt)
                    cnt_edge_ave.append(cnt)
                # velocity算出.
                if cnt/self.num_frame != 1:
                    _vel = (edge_ave[-2]-edge_ave[-1])*self.coefficient
                    vel.append(_vel)  # vel保存.
                    time_vel.append(dt)
                    cnt_vel.append(cnt)
                    print(cnt, ": Velocity :", _vel)
            
            cv2.imshow("org", self.frame_org)
            cv2.imshow("edge", frame_edge)
            
            cnt += 1
            self.end_flag, self.frame_org = self.org.read()
            
            # ESCキーで終了.
            key = cv2.waitKey(1)  # 1ms.
            if key == 27:
                break
        
        cv2.destroyAllWindows()
        self.org.release()
        
        # グラフ描写.
        plt.figure(1)
        plt.subplot(211)
        plt.plot(cnt_edge_org, edge_org, label="org")
        plt.plot(cnt_edge_ave, edge_ave, label="ave")
        plt.xlabel("frame[frame]")
        plt.ylabel("edge[pixels]")
        plt.grid()
        plt.legend()
        plt.subplot(212)
        plt.plot(cnt_vel, vel)
        plt.xlabel("frame[frame]")
        plt.ylabel("velocity[pixels/frame]")
        plt.grid()
        plt.show()
        
        plt.figure(1)
        plt.subplot(211)
        plt.plot(time_edge_org, edge_org, label="org")
        plt.plot(time_edge_ave, edge_ave, label="ave")
        plt.xlabel("time[s]")
        plt.ylabel("edge[pixels]")
        plt.grid()
        plt.legend()
        plt.subplot(212)
        plt.plot(time_vel, vel)
        plt.xlabel("time[s]")
        plt.ylabel("velocity[pixels/s]")
        plt.grid()
        plt.show()
        
if __name__ == "__main__":
    vse_e = Vertical_speed_estimation_by_edge(input_video = "../movie/video_1.mp4",
                                              num_frame   = 10,
                                              num_block   = 3,
                                              th_canny_1  = 100,
                                              th_canny_2  = 50,
                                              th_edge_org = 0.8,
                                              th_edge_ave = 0.8,
                                              coefficient = 1,
                                              height      = 540,
                                              width       = 960)
    vse_e.speed_estimation()