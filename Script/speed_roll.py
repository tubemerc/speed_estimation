# -*- coding: utf-8 -*-

'''
ToDo:
    - 路線変更したのでロール角算出に適した形にする
    - ﾏｰｶｰ座標算出方法を最適化する（今は輪郭->重心->平均）
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
                 "length"      : 500, 
                 "areasize"    : 0.0008,
                 "height"      : 240,
                 "width"       : 320,
                 "flag_rec"    : True,
                 "flag_draw"   : True
                 }


class Calculating_roll_from_com:
    '''
    - input_video : 入力映像. device number or file path.
    - th_binary_1 : 2値化閾値1.
    - th_binary_2 : 2値化閾値2.
    - th_binary_3 : 2値化処理方法.
    - coefficient : [pixels/frame(s)]を[m/s]に変換するための係数.
    - length      : カメラからマーカー中心までの距離, [mm].
    - areasize    : 輪郭の採用範囲.
    - height      : 入力画像のリサイズ高さ.
    - width       : 入力画像のリサイズ幅.
    - flag_rec    : 録画設定.
    - flag_rec    : 描写設定.
    '''
    
    def __init__(self, input_video,
                       th_binary_1, th_binary_2, th_binary_3,
                       coefficient,
                       length,
                       areasize,
                       height=None, width=None,
                       flag_rec=False,
                       flag_draw=False):
        self.th_binary_1 = th_binary_1
        self.th_binary_2 = th_binary_2
        self.th_binary_3 = th_binary_3
        self.coefficient = coefficient
        self.length      = length
        self.height      = height
        self.width       = width
        self.flag_rec    = flag_rec
        self.flag_draw   = flag_draw
        
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
              "coefficient :", self.coefficient, "\n",
              "Areasize    :", self.areasize,    "\n",
              "flag_rec    :", self.flag_rec,    "\n",
              "flag_draw   :", self.flag_draw,   "\n")
        
        cv2.namedWindow("org")
        cv2.namedWindow("bin")
    
        if self.flag_rec == True:
            fmt = cv2.VideoWriter_fourcc("m", "p", "4", "v")
            self.writer = cv2.VideoWriter("./video_roll.mp4",
                                          fmt,
                                          30.0,
                                          (self.width ,self.height))
    
    
    def roll_estimation(self):
        mark_place  = []
        angle       = []
        radius      = []
        time_i     = []
        cnt_i      = []
        
        cnt = 0
        x_o = int(self.width/2)
        y_o = int(self.height/2)
        GRAY   = cv2.COLOR_BGR2GRAY
        FONT   = cv2.FONT_HERSHEY_SIMPLEX
        MARKER = cv2.MARKER_CROSS
        LINE   = cv2.LINE_4
        time_0 = time.time()
        
        print("Speed estimation start.")
        while self.end_flag == True:
            t_i = time.time() - time_0  # 時間
            cnt += 1  # カウント
            
            '''
            前処理
            '''
            self.frame_org = cv2.resize(self.frame_org,
                                        dsize=(self.width, self.height))
            frame_bin = cv2.cvtColor(self.frame_org, GRAY)
            _, frame_bin = cv2.threshold(frame_bin, self.th_binary_1,
                                                    self.th_binary_2,
                                                    self.th_binary_3)
            
            '''
            マーカー座標算出
            '''
            contours, _ = cv2.findContours(frame_bin, 
                                           cv2.RETR_EXTERNAL, 
                                           cv2.CHAIN_APPROX_TC89_L1)
            if self.flag_draw == True:
                cv2.drawContours(self.frame_org, contours, -1, (0,255,0), 1)
            com_org = []
            for c in contours:
                moment = cv2.moments(c)
                if(moment["m00"] != 0):
                    if(moment["m00"] > self.areasize):
                        _com_x = int(moment["m10"]/moment["m00"])
                        _com_y = int(moment["m01"]/moment["m00"])
                        com_org.append((_com_x, _com_y))
            if com_org:
                if self.flag_draw == True:
                    for com in com_org:
                        self.frame_org = cv2.drawMarker(self.frame_org,
                                                        com,
                                                        color      = (0,0,255), 
                                                        markerType = MARKER, 
                                                        markerSize = 20, 
                                                        thickness  = 1)
                _com_x = [i[0] for i in com_org]
                _com_y = [i[1] for i in com_org]
                _len_x = len(_com_x)
                _len_y = len(_com_y)
                com_sum_x = int(sum(_com_x)/_len_x)
                com_sum_y = int(sum(_com_y)/_len_y)
                
                mark_place.append((com_sum_x, com_sum_y))  # マーカー座標
                time_i.append(t_i)
                cnt_i.append(cnt)
                
                if self.flag_draw == True:
                    if(_len_x*_len_y != 0):
                        self.frame_org = cv2.drawMarker(self.frame_org,
                                                        mark_place[-1],
                                                        color      = (0,0,255),
                                                        markerType = MARKER,
                                                        markerSize = 20,
                                                        thickness  = 2)
                
                '''
                マーカー座標変換(->曲座標)
                '''
                x = mark_place[-1][0] - x_o
                y = y_o - mark_place[-1][1]
                radius.append(pow(x**2 + y**2, 1/2))
                angle.append(np.arctan2(y, x) * 180/3.14159265)  # ロール角(仮)
                
                if self.flag_draw == True:
                    self.frame_org = cv2.drawMarker(self.frame_org,
                                                    (x_o, y_o),
                                                    color      = (0,0,255),
                                                    markerType = MARKER,
                                                    markerSize = 20,
                                                    thickness  = 3)
                    cv2.arrowedLine(self.frame_org,
                                    (x_o, y_o),
                                    mark_place[-1],
                                    color     = (255,0,0),
                                    thickness = 2,
                                    line_type = 8,
                                    shift     = 0,
                                    tipLength = 0.1)
                    cv2.putText(self.frame_org,
                               "angle:"+"{:.4g}".format(angle[-1]),
                               (0, 20), FONT, 0.8, (255, 0, 0), 2, LINE)
                    cv2.putText(self.frame_org,
                               "radius:"+"{:.4g}".format(radius[-1]),
                               (0, 50), FONT, 0.8, (255, 0, 0), 2, LINE)
                
                '''
                ロール角信用率をradiusから出す
                '''
            
            '''
            後処理
            '''
            cv2.imshow("org", self.frame_org)
            cv2.imshow("bin", frame_bin)
            
            if self.flag_rec == True:
                self.writer.write(self.frame_org)
            
            self.end_flag, self.frame_org = self.org.read()
            
            # ESCキーで終了.
            key = cv2.waitKey(1)
            if key == 27:
                break
        
        cv2.destroyAllWindows()
        self.org.release()
        self.writer.release()
        
        '''
        グラフ出力
        '''
        # マーカー軌跡
        fig_cs = plt.figure(figsize=(9,9))
        axes = fig_cs.add_subplot(111, projection="3d")
        axes.view_init(elev=0, azim=0)
        axes.set_title("marker locus", size=20)
        axes.set_xlabel("frame[frame]", size=15)
        axes.set_ylabel("marker_x[pixels]", size=15)
        axes.set_zlabel("marker_y[pixels]", size=15)
        axes.set_ylim(0, self.width)
        axes.set_zlim(self.height, 0)
        axes.plot(cnt_i,
                  [i[0] for i in mark_place],
                  [i[1] for i in mark_place])
        plt.show()
        
        # 角度等
        plt.figure(figsize=(8,7))
        plt.subplot(211)
        plt.plot(time_i, angle)
        plt.xlabel("time[s]")
        plt.ylabel("angle[degree]")
        plt.grid()
        plt.subplot(212)
        plt.plot(time_i, radius)
        plt.xlabel("time[s]")
        plt.ylabel("radius[pixels]")
        plt.grid()
        plt.show()


if __name__ == "__main__":
    crfc = Calculating_roll_from_com(**set_parameter)
    crfc.roll_estimation()