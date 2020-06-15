# -*- coding: utf-8 -*-

'''
ToDo:
- up
    - ﾏｰｶｰ座標算出方法を最適化する（今は輪郭->重心->平均）
- down
    - 全て
- other
    - それぞれモジュール化してもいいししなくてもいい
    - ERRORを置く
'''

import cv2
import numpy as np
import time
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# "up" or "down" or "all"
FLAG_TEST = "down"

set_parameter_up = {
    "input_video": 0,
    "th_binary_1": 50,
    "th_binary_2": 255,
    "th_binary_3": cv2.THRESH_BINARY_INV,
    "areasize": 0.0008,
    "height": 240,
    "width": 320,
    "flag_rec": True,
    "flag_draw": True
}
set_parameter_down = {
    "input_video": 0,
    "height": 240,
    "width": 320,
    "flag_rec": True,
    "flag_draw": True
}
# OpenCV Setting
GRAY = cv2.COLOR_BGR2GRAY
FONT = cv2.FONT_HERSHEY_SIMPLEX
MARKER = cv2.MARKER_CROSS
LINE = cv2.LINE_4
COLOR = np.random.randint(0, 255, (100, 3))
FMT = cv2.VideoWriter_fourcc("m", "p", "4", "v")


class Calculating_up:

    def __init__(self, input_video,
                 th_binary_1, th_binary_2, th_binary_3,
                 areasize,
                 height, width,
                 flag_rec, flag_draw):

        self.th_binary_1 = th_binary_1
        self.th_binary_2 = th_binary_2
        self.th_binary_3 = th_binary_3
        self.height = height
        self.width = width
        self.flag_rec = flag_rec
        self.flag_draw = flag_draw

        self.org = cv2.VideoCapture(input_video)
        self.end_flag, self.frame_org = self.org.read()

        _height, _width, _ = self.frame_org.shape
        print("<Original>\n "
              "Height :", _height, "\n",
              "Width  :", _width)

        self.areasize = self.height*self.width * areasize

        print("<Setting>\n "
              "Height      :", self.height,      "\n",
              "Width       :", self.width,       "\n",
              "th_binary_1 :", self.th_binary_1, "\n",
              "th_binary_2 :", self.th_binary_2, "\n",
              "th_binary_3 :", self.th_binary_3, "\n",
              "Areasize    :", self.areasize,    "\n",
              "flag_rec    :", self.flag_rec,    "\n",
              "flag_draw   :", self.flag_draw,   "\n")

        self.x_o = int(self.width / 2)
        self.y_o = int(self.height / 2)

        self.mark_place = []
        self.mark_angle = []
        self.mark_radius = []
        self.time = []
        self.cnt = []

        cv2.namedWindow("org_up")
        cv2.namedWindow("bin_up")

        if self.flag_rec:
            self.writer = cv2.VideoWriter("./video.mp4",
                                          FMT,
                                          30.0,
                                          (self.width, self.height))

    def estimation(self, TIME, CNT):
        '''
        前処理
        '''
        self.frame_org = cv2.resize(self.frame_org,
                                    dsize=(self.width, self.height))
        frame_bin = cv2.cvtColor(self.frame_org, GRAY)
        _, frame_bin = cv2.threshold(frame_bin,
                                     self.th_binary_1,
                                     self.th_binary_2,
                                     self.th_binary_3)

        '''
        マーカー座標算出
        '''
        contours, _ = cv2.findContours(frame_bin,
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_TC89_L1)
        if self.flag_draw:
            cv2.drawContours(self.frame_org, contours, -1, (0, 255, 0), 1)
        com_org = []
        for c in contours:
            moment = cv2.moments(c)
            if(moment["m00"] != 0):
                if(moment["m00"] > self.areasize):
                    _com_x = int(moment["m10"]/moment["m00"])
                    _com_y = int(moment["m01"]/moment["m00"])
                    com_org.append((_com_x, _com_y))
        if com_org:
            if self.flag_draw:
                for com in com_org:
                    self.frame_org = cv2.drawMarker(self.frame_org,
                                                    com,
                                                    color=(0, 0, 255),
                                                    markerType=MARKER,
                                                    markerSize=20,
                                                    thickness=1)
            _com_x = [i[0] for i in com_org]
            _com_y = [i[1] for i in com_org]
            _len_x = len(_com_x)
            _len_y = len(_com_y)
            com_sum_x = int(sum(_com_x)/_len_x)
            com_sum_y = int(sum(_com_y)/_len_y)

            self.mark_place.append((com_sum_x, com_sum_y))  # マーカー座標
            self.time.append(TIME)
            self.cnt.append(CNT)

            if self.flag_draw:
                if(_len_x*_len_y != 0):
                    self.frame_org = cv2.drawMarker(self.frame_org,
                                                    self.mark_place[-1],
                                                    color=(0, 0, 255),
                                                    markerType=MARKER,
                                                    markerSize=20,
                                                    thickness=2)

            '''
            マーカー座標変換(->曲座標)
            '''
            x = self.mark_place[-1][0] - self.x_o
            y = self.y_o - self.mark_place[-1][1]
            self.mark_radius.append(pow(x**2 + y**2, 1/2))
            self.mark_angle.append(
                np.arctan2(y, x) * 180/3.14159265)  # ロール角(仮)

            if self.flag_draw:
                self.frame_org = cv2.drawMarker(self.frame_org,
                                                (self.x_o, self.y_o),
                                                color=(0, 0, 255),
                                                markerType=MARKER,
                                                markerSize=20,
                                                thickness=3)
                cv2.arrowedLine(self.frame_org,
                                (self.x_o, self.y_o),
                                self.mark_place[-1],
                                color=(255, 0, 0),
                                thickness=2,
                                line_type=8,
                                shift=0,
                                tipLength=0.1)
                cv2.putText(self.frame_org,
                            "angle:"+"{:.4g}".format(self.mark_angle[-1]),
                            (0, 20), FONT, 0.8, (255, 0, 0), 2, LINE)
                cv2.putText(self.frame_org,
                            "radius:"+"{:.4g}".format(self.mark_radius[-1]),
                            (0, 50), FONT, 0.8, (255, 0, 0), 2, LINE)

            '''
            ロール角信用率をradiusから出す
            '''

        '''
        後処理
        '''
        cv2.imshow("org_up", self.frame_org)
        cv2.imshow("bin_up", frame_bin)

        if self.flag_rec:
            self.writer.write(self.frame_org)

        self.end_flag, self.frame_org = self.org.read()

    def draw_graph(self):
        # マーカー軌跡
        fig_cs = plt.figure(figsize=(9, 9))
        axes = fig_cs.add_subplot(111, projection="3d")
        axes.view_init(elev=0, azim=0)
        axes.set_title("marker locus", size=20)
        axes.set_xlabel("frame[frame]", size=15)
        axes.set_ylabel("marker_x[pixels]", size=15)
        axes.set_zlabel("marker_y[pixels]", size=15)
        axes.set_ylim(0, self.width)
        axes.set_zlim(self.height, 0)
        axes.plot(self.cnt,
                  [i[0] for i in self.mark_place],
                  [i[1] for i in self.mark_place])
        plt.show()

        # 角度等
        plt.figure(figsize=(8, 7))
        plt.subplot(211)
        plt.plot(self.time, self.mark_angle)
        plt.xlabel("time[s]")
        plt.ylabel("angle[degree]")
        plt.grid()
        plt.subplot(212)
        plt.plot(self.time, self.mark_radius)
        plt.xlabel("time[s]")
        plt.ylabel("radius[pixels]")
        plt.grid()
        plt.show()

    def __del__(self):
        self.draw_graph()

        cv2.destroyAllWindows()
        self.org.release()
        if self.flag_rec:
            self.writer.release()


class Calculating_down:

    def __init__(self, input_video,
                 height, width,
                 flag_rec, flag_draw):

        self.height = height
        self.width = width
        self.flag_rec = flag_rec
        self.flag_draw = flag_draw

        self.org = cv2.VideoCapture(input_video)
        self.end_flag, self.frame_org = self.org.read()

        _height, _width, _ = self.frame_org.shape
        print("<Original>\n "
              "Height :", _height, "\n",
              "Width  :", _width)

        print("<Setting>\n "
              "Height      :", self.height,      "\n",
              "Width       :", self.width,       "\n",
              "flag_rec    :", self.flag_rec,    "\n",
              "flag_draw   :", self.flag_draw,   "\n")

        self.x_o = int(self.width / 2)
        self.y_o = int(self.height / 2)

        self.mark_place = []
        self.mark_angle = []
        self.mark_radius = []
        self.time = []
        self.cnt = []

        self.frame_org = cv2.resize(self.frame_org,
                                    dsize=(self.width, self.height))
        gray_prev = cv2.cvtColor(self.frame_org, GRAY)
        feature_params = dict(maxCorners=100,
                              qualityLevel=0.3,
                              minDistance=7,
                              blockSize=7)
        self.lk_params = dict(winSize=(15, 15),
                              maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS |
                                        cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.feature_prev = cv2.goodFeaturesToTrack(gray_prev,
                                                    mask=None,
                                                    **feature_params)
        self.MASK = np.zeros_like(self.frame_org)

        self.frame_org = cv2.resize(self.frame_org,
                                    dsize=(self.width, self.height))
        self.frame_gray_old = cv2.cvtColor(self.frame_org, GRAY)

        self.end_flag, self.frame_org = self.org.read()

        cv2.namedWindow("org_down")
        cv2.namedWindow("gray_down")

        if self.flag_rec:
            self.writer = cv2.VideoWriter("./video.mp4",
                                          FMT,
                                          30.0,
                                          (self.width, self.height))

    def estimation(self, TIME, CNT):
        '''
        前処理
        '''
        self.frame_org = cv2.resize(self.frame_org,
                                    dsize=(self.width, self.height))
        frame_gray = cv2.cvtColor(self.frame_org, GRAY)

        '''
        OpticalFlow計算
        '''
        feature_next, status, err = \
            cv2.calcOpticalFlowPyrLK(self.frame_gray_old,
                                     frame_gray,
                                     self.feature_prev,
                                     None,
                                     **self.lk_params)

        if self.feature_prev[status == 1].all() and \
                feature_next[status == 1].all:
            good_prev = self.feature_prev[status == 1]
            good_next = feature_next[status == 1]

        for i, (next_point, prev_point) in \
                enumerate(zip(good_next, good_prev)):
            prev_x, prev_y = prev_point.ravel()
            next_x, next_y = next_point.ravel()
            if self.flag_draw:
                self.mask = cv2.line(self.MASK,
                                     (next_x, next_y),
                                     (prev_x, prev_y),
                                     COLOR[i].tolist(),
                                     2)
                self.frame_org = cv2.circle(self.frame_org,
                                            (next_x, next_y),
                                            5,
                                            COLOR[i].tolist(),
                                            -1)

        '''
        角度計算
        '''

        '''
        後処理
        '''
        cv2.imshow("org_down", self.frame_org)
        cv2.imshow("gray_down", frame_gray)

        if self.flag_rec:
            self.writer.write(self.frame_org)

        self.end_flag, self.frame_org = self.org.read()

    def draw_graph(self):
        pass

    def __del__(self):
        self.draw_graph()

        cv2.destroyAllWindows()
        self.org.release()
        if self.flag_rec:
            self.writer.release()


if __name__ == "__main__":
    print("Start.")
    if FLAG_TEST == "up":
        up = Calculating_up(**set_parameter_up)
        cnt_i = 0
        t_0 = time.time()
        while(1):
            t_i = time.time() - t_0
            cnt_i += 1
            if up.end_flag:
                up.estimation(t_i, cnt_i)
            else:
                pass
            key = cv2.waitKey(1)
            if key == 27:
                break
        del up

    elif FLAG_TEST == "down":
        down = Calculating_down(**set_parameter_down)
        cnt_i = 0
        t_0 = time.time()
        while(1):
            t_i = time.time() - t_0
            cnt_i += 1
            if down.end_flag:
                down.estimation(t_i, cnt_i)
            else:
                pass
            key = cv2.waitKey(1)
            if key == 27:
                break
        del down

    elif FLAG_TEST == "all":
        up = Calculating_up(**set_parameter_up)
        down = Calculating_down(**set_parameter_down)

        cnt_i = 0
        t_0 = time.time()

        while(1):
            t_i = time.time() - t_0
            cnt_i += 1

            if up.end_flag:
                up.estimation(t_i, cnt_i)
            else:
                pass

            if down.end_flag:
                down.estimation(t_i, cnt_i)
            else:
                pass

            # ESCキーで終了.
            key = cv2.waitKey(1)
            if key == 27:
                break

        del up
        del down

    else:
        print("[ERROR] FLAG is undefined.")
