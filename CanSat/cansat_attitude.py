# -*- coding: utf-8 -*-
import time
import cv2
import smbus
import bme280
import can
import csv
import numpy as np
import struct
import binascii
import os
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

os.system('sudo /sbin/ip link set can0 up type can bitrate 100000 loopback off')
os.system('sudo ifconfig can0 txqueuelen 1000000')

set_parameter_up = {
    "input_video": 2,
    "th_binary_1": 50,
    "th_binary_2": 255,
    "th_binary_3": cv2.THRESH_BINARY_INV,
    "areasize_1": 0.005,
    "areasize_2": 0.015,
    "th_x": 50,
    "th_y": 10,
    "height": 480,
    "width": 640,
    "flag_rec": True,
    "flag_draw": False
}
set_parameter_down = {
    "input_video": 0,
    "cofficient": 17.5,
    "height": 480,
    "width": 640,
    "flag_rec": True,
    "flag_draw": False
}

GRAY = cv2.COLOR_BGR2GRAY
FONT = cv2.FONT_HERSHEY_SIMPLEX
MARKER = cv2.MARKER_CROSS
LINE = cv2.LINE_4
COLOR = np.random.randint(0, 255, (100, 3))
FMT = cv2.VideoWriter_fourcc("m", "p", "4", "v")

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0
plt.rcParams['font.size'] = 14
plt.rcParams['axes.linewidth'] = 1.0


class Calculating_up:

    def __init__(self, input_video,
                 th_binary_1, th_binary_2, th_binary_3,
                 areasize_1, areasize_2,
                 th_x, th_y,
                 height, width,
                 flag_rec, flag_draw):
        print('[INIT] :up.')

        self.th_binary_1 = th_binary_1
        self.th_binary_2 = th_binary_2
        self.th_binary_3 = th_binary_3
        self.height = height
        self.width = width
        self.flag_rec = flag_rec
        self.flag_draw = flag_draw

        self.org = cv2.VideoCapture(input_video)
        if not self.org.isOpened():
            print("[ERROR] Camera is not opened. :up")
        self.end_flag, self.frame_org = self.org.read()

        _height, _width, _ = self.frame_org.shape

        self.areasize_1 = self.height*self.width * areasize_1
        self.areasize_2 = self.height*self.width * areasize_2
        self.th_x = th_x
        self.th_y = th_y

        self.x_o = int(self.width / 2)
        self.y_o = int(self.height / 2)

        self.mark_place = []
        self.mark_place_append = self.mark_place.append
        self.time = []
        self.time_append = self.time.append
        self.cnt = []
        self.cnt_append = self.cnt.append
        self.mark_angle = []
        self.mark_angle_append = self.mark_angle.append
        self.mark_radius = []
        self.mark_radius_append = self.mark_radius.append

        self.angle_roll = 0.

        if self.flag_draw:
            cv2.namedWindow("org_up")
            cv2.namedWindow("bin_up")

        if self.flag_rec:
            self.writer = cv2.VideoWriter("./video/video_up_test.mp4",
                                          FMT,
                                          10.0,
                                          (self.width, self.height))

    def my_rec(self):
        self.end_flag, self.frame_org = self.org.read()
        self.writer.write(self.frame_org)

    def estimation(self, TIME, CNT):

        self.end_flag, self.frame_org = self.org.read()
        self.frame_org = cv2.resize(self.frame_org,
                                    dsize=(self.width, self.height))

        if self.frame_org is not None:

            if self.flag_rec:
                self.writer.write(self.frame_org)

            frame_bin = cv2.cvtColor(self.frame_org, GRAY)
            _, frame_bin = cv2.threshold(frame_bin,
                                         self.th_binary_1,
                                         self.th_binary_2,
                                         self.th_binary_3)

            contours, _ = cv2.findContours(frame_bin,
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_TC89_L1)
            if self.flag_draw:
                cv2.drawContours(self.frame_org, contours, -1, (0, 255, 0), 1)

            com_org = []
            com_org_append = com_org.append
            for c in contours:
                moment = cv2.moments(c)
                if(moment["m00"] != 0):
                    if(self.areasize_2 > moment["m00"] and
                       moment["m00"] > self.areasize_1):
                        _com_x = int(moment["m10"]/moment["m00"])
                        _com_y = int(moment["m01"]/moment["m00"])
                        if(self.width-self.th_x > _com_x and
                           _com_x > self.th_x):
                            if(self.height-self.th_y > _com_y and
                               _com_y > self.th_y):
                                com_org_append((_com_x, _com_y))

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

                self.mark_place_append((com_sum_x, com_sum_y))
                self.time_append(TIME)
                self.cnt_append(CNT)

                if self.flag_draw:
                    if(_len_x*_len_y != 0):
                        self.frame_org = cv2.drawMarker(self.frame_org,
                                                        self.mark_place[-1],
                                                        color=(0, 0, 255),
                                                        markerType=MARKER,
                                                        markerSize=20,
                                                        thickness=2)

                mark_x = self.mark_place[-1][0] - self.x_o
                mark_y = self.y_o - self.mark_place[-1][1]
                self.angle_roll = np.arctan2(
                    mark_y, mark_x) * 180/3.14159265 + 180

                self.mark_radius_append(pow(mark_x**2 + mark_y**2, 1/2))
                self.mark_angle_append(self.angle_roll)

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
                                "radius:" +
                                "{:.4g}".format(self.mark_radius[-1]),
                                (0, 50), FONT, 0.8, (255, 0, 0), 2, LINE)

            if self.flag_draw:
                cv2.imshow("org_up", self.frame_org)
                cv2.imshow("bin_up", frame_bin)

        else:
            print('[CHECK] no frame. :up')

    def end_flag_reset(self):
        self.end_flag, self.frame_org = self.org.read()

    def draw_graph(self):
        print('[START] save. :up')

        with open('up.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(self.mark_radius)
            writer.writerow(self.mark_angle)
            writer.writerow(self.time)
            writer.writerow(self.cnt)
        f.close()

        # 軌跡
        fig_cs = plt.figure(figsize=(9, 9))
        axes = fig_cs.add_subplot(111, projection="3d")
        axes.view_init(elev=0, azim=0)
        axes.set_xlabel("frame[frame]")
        axes.set_ylabel("marker_x[pixels]")
        axes.set_zlabel("marker_y[pixels]")
        axes.set_ylim(0, self.width)
        axes.set_zlim(self.height, 0)
        axes.plot(self.cnt,
                  [i[0] for i in self.mark_place],
                  [i[1] for i in self.mark_place])
        plt.grid()
        plt.savefig('./up_marker_locus.png')

        # 角度
        plt.figure(figsize=(8, 7))
        plt.subplot(211)
        plt.plot(self.time, self.mark_angle)
        plt.xlabel("time[s]")
        plt.ylabel("roll[deg]")
        plt.grid()
        plt.subplot(212)
        plt.plot(self.cnt, self.mark_angle)
        plt.xlabel("cnt[-]")
        plt.ylabel("roll[deg]")
        plt.grid()
        plt.savefig('./up_roll.png')

        # 半径
        plt.figure(figsize=(8, 7))
        plt.subplot(211)
        plt.plot(self.time, self.mark_radius)
        plt.xlabel("time[s]")
        plt.ylabel("radius[pixels]")
        plt.grid()
        plt.subplot(212)
        plt.plot(self.cnt, self.mark_radius)
        plt.xlabel("cnt[-]")
        plt.ylabel("radius[pixels]")
        plt.grid()
        plt.savefig('./up_radius.png')

        print('[COMPLETE] save. :up')

    def __del__(self):
        self.draw_graph()

        if self.flag_draw:
            cv2.destroyAllWindows()

        self.org.release()

        if self.flag_rec:
            self.writer.release()


class Calculating_down:

    def __init__(self, input_video,
                 cofficient,
                 height, width,
                 flag_rec, flag_draw):
        print('[INIT] :down')

        self.height = height
        self.width = width
        self.flag_rec = flag_rec
        self.flag_draw = flag_draw

        self.org = cv2.VideoCapture(input_video)
        if not self.org.isOpened():
            print("[ERROR] Camera is not opened. :down")
        self.end_flag, self.frame_org = self.org.read()

        _height, _width, _ = self.frame_org.shape

        self.x_o = int(self.width / 2)
        self.y_o = int(self.height / 2)

        self.cofficient = cofficient

        self.feature_params = dict(maxCorners=255,
                                   qualityLevel=0.3,
                                   minDistance=7,
                                   blockSize=7)
        self.lk_params = dict(winSize=(15, 15),
                              maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS |
                                        cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        self.frame_org = cv2.resize(self.frame_org,
                                    dsize=(self.width, self.height))
        frame_gray = cv2.cvtColor(self.frame_org, GRAY)
        self.feature_prev = cv2.goodFeaturesToTrack(frame_gray,
                                                    mask=None,
                                                    **self.feature_params)
        self.frame_gray_old = frame_gray.copy()

        self.d_angle = []
        self.d_angle_append = self.d_angle.append
        self.angle = []
        self.angle_append = self.angle.append
        self.time = []
        self.time_append = self.time.append
        self.cnt = []
        self.cnt_append = self.cnt.append

        self.angle_x = 0.
        self.angle_y = 0.

        if self.flag_draw:
            cv2.namedWindow("org_down")
            cv2.namedWindow("gray_down")

        if self.flag_rec:
            self.writer = cv2.VideoWriter("./video/video_down_test.mp4",
                                          FMT,
                                          10.0,
                                          (self.width, self.height))

    def my_rec(self):
        self.end_flag, self.frame_org = self.org.read()
        self.writer.write(self.frame_org)

    def estimation(self, TIME, CNT):

        self.end_flag, self.frame_org = self.org.read()
        self.frame_org = cv2.resize(self.frame_org,
                                    dsize=(self.width, self.height))

        if self.frame_org is not None:

            if self.flag_rec:
                self.writer.write(self.frame_org)

            frame_gray = cv2.cvtColor(self.frame_org, GRAY)

            feature_next, status, err = \
                cv2.calcOpticalFlowPyrLK(self.frame_gray_old,
                                         frame_gray,
                                         self.feature_prev,
                                         None,
                                         **self.lk_params)

            if feature_next[status == 1].all is not None:
                if self.feature_prev[status == 1].all() and \
                        feature_next[status == 1].all:
                    good_prev = self.feature_prev[status == 1]
                    good_next = feature_next[status == 1]

                if len(good_next) > 10:

                    prev_x_ave = 0
                    prev_y_ave = 0
                    next_x_ave = 0
                    next_y_ave = 0

                    for i, (next_point, prev_point) in \
                            enumerate(zip(good_next, good_prev)):
                        prev_x, prev_y = prev_point.ravel()
                        next_x, next_y = next_point.ravel()
                        if self.flag_draw:
                            print(i)
                            self.frame_org = cv2.circle(self.frame_org,
                                                        (next_x, next_y),
                                                        5,
                                                        COLOR[i].tolist(),
                                                        -1)
                        prev_x_ave += prev_x
                        prev_y_ave += prev_y
                        next_x_ave += next_x
                        next_y_ave += next_y

                    if len(good_prev)*len(good_next) != 0:
                        prev_x_ave /= len(good_prev)
                        prev_y_ave /= len(good_prev)
                        next_x_ave /= len(good_next)
                        next_y_ave /= len(good_next)

                        dx = next_x_ave - prev_x_ave
                        dy = next_y_ave - prev_y_ave

                        dx /= self.cofficient
                        dy /= self.cofficient

                        self.d_angle_append((dx, dy))
                        self.time_append(TIME)
                        self.cnt_append(CNT)

                        self.angle_x += dx
                        self.angle_y += dy

                        self.angle_append(
                            (self.angle_x+180., self.angle_y+180.))

                        self.frame_gray_old = frame_gray.copy()
                        self.feature_prev = good_next.reshape(-1, 1, 2)
                else:
                    self.OFreset()

            else:
                self.OFreset()

            if self.flag_draw:
                cv2.imshow("org_down", self.frame_org)
                cv2.imshow("gray_down", frame_gray)

        else:
            print('[CHECK] no frame. :up')

    def OFreset(self):
        self.end_flag, self.frame_org = self.org.read()

        self.frame_org = cv2.resize(self.frame_org,
                                    dsize=(self.width, self.height))
        frame_gray = cv2.cvtColor(self.frame_org, GRAY)
        self.feature_prev = cv2.goodFeaturesToTrack(frame_gray,
                                                    mask=None,
                                                    **self.feature_params)
        self.frame_gray_old = frame_gray.copy()

    def end_flag_reset(self):
        self.end_flag, self.frame_org = self.org.read()

    def draw_graph(self):
        print('[START] save. :down')

        with open('down.csv', 'a') as f2:
            writer = csv.writer(f2)
            writer.writerow([i[0] for i in self.d_angle])
            writer.writerow([i[1] for i in self.d_angle])
            writer.writerow([i[0] for i in self.angle])
            writer.writerow([i[1] for i in self.angle])
            writer.writerow(self.time)
            writer.writerow(self.cnt)
        f2.close()

        plt.figure(figsize=(8, 7))
        plt.subplot(211)
        plt.plot(self.time, [i[0] for i in self.d_angle], label='[deg/s]')
        plt.plot(self.time, [i[0] for i in self.angle], label='[deg]')
        plt.xlabel("time[s]")
        plt.ylabel("pitch[deg]")
        plt.grid()
        plt.legend()
        plt.subplot(212)
        plt.plot(self.time, [i[1] for i in self.d_angle], label='[deg/s]')
        plt.plot(self.time, [i[1] for i in self.angle], label='[deg]')
        plt.xlabel("time[s]")
        plt.ylabel("yaw[deg]")
        plt.grid()
        plt.savefig('./down_pitch_and_yaw_time.png')

        plt.figure(figsize=(8, 7))
        plt.subplot(211)
        plt.plot(self.cnt, [i[0] for i in self.d_angle], label='[deg/s]')
        plt.plot(self.cnt, [i[0] for i in self.angle], label='[deg]')
        plt.xlabel("cnt[-]")
        plt.ylabel("pitch[deg]")
        plt.grid()
        plt.legend()
        plt.subplot(212)
        plt.plot(self.cnt, [i[1] for i in self.d_angle], label='[deg/s]')
        plt.plot(self.cnt, [i[1] for i in self.angle], label='[deg]')
        plt.xlabel("cnt[-]")
        plt.ylabel("yaw[deg]")
        plt.grid()
        plt.savefig('./down_pitch_and_yaw_cnt.png')

        print('[COMPLETE] save. :down')

    def __del__(self):
        self.draw_graph()

        if self.flag_draw:
            cv2.destroyAllWindows()

        self.org.release()

        if self.flag_rec:
            self.writer.release()


def save_pres(_pres, _time, _cnt):
    print('[START] save. :pressure')
    with open('pres.csv', 'a') as f3:
        writer = csv.writer(f3)
        writer.writerow(_pres)
        writer.writerow(_time)
        writer.writerow(_cnt)
    f3.close()
    print('[COMPLETE] save. :pressure')


if __name__ == "__main__":
    print('[START] main loop.')

    canbus = can.interface.Bus(channel='can0', bustype='socketcan_native')
    print(canbus)

    bus = smbus.SMBus(1)
    bme280.load_calibration_params(bus, 0x76)

    up = Calculating_up(**set_parameter_up)
    down = Calculating_down(**set_parameter_down)

    pressure = []
    pressure_append = pressure.append
    time_pres = []
    time_pres_append = time_pres.append
    cnt_pres = []
    cnt_pres_append = cnt_pres.append

    cnt_i = 0
    cntline = []
    cntline_append = cntline.append

    t_0 = 0.
    t_i = 0.
    timeline = []
    timeline_append = timeline.append

    try:
        print('[START] waiting mode.')

        while(1):
            msg_r = canbus.recv(0.0)
            if msg_r is not None:
                if msg_r.arbitration_id == 2:
                    break

        t_0 = time.time()

        print('[START] attitude estimation.')
        while(1):

            t_i = time.time() - t_0
            cnt_i += 1
            up.estimation(t_i, cnt_i)
            down.estimation(t_i, cnt_i)

            s = bme280.sample(bus, 0x76)
            if s is not None:
                findString = 'pressure='
                t = str(s).find(findString)
                _pres = str(s)[t+len(findString): t+len(findString)+7]
                pressure_append(_pres)
                time_pres_append(t_i)
                cnt_pres_append(cnt_i)

            x = down.angle_x + 180.
            y = down.angle_y + 180.
            r = up.angle_roll

            print('x,y,r: ', x, y, r)

            r /= 10
            r_h = int(np.floor(r))
            r_l = int(round((r-r_h)*100))
            x /= 10
            x_h = int(np.floor(x))
            x_l = int(round((x-x_h)*100))
            y /= 10
            y_h = int(np.floor(y))
            y_l = int(round((y-y_h)*100))

            msg_s = can.Message(arbitration_id=0x05,
                                data=bytearray([x_h, x_l, y_h, y_l, r_h, r_l]),
                                extended_id=False)
            canbus.send(msg_s)

        print('[BREAK] mainloop')
        del up
        del down
        save_pres(pressure, time_pres, cnt_pres)

    finally:
        del up
        del down
        save_pres(pressure, time_pres, cnt_pres)
