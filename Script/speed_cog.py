# -*- coding: utf-8 -*-
import cv2

class Horizontal_speed__estimation_by_com:
    # input_video : 入力映像. device number or file path.
    # th_binary_1 : 2値化閾値1.
    # th_binary_2 : 2値化閾値1.
    # th_binary_3 : 2値化処理方法.
    # height      : 入力画像のリサイズ高さ.
    # width       : 入力画像のリサイズ幅.
    
    def __init__(self, input_video,
                       th_binary_1, th_binary_2, th_binary_3,
                       height=None, width=None):
        self.th_binary_1  = th_binary_1
        self.th_binary_2  = th_binary_2
        self.th_binary_3  = th_binary_3
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
              "th_binary_1 :", self.th_binary_1, "\n",
              "th_binary_2 :", self.th_binary_2, "\n")
        
        cv2.namedWindow("org")
        cv2.namedWindow("bin")
        
    def speed_estimation(self):
        com_sum = []
        
        while self.end_flag == True:
            frame_org = cv2.resize(self.frame_org, dsize=(self.width, self.height))
            frame_bin = cv2.cvtColor(frame_org, cv2.COLOR_BGR2GRAY)
            _, frame_bin = cv2.threshold(frame_bin, self.th_binary_1,
                                                    self.th_binary_2,
                                                    self.th_binary_3)
            
            # 重心座標算出
            contours, _ = cv2.findContours(frame_bin, 
                                           cv2.RETR_EXTERNAL, 
                                           cv2.CHAIN_APPROX_NONE)
            com_org = []
            for c in contours:
                _moment = cv2.moments(c)
                if(_moment['m00'] != 0):
                    _com_x = int(_moment['m10']/_moment['m00'])
                    _com_y = int(_moment['m01']/_moment['m00'])
                    com_org.append((_com_x, _com_y))
            
            for com in com_org:  # com_org描写
                self.frame_org = cv2.drawMarker(frame_org, com, (255,0,0), 
                                                markerType = cv2.MARKER_CROSS, 
                                                markerSize = 20, 
                                                thickness = 1)
            
            _com_x = [i[0] for i in com_org]
            _com_y = [i[1] for i in com_org]
            _len_x = len(_com_x)
            _len_y = len(_com_y)
            com_sum_x = int(sum(_com_x)/_len_x)
            com_sum_y = int(sum(_com_y)/_len_y)
            com_sum.append((com_sum_x, com_sum_y))  # com_sum保存
            
            if(_len_x*_len_y != 0):  # com_sum描写
                self.frame_org = cv2.drawMarker(frame_org,
                                                (com_sum_x, com_sum_y),
                                                (0,0,255),
                                                markerType = cv2.MARKER_CROSS,
                                                markerSize = 20,
                                                thickness = 2)
            
            cv2.drawContours(frame_org, contours, -1, (0,255,0), 1)
            cv2.imshow("org", frame_org)
            cv2.imshow("bin", frame_bin)
            
            self.end_flag, self.frame_org = self.org.read()
            
            # ESCキーで終了.
            key = cv2.waitKey(1)  # 1ms
            if key == 27:
                break
            
        cv2.destroyAllWindows()
        self.org.release()
        
        print(com_sum)

if __name__ == "__main__":
    hse_c = Horizontal_speed__estimation_by_com(input_video = 0,
                                                th_binary_1 = 50,
                                                th_binary_2 = 255,
                                                th_binary_3 = cv2.THRESH_BINARY_INV,
                                                height      = 240,
                                                width       = 320)
    hse_c.speed_estimation()