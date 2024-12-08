import os
import sys
import cv2
import torch
import numpy as np
from pathlib import Path

names = [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush' ]


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov7') not in sys.path:
    sys.path.append(str(ROOT / 'yolov7'))  # add yolov5 ROOT to PATH
if str(ROOT / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'strong_sort'))  # add strong_sort ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from yolov7.utils.plots import plot_one_box


# 从txt文件中读取数据
def read_tracking_data(file_path):
    data = np.loadtxt(file_path)
    return data

COLORS_10 =[(144,238,144),(178, 34, 34),(221,160,221),(  0,255,  0),(  0,128,  0),(210,105, 30),(220, 20, 60),
            (192,192,192),(255,228,196),( 50,205, 50),(139,  0,139),(100,149,237),(138, 43,226),(238,130,238),
            (255,  0,255),(  0,100,  0),(127,255,  0),(255,  0,255),(  0,  0,205),(255,140,  0),(255,239,213),
            (199, 21,133),(124,252,  0),(147,112,219),(106, 90,205),(176,196,222),( 65,105,225),(173,255, 47),
            (255, 20,147),(219,112,147),(186, 85,211),(199, 21,133),(148,  0,211),(255, 99, 71),(144,238,144),
            (255,255,  0),(230,230,250),(  0,  0,255),(128,128,  0),(189,183,107),(255,255,224),(128,128,128),
            (105,105,105),( 64,224,208),(205,133, 63),(  0,128,128),( 72,209,204),(139, 69, 19),(255,245,238),
            (250,240,230),(152,251,152),(  0,255,255),(135,206,235),(  0,191,255),(176,224,230),(  0,250,154),
            (245,255,250),(240,230,140),(245,222,179),(  0,139,139),(143,188,143),(255,  0,  0),(240,128,128),
            (102,205,170),( 60,179,113),( 46,139, 87),(165, 42, 42),(178, 34, 34),(175,238,238),(255,248,220),
            (218,165, 32),(255,250,240),(253,245,230),(244,164, 96),(210,105, 30)]
COLORS_20 = [(230, 25, 75), (60, 180, 75), (255, 225, 25), 
             (0, 130, 200), (245, 130, 48), (145, 30, 180), 
             (70, 240, 240), (240, 50, 230), (210, 245, 60), 
             (250, 190, 212), (0, 128, 128), (220, 190, 255), 
             (170, 110, 40), (255, 250, 200), (128, 0, 0), (170, 255, 195), 
             (128, 128, 0), (255, 215, 180), (0, 0, 128), (128, 128, 128)]

import random
random.shuffle(COLORS_20)

# xyxy2tlwh函数  这个函数一般都会自带
def xyxy2tlwh(x):
    '''
    (top left x, top left y,width, height)
    '''
    y = torch.zeros_like(x) if isinstance(x,
                                          torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0]
    y[:, 1] = x[:, 1]
    y[:, 2] = x[:, 2] - x[:, 0]
    y[:, 3] = x[:, 3] - x[:, 1]
    return y

def plot_trajectory(video_path, tdata, out_path):
    # 读取视频帧
    cap = cv2.VideoCapture(video_path)
    # 获取视频帧率
    fps = cap.get(cv2.CAP_PROP_FPS)
    # 获取视频宽度，获取视频高度
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # 设置输出视频路径
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))
    dict_box = {}
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 获取当前帧
        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        # 获取当前帧的跟踪数据
        frame_data = tdata[tdata[:, 0] == frame_idx]
        frame_id_list = []
        for j in range(frame_data.shape[0]):
            # 获取跟踪目标的ID
            id = int(frame_data[j, 1])
            # 获取跟踪目标的位置信息
            bbox = frame_data[j, 2:6]
            x_center = bbox[0] + bbox[2] / 2
            y_center = bbox[1] + bbox[3] / 2
            dict_box.setdefault(id,[]).append([x_center, y_center])
            frame_id_list.append(id)
            # 获取跟踪目标的类别信息
            cls = int(frame_data[j, 6])
            # 获取跟踪目标的置信度信息
            conf = frame_data[j, 7]
            # 获取跟踪目标的颜色
            color = COLORS_20[id % len(COLORS_20)]
            # 绘制跟踪目标的框
            name_str = names[cls]
            label = f'{id} {name_str} {conf:.2f}'
            bbox_xyxy = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
            plot_one_box(bbox_xyxy, frame, label=label, color=color, line_thickness=2)
            #cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])), color, 2)
            # 绘制跟踪目标的ID
            #cv2.putText(frame, str(id), (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        # 将跟踪轨迹绘制到frame上
        for obj_id in frame_id_list:
            if len(dict_box[obj_id]) > 1:
                color = COLORS_20[obj_id % len(COLORS_20)]
                for i in range(1, len(dict_box[obj_id])):
                    cv2.line(frame, (int(dict_box[obj_id][i-1][0]), int(dict_box[obj_id][i-1][1])),
                             (int(dict_box[obj_id][i][0]), int(dict_box[obj_id][i][1])), color, 2)        
        out.write(frame)
        print(f"Processing frame {frame_idx}")
    cap.release()

if __name__ == "__main__":
    file_path  = "../cases/man_del/output_del.txt"
    video_path = "../cases/man_del/output_del.mp4"
    out_path   = "../cases/man_del/output_del_tracking.mp4"
    tracking_data = read_tracking_data(file_path)
    plot_trajectory(video_path, tracking_data, out_path)

