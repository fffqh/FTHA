import os
import cv2
import argparse
import numpy as np
from PIL import Image


def frame2video(im_dir, video_dir, fps):

    im_list = os.listdir(im_dir)
    im_list = [file for file in im_list if file.endswith((".png", ".jpg"))]
    # im_list.sort(key=lambda x: int(x.replace("frame", "").split('_')[0]))  # 最好再看看图片顺序对不
    # im_list.sort(key=lambda x: int(x.replace("frame", "").split('.')[0]))#原图用这个
    im_list.sort()
    img = Image.open(os.path.join(im_dir, im_list[0]))
    img_size = img.size  # 获得图片分辨率，im_dir文件夹下的图片分辨率需要一致

    # fourcc = cv2.cv.CV_FOURCC('M','J','P','G') #opencv版本是2
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # opencv版本是3
    videoWriter = cv2.VideoWriter(video_dir, fourcc, fps, img_size)
    # count = 1
    for i in im_list:
        im_name = os.path.join(im_dir + i)
        frame = cv2.imdecode(np.fromfile(im_name, dtype=np.uint8), -1)
        videoWriter.write(frame)
        # count+=1
        # if (count == 200):
        #     print(im_name)
        #     break
    videoWriter.release()
    print('finish.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='f2v')
    parser.add_argument('--i', type=str, required=True)
    parser.add_argument('--o', type=str, required=True)
    parser.add_argument('--f', type=int, default=2)
    args = parser.parse_args()
    print(args)

    fps = args.f
    im_dir = args.i
    video_dir = args.o
    frame2video(im_dir, video_dir, fps)
    print(f"out_path:{video_dir}")

    # # 视频验证
    # # docs = ['road2', 'road6_1', 'road6_2']
    # docs = ['road7_1']
    # for doc in docs:
    #     im_dir='inference/'+doc+'/moveres/'
    #     video_dir='inference/'+doc+'/moveres.mp4'
    #     fps = 2  # 帧率，每秒钟帧数越多，所显示的动作就会越流畅
    #     frame2video(im_dir, video_dir, fps)

    # 检测防御
    # denoise_style_list = ['Wavelet', 'bilateralFilter', 'GaussianBlur', 'medianBlur']
    # denoise_style_list = ['medianBlur','Quantized','Quantized+medianBlur','medianBlur+Quantized']
    # denoise_style_list = [ 'medianBlur+Quantized']
    # for denoise_style in denoise_style_list:
    #     im_dir='inference/road7_1/origin/defend/'+denoise_style+'/'
    #     video_dir='inference/road7_1/origin/defend/'+denoise_style+'/'+denoise_style+'.mp4'
    #     fps = 2  # 帧率，每秒钟帧数越多，所显示的动作就会越流畅
    #     frame2video(im_dir, video_dir, fps)