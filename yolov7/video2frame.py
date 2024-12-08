import cv2


def video2frame(videos_path, frames_save_path, time_interval):
    '''
    :param videos_path: 视频的存放路径
    :param frames_save_path: 视频切分成帧之后图片的保存路径
    :param time_interval: 保存间隔
    :return:
    '''
    vidcap = cv2.VideoCapture(videos_path)

    count = 1
    while True:
        success, image = vidcap.read()
        if image is None:
            break
        else:
            if count % time_interval == 0:
                cv2.imencode('.jpg', image)[1].tofile(frames_save_path + "/frame%d.jpg" % count)
        # if count == 20:
        #   break
        count += 1
    print(count)


if __name__ == '__main__':
    video2frame("../cross.mp4", "../data/frames_cross", 1)