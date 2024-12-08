import os
import cv2
import argparse

from skimage import metrics
from skimage.metrics import peak_signal_noise_ratio


def get_ssim_psnr(img1,img2):
    #img1 = cv2.cvtColor(np.asarray(img1), cv2.COLOR_RGB2BGR)
    #img2 = cv2.cvtColor(np.asarray(img2), cv2.COLOR_RGB2BGR)
    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)

    PSNR = peak_signal_noise_ratio(img1, img2)
    SSIM = metrics.structural_similarity(img1, img2, multichannel=True)
    return PSNR, SSIM

def cal_all(rpath, ipath, opath):
    rfile_list = os.listdir(rpath)
    rimgs = [file for file in rfile_list if file.endswith((".png",".jpg"))]
    rimgs.sort()

    ifile_list = os.listdir(ipath)
    iimgs = [file for file in ifile_list if file.endswith((".png",".jpg"))]
    iimgs.sort()

    assert len(rimgs) == len(iimgs), "len(rimgs) != len(iimgs)"
    if not os.path.exists(opath):
        os.makedirs(opath)
    
    ofile = open(opath+'frames_ssim_psnr.txt', 'w+')

    # 开始循环
    for i, (rimg, iimg) in enumerate(zip(rimgs, iimgs)):
        rimg_path = rpath + rimg
        iimg_path = ipath + iimg
        img_name = rimg.split('.')[0]
        psnr,ssim = get_ssim_psnr(rimg_path, iimg_path)
        ofile.write(f"{i},{psnr},{ssim},{img_name}\n")
    ofile.close()
    print(f"Done. save to: {opath}frames_ssim_psnr.txt")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='sim')
    parser.add_argument('--r', type=str, required=True)
    parser.add_argument('--i', type=str, required=True)
    parser.add_argument('--o', type=str, required=True)
    args = parser.parse_args()
    print(args)

    assert os.path.exists(args.r), f"reference path error!"
    assert os.path.exists(args.i), f"input path error!"

    cal_all(rpath=args.r, ipath=args.i, opath=args.o)




