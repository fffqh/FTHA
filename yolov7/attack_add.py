import argparse
import os
import cv2
import numpy as np

import torch
from torch.autograd import Variable
import torchvision.transforms as transforms

from skimage import metrics
from skimage.metrics import peak_signal_noise_ratio


from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.datasets import letterbox
from visualize_layers import FM_visualize_pro

save_eps_list = None
save_rds_list = None
save_out_path = None

def get_ssim_psnr(img1,img2):
    img1 = cv2.cvtColor(np.asarray(img1), cv2.COLOR_RGB2BGR)
    img2 = cv2.cvtColor(np.asarray(img2), cv2.COLOR_RGB2BGR)

    PSNR = peak_signal_noise_ratio(img1, img2)
    SSIM = metrics.structural_similarity(img1, img2, multichannel=True)
    return PSNR, SSIM

def scale_image(img1_shape,img1, img0_shape):
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
    pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding

    img1=img1.crop((pad[0],pad[1],img1_shape[1]-pad[0],img1_shape[0]-pad[1]))
    img_resize=img1.resize((img0_shape[1],img0_shape[0]))
    return img_resize

def read_picture(im0):
    im = letterbox(im0, [640, 640], stride=32, auto=True)[0]  # padded resize
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)  # contiguous
    im = torch.from_numpy(im)
    im = im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim
    return im

def fgsm_attack(model, x, y, epsilon, layers, filepath, img0_shape, outdir, rounds):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ori_x = transforms.ToPILImage()(x.squeeze(0).detach().cpu())

    x = Variable(x.to(device), requires_grad=True)
    y = Variable(y.to(device))
    model.eval()
    x.requires_grad = True

    ELANS = []
    ELANS_T = []
    for layer in layers:
        ELANS.append(FM_visualize_pro(layer))
        ELANS_T.append(FM_visualize_pro(layer))
    target = model(y)
    fv_t = []
    for E in ELANS_T:
        fv_t.append(E.features)

    loss_function = torch.nn.MSELoss()
    momentum = torch.zeros_like(x).detach().to(x.device)
    # 循环
    for iteration in range(rounds):
        x.requires_grad = True
        img1_shape=x.shape[2:]
        output = model(x)[0]
        fv = []
        for E in ELANS:
            fv.append(E.features)

        loss = 0
        loss_len = len(fv)
        for i in range(loss_len):
            loss += loss_function(fv_t[i], fv[i])
        loss /= loss_len
        print(loss)
        # Calculate the gradients of the loss w.r.t. the input image
        with torch.autograd.set_detect_anomaly(True):
            model.zero_grad()
            loss.backward(retain_graph=True)
        grad = x.grad.data
        grad = 0.9*momentum + grad / (grad.abs().sum() + 1e-8)
        momentum = grad
        x.data = x.data - epsilon*torch.sign(grad)
        x.data = torch.clamp(x.data, 0, 1)
        x=x.detach()
        momentum = momentum.detach()
        # 判断是否需要计算ssim并保存
        if (epsilon in save_eps_list) and ((iteration+1) in save_rds_list):
            adv_x = torch.clamp(x, 0, 1)
            adv_x = transforms.ToPILImage()(adv_x.squeeze(0).detach().cpu())
            psnr,ssim = get_ssim_psnr(ori_x, adv_x)
            with open(save_out_path, 'a+') as f:
                f.write(f"{epsilon} {iteration+1} {psnr} {ssim} {filepath}\n")
            print(f"[info] save ssim psnr! eps:{epsilon} {iteration+1} {filepath}")

    # Clip the pixel values of the adversarial example to ensure that they remain within the valid range
    adversarial_example = torch.clamp(x, 0, 1)

    # Save
    perturbed_image = adversarial_example.squeeze(0).detach().cpu()
    perturbed_image = transforms.ToPILImage()(perturbed_image)
    perturbed_image = scale_image(img1_shape, perturbed_image, img0_shape)
    perturbed_image.save(outdir + filepath[:-4] + '_fgm' + str(epsilon)+'_'+str(iteration+1) + filepath[-4:])

def do_fgsm_attack(model, indir, outdir, epsilon, rounds, layers, doc):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    file_list = os.listdir(indir)
    images = [file for file in file_list if file.endswith((".png",".jpg"))]
    images.sort()
    for filepath in images:
        im0 = cv2.imread(indir + filepath)  # BGR
        img0_shape = im0.shape
        im=read_picture(im0)
        print(filepath)
        target=cv2.imread('inference/'+doc+'/add_target/'+'addresult_'+filepath)
        target_im=read_picture(target)
        fgsm_attack(model, im, target_im, epsilon, layers, filepath, img0_shape, outdir, rounds)


if __name__ == '__main__':
    my_parser = argparse.ArgumentParser(description='111')
    my_parser.add_argument('--cases', nargs='+', type=str)
    my_parser.add_argument('--i', type=str, default=None)
    my_parser.add_argument('--o', type=str, default=None)
    my_parser.add_argument('--eps', type=float,nargs='+', default=None)
    my_parser.add_argument('--rds', type=int,  nargs='+',  default=None)
    my_parser.add_argument('--save_eps', type=float, nargs='+')
    my_parser.add_argument('--save_rds', type=int, nargs='+')
    my_parser.add_argument('--save_out', type=str, default='save.txt')
    my_args = my_parser.parse_args()
    print(my_args)

    iname = 'origin' if my_args.i is None else str(my_args.i)
    oname = 'output_add' if my_args.o is None else str(my_args.o)
    epsilons = [0.01, 0.002, 0.001, 0.0008, 0.0005, 0.0003] if my_args.eps is None else list(my_args.eps)
    rounds   = [  20,   200,   200,    300,    400,    600] if my_args.rds is None else list(my_args.rds)
    print(f"eps:{epsilons}")
    print(f"rds:{rounds}")

    assert len(epsilons) == len(rounds), "eps != rds"

    save_rds_list = [50,100,150,200] if (my_args.save_rds is None) or (len(list(my_args.save_rds)) == 0) else list(my_args.save_rds)
    save_eps_list = [0.002,0.001,0.0005] if (my_args.save_eps is None) or (len(list(my_args.save_eps)) == 0) else list(my_args.save_eps)
    print(f"save_eps:{save_eps_list}")
    print(f"save_rds:{save_rds_list}")

    weights='yolov7.pt'
    device = select_device('')
    model  = attempt_load(weights,map_location=device)
    model_layers_info = {
            'ELAN_24':model.model[24].act,
            'ELAN_37':model.model[37].act,
            'ELAN_50':model.model[50].act,
        }
    layers=[]
    for name, layer in model_layers_info.items():
        layers.append(layer)

    for case in my_args.cases:
        print(f"case: {case}")
        for eps, rds in zip(epsilons, rounds):
            indir = f"inference/{case}/{iname}/"
            outdir = f"inference/{case}/{oname}_{eps}_{rds}/"
            save_out_path = f"inference/{case}/{my_args.save_out}"
            if not os.path.exists(indir):
                print(f"[wrong] case {case}, input path error: {indir}")
                print(f"go to the next case!")
                break
            # try:
            do_fgsm_attack(model, indir, outdir, eps, rds, layers, case)
            # except:
            #     print(f"[wrong] {case} - {eps} - {rds}")
            #     continue
            # else:
            #     print(f"[done] {case} - {eps} - {rds}, save to path: {outdir}")

