import os
import cv2
import argparse
import numpy as np

import torch
from torch.autograd import Variable
import torchvision.transforms as transforms


from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.datasets import letterbox
from visualize_layers import FM_visualize_pro

def save_im_torch(im, path):
    im = im.squeeze(0).detach().cpu()
    im = transforms.ToPILImage()(im)
    im.save(path)

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
def reverse_scale_coords(img1_shape, coords, img0_shape):
    # Rescale coords (xyxy) from img0_shape to img1_shape

    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain = old / new
    pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding



    coords[0] *= gain  # x scaling
    coords[2] *= gain
    coords[1] *= gain
    coords[3] *= gain
    coords[0] += pad[0]  # x padding
    coords[2] += pad[0]
    coords[1] += pad[1]
    coords[3] += pad[1]

    return coords

def fgsm_attack(model, x, y, epsilon, layers, filepath, img0_shape, outdir, txts_path, rounds):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = Variable(x.to(device), requires_grad=True)
    y = Variable(y.to(device))
    # Set the model to evaluation mode
    model.eval()
    # Set requires_grad attribute of tensor x
    x.requires_grad = True
    ELAN24 = FM_visualize_pro(layers[0])
    ELAN37 = FM_visualize_pro(layers[1])
    ELAN50 = FM_visualize_pro(layers[2])
    ELAN24_target = FM_visualize_pro(layers[0])
    ELAN37_target = FM_visualize_pro(layers[1])
    ELAN50_target = FM_visualize_pro(layers[2])
    target = model(y)
    ELAN37_target_features = ELAN37_target.features
    ELAN50_target_features = ELAN50_target.features
    ELAN24_target_features = ELAN24_target.features
    loss_function = torch.nn.L1Loss(reduction='sum')
    # Forward pass through the model
    with open(txts_path, 'r') as file:
        coords = file.read()
        coords = [int(val) for val in coords.split(' ')]
    coords = [round(val) for val in reverse_scale_coords(x.shape[2:], coords, img0_shape)]
    momentum = torch.zeros_like(x).detach().to(x.device)

    # 循环
    for iteration in range(rounds):
        x.requires_grad = True
        img1_shape=x.shape[2:]
        output = model(x)[0]
        ELAN37_features = ELAN37.features
        ELAN50_features = ELAN50.features
        ELAN24_features = ELAN24.features

        loss = loss_function(ELAN37_target_features, ELAN37_features)/3+loss_function(ELAN50_target_features, ELAN50_features)/3+loss_function(ELAN24_target_features, ELAN24_features)/3
        print(loss)
        with torch.autograd.set_detect_anomaly(True):
            model.zero_grad()
            loss.backward(retain_graph=True)
        grad = x.grad.data
        grad = 0.9*momentum + grad / (grad.abs().sum() + 1e-8)
        momentum = grad
        sign_grad = torch.sign(grad)
        perturbation = epsilon * sign_grad
        perturbation_ori = perturbation.clone()
        w = x.shape[2:][1]
        h = x.shape[2:][0]
        perturbation[0][:, slice(0, coords[1]), slice(0, w)] = 0  # 只在人bbox中加干扰 行，列
        perturbation[0][:, slice(coords[3], h), slice(0, w)] = 0
        perturbation[0][:, slice(coords[1], coords[3]), slice(0, coords[0])] = 0
        perturbation[0][:, slice(coords[1], coords[3]), slice(coords[2], w)] = 0
        # Create the adversarial example by adding the perturbation to the input image
        x.data = x.data - perturbation
        x.data = torch.clamp(x.data, 0, 1)
        x=x.detach()
        momentum = momentum.detach()

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
        filename=filepath.split('_')[0]
        im0 = cv2.imread(indir + filepath)  # BGR
        img0_shape = im0.shape
        im=read_picture(im0)
        print(filepath)

        target_im_path = 'inference/'+doc+'/mov_target/movresult_' + filename + '.jpg'
        target=cv2.imread(target_im_path)
        target=cv2.resize(target,(img0_shape[1],img0_shape[0]))
        target_im=read_picture(target)
        txts_path = 'inference/'+doc+'/del_target/txts_outputs/' + filename + '.txt'
        fgsm_attack(model, im, target_im, epsilon, layers, filepath, img0_shape, outdir, txts_path, rounds)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='111')
    parser.add_argument('--cases', nargs='+', type=str)
    parser.add_argument('--i', type=str, default=None)
    parser.add_argument('--o', type=str, default=None)
    parser.add_argument('--eps', nargs='+', type=float, default=None)
    parser.add_argument('--rds', nargs='+', type=int,   default=None)
    args = parser.parse_args()
    print(args)

    iname = 'output_add' if args.i is None else str(args.i)
    oname = 'output_mov' if args.o is None else str(args.o)
    epsilons = [0.01, 0.002, 0.001, 0.0008, 0.0005, 0.0003] if args.eps is None else list(args.eps)
    rounds   = [  20,   200,   200,    300,    400,    600] if args.rds is None else list(args.rds)
    print(f"eps:{epsilons}")
    print(f"rds:{rounds}")
    assert len(epsilons) == len(rounds), "eps != rds"

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


    for case in args.cases:
        print(f"case: {case}")
        for eps, rds in zip(epsilons, rounds):
            indir = f"inference/{case}/{iname}/"
            outdir = f"inference/{case}/{oname}_{eps}_{rds}/"
            if not os.path.exists(indir):
                print(f"[wrong] case {case}, input path error: {indir}")
                print(f"go to the next case!")
                break
            try:
                do_fgsm_attack(model, indir, outdir, eps, rds, layers, case)
            except:
                print(f"[wrong] {case} - {eps} - {rds}")
                continue
            else:
                print(f"[done] {case} - {eps} - {rds}, save to path: {outdir}")
