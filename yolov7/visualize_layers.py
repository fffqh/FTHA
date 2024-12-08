from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import numpy as np
from utils.torch_utils import select_device
from models.experimental import attempt_load
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

class FM_visualize:
    def __init__(self, module_name, layer_index):
        self.hook = module_name[layer_index].register_forward_hook(self.hook_fn)
    
    def hook_fn(self, module, input, output):
        self.features = output.cpu().data.numpy()


class FM_visualize_pro:
    def __init__(self, module_name):
        self.hook = module_name.register_forward_hook(self.hook_fn)
    
    def hook_fn(self, module, input, output):
        # self.features = output.cpu().data.numpy()
        self.features = output


class VisLayers:
    def __init__(self, model, layers_info, model_name='model'):
        self.model = model
        self.layers_info = layers_info
        self.model_name = model_name
    def getfeature(self,img):
        trans = transforms.ToTensor()
        img = trans(img).unsqueeze(0)
        visual=[]
        for name, layer in self.layers_info.items():
            visual.append(layer)
        ELAN24=FM_visualize_pro(visual[0])
        ELAN37 = FM_visualize_pro(visual[1])
        ELAN50 = FM_visualize_pro(visual[2])
        self.model(img)
        ELAN24_features=ELAN24.features
        ELAN37_features = ELAN37.features
        ELAN50_features = ELAN50.features
        print(ELAN24_features)
        print(ELAN37_features)
        print(ELAN50_features)
        return ELAN24_features,ELAN37_features,ELAN50_features

    def run(self, img):
        trans = transforms.ToTensor()
        img = trans(img).unsqueeze(0)
        # self.model(img)
        for name, layer in self.layers_info.items():
            visual = FM_visualize_pro(layer)
            self.model(img)
            activations = visual.features
            print(activations)
            print(f"[{name}]\t{activations.shape}")
            if len(activations.shape) != 4:
                continue
            _, n_features, h_size, w_size = activations.shape
            display_grid = np.zeros((h_size, w_size*n_features))
            print(f"[display_grid] {display_grid.shape}")
            for i in range(n_features):
                x = activations[0,i,:,:]
                x -= x.mean()
                x /= x.std()
                x *= 64
                x +=128
                x = np.clip(x, 0, 255).astype('uint8')
                display_grid[:,i*w_size:(i+1)*w_size] = x
            scale = 100. / n_features
            plt.figure(figsize=(scale*n_features, scale*(h_size/w_size)))
            plt.title(name)
            plt.grid(False)
            plt.imshow(display_grid, aspect='auto', cmap='viridis')
            plt.savefig(f'{self.model_name}_{name}.jpg')

            # out_channal = activations.shape[1]
            # if out_channal > 256:
            #     print(f"!! out channal is {out_channal}, down to 128.")
            #     out_channal = 128

            # rows = int(out_channal/8)
            # columns = 8
            # fig, axes = plt.subplots(rows,columns,figsize=(30, 30))
            # for row in range(rows):
            #     for column in range(columns):
            #         axis = axes[row][column]
            #         axis.get_xaxis().set_ticks([])
            #         axis.get_yaxis().set_ticks([])
            #         axis.imshow(activations[0][row*8+column])
            # plt.savefig(f'{self.model_name}_{name}.jpg')
            

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Feature Map Visualizing')
    parser.add_argument('--img_path', default='inference/kitti/val/origin/000009.png', type=str, help='image path')
    args = parser.parse_args()
    weights='yolov7.pt'
    device = select_device('')
    model  = attempt_load(weights,map_location=device)
    print('Check the module name and number of the model!!')

    model_layers_info = {
        # 'conv_0':model.model[0].conv,
        # 'conv_1':model.model[1].conv,
        # 'conv_2':model.model[2].conv,
        # 'conv_3': model.model[3].conv,
        # 'conv_4': model.model[4].conv,
        # 'conv_5': model.model[5].conv,
        # 'conv_6': model.model[6].conv,
        # 'conv_7': model.model[7].conv,
        # 'conv_8': model.model[8].conv,
        # 'conv_9': model.model[9].conv,
        'ELAN_24':model.model[24].act,
        'ELAN_37':model.model[37].act,
        'ELAN_50':model.model[50].act,
        # 'conv2d_0': model.model[105].m[0],
        # 'conv2d_1': model.model[105].m[1],
        # 'conv2d_2': model.model[105].m[2],
    #    'ts1':model.features.transition1.conv,
    #    'ts2':model.features.transition2.conv,
    #    'ts3':model.features.transition3.conv,
    }

    layers=[]
    for name, layer in model_layers_info.items():
        layers.append(layer)

    img = Image.open(args.img_path)
    Vis = VisLayers(model,model_layers_info,'yolov7')
    Vis.getfeature(img)


