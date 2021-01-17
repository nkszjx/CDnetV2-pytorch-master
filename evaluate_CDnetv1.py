import argparse
import scipy
from scipy import ndimage
import cv2
import numpy as np
import sys
from collections import OrderedDict
import os

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
from torch.utils import data, model_zoo

from CDnetv1_model import CDnetV1

#from model.deeplabv3p import Res_Deeplab
from data.voc_dataset import VOCDataSet,VOCDataTestSet, VOCDataTestSet_jpg
from data import get_data_path, get_loader
import torchvision.transforms as transform

from PIL import Image
import scipy.misc


from metric import ConfusionMatrix

#IMG_MEAN = np.array((62.00698793,58.66876762,58.67891434 ), dtype=np.float32) # gf
#IMG_MEAN = np.array((69.44698793,51.68876762,49.67891434), dtype=np.float32) # landsat
IMG_MEAN = np.array((85.27698793,85.21876762,85.17891434), dtype=np.float32) # zy3

DATASET = 'pascal_voc' # pascal_context

# MODEL = 'deeplab_multi' # deeeplabv2, deeplabv3p
DATA_DIRECTORY = '/home/lab532/Remote_sensing/Unsupervised_SemSeg_master/ZY3_data/'
DATA_LIST_PATH = '/home/lab532/Remote_sensing/Unsupervised_SemSeg_master/ZY3_data/test_zy3_evaluation_50.txt'
IGNORE_LABEL = 255

NUM_CLASSES = 2 # 60 for pascal context
GPU_NUMBER=0

W=1200
H=1200
#os.environ['CUDA_VISIBLE_DEVICES']='1'
os.environ['CUDA_VISIBLE_DEVICES']='1'

LAYER_NAME= 'layer4'

UNSUPERVISION_ADAPT = 'voc_multi_split_landsat'

MODEL= 'VOC_80000'

RESTORE_FROM1 = './checkpoints/checkpoints_zy3_CDnetV1/'+ MODEL +'.pth'


PRETRAINED_MODEL = None
SAVE_DIRECTORY = './results_zy3_50/'


def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="VOC evaluation script")
    parser.add_argument("--gpu", type=int, default=GPU_NUMBER,
                        help="choose gpu device.")
    parser.add_argument("--layer_name", type=str, default=LAYER_NAME,
                        help="layer_name to be used")
						
    parser.add_argument("--model", type=str, default=MODEL,
                        help="available options : DeepLab/DRN")
    parser.add_argument("--dataset", type=str, default=DATASET,
                        help="dataset name pascal_voc or pascal_context")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--restore-from1", type=str, default=RESTORE_FROM1,
                        help="Where restore model parameters from.")
    # parser.add_argument("--restore-from2", type=str, default=RESTORE_FROM2,
                        # help="Where restore model parameters from.")						
						
    parser.add_argument("--save-dir", type=str, default=SAVE_DIRECTORY,
                        help="Directory to store results")

    parser.add_argument("--with-mlmt", action="store_true",
                        help="combine with Multi-Label Mean Teacher branch")
    parser.add_argument("--save-output-images", action="store_false",
                        help="save output images")
    parser.add_argument("--width", type=int, default=W,
                        help=" IMAGE width.")
    parser.add_argument("--high", type=int, default=H,
                        help="IMAGE high.")						
    return parser.parse_args()

class Clou_Colorize(object):
    def __init__(self, n=3):
        self.cmap = color_map(3)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.shape
        color_image = np.zeros((3, size[0], size[1]), dtype=np.uint8)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image)
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        # handle void
        mask = (255 == gray_image)
        color_image[0][mask] = color_image[1][mask] = color_image[2][mask] = 255

        return color_image
		
		
class VOCColorize(object):
    def __init__(self, n=22):
        self.cmap = color_map(22)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.shape
        color_image = np.zeros((3, size[0], size[1]), dtype=np.uint8)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image)
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        # handle void
        mask = (255 == gray_image)
        color_image[0][mask] = color_image[1][mask] = color_image[2][mask] = 255

        return color_image

def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):  # 0,1,2,...,N-1
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap

def get_label_vector(target, nclass):
    # target is a 3D Variable BxHxW, output is 2D BxnClass
    hist, _ = np.histogram(target, bins=nclass, range=(0, nclass-1))
    vect = hist>0
    vect_out = np.zeros((21,1))
    for i in range(len(vect)):
        if vect[i] == True:
            vect_out[i] = 1
        else:
            vect_out[i] = 0

    return vect_out

def get_iou(args, data_list, class_num, save_path=None):
    from multiprocessing import Pool
	
    ConfM = ConfusionMatrix(class_num)
    f = ConfM.generateM
    pool = Pool() 
    m_list = pool.map(f, data_list)
    pool.close() 
    pool.join() 
    
    for m in m_list:
        ConfM.addM(m)

    aveJ, j_list, M = ConfM.jaccard()

    if args.dataset == 'pascal_voc':
        classes = np.array(('background',  # always index 0
            'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor'))
    elif args.dataset == 'pascal_context':
        classes = np.array(('background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'table', 'dog', 'horse', 'motorbike', 'person',
                'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor', 'bag', 'bed', 'bench', 'book', 'building', 'cabinet' , 'ceiling', 'cloth', 'computer', 'cup',
                'door', 'fence', 'floor', 'flower', 'food', 'grass', 'ground', 'keyboard', 'light', 'mountain', 'mouse', 'curtain', 'platform', 'sign', 'plate',
                'road', 'rock', 'shelves', 'sidewalk', 'sky', 'snow', 'bedclothes', 'track', 'tree', 'truck', 'wall', 'water', 'window', 'wood'))
    # elif args.dataset == 'cityscapes':
        # classes = np.array(("road", "sidewalk",
            # "building", "wall", "fence", "pole",
            # "traffic_light", "traffic_sign", "vegetation",
            # "terrain", "sky", "person", "rider",
            # "car", "truck", "bus",
            # "train", "motorcycle", "bicycle")) 

    for i, iou in enumerate(j_list):
        print('class {:2d} {:12} IU {:.2f}'.format(i, classes[i], j_list[i]))
    
    print('meanIOU: ' + str(aveJ) + '\n')
    if save_path:
        with open(save_path, 'w') as f:
            for i, iou in enumerate(j_list):
                f.write('class {:2d} {:12} IU {:.2f}'.format(i, classes[i], j_list[i]) + '\n')
            f.write('meanIOU: ' + str(aveJ) + '\n')

def main():
    """Create the model and start the evaluation process."""

    args = get_arguments()
    #gpu0 = args.gpu

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        print('save_dir'+ args.save_dir)		

    model = CDnetV1(num_classes=args.num_classes)
	

	
    # create network
    # if args.layer_name == 'layer1':
        # model = Split_Res_Deeplab(num_classes=args.num_classes)
    # elif args.layer_name == 'layer2':
        # model = Split_Res_Deeplab(num_classes=args.num_classes)	
    # elif args.layer_name == 'layer3':
        # model = Split_Res_Deeplab_layer3(num_classes=args.num_classes)	
    # elif args.layer_name == 'layer4':	
        # model = Split_Res_Deeplab(num_classes=args.num_classes)	

    model.cuda()

    # if args.restore_from[:4] == 'http' :
        # saved_state_dict = model_zoo.load_url(args.restore_from)
    # else:
        # saved_state_dict = torch.load(args.restore_from)
		
    saved_state_dict1 = torch.load(args.restore_from1)	# load model parameters path
    model.load_state_dict(saved_state_dict1)  # load the model parameters to the network

    model.eval()  # test stage
    model.cuda() # using GPU 
	
    # saved_state_dict2 = torch.load(args.restore_from2)	# load model parameters path
    # model_2.load_state_dict(saved_state_dict2)  # load the model parameters to the network

    # model_2.eval()  # test stage
    # model_2.cuda() # using GPU 	
	
	
    #model.cuda() 	
    # load image for test
    if args.dataset == 'pascal_voc':
        # testloader = data.DataLoader(VOCDataSet(args.data_dir, args.data_list, crop_size=(321, 321), mean=IMG_MEAN, scale=False, mirror=False), 
                                    # batch_size=1, shuffle=False, pin_memory=True)
        testloader = data.DataLoader(VOCDataTestSet_jpg(args.data_dir, args.data_list, crop_size=(args.width, args.high), mean=IMG_MEAN), 
                                    batch_size=1, shuffle=False, pin_memory=True)									
        interp = nn.Upsample(size=(args.width, args.high), mode='bilinear', align_corners=True)

    elif args.dataset == 'pascal_context':
        input_transform = transform.Compose([transform.ToTensor(),
                transform.Normalize([.485, .456, .406], [.229, .224, .225])])
        # data_kwargs = {'transform': input_transform, 'base_size': 512, 'crop_size': 512}
        # data_loader = get_loader('pascal_context')
        # data_path = get_data_path('pascal_context')
        # test_dataset = data_loader(data_path, split='val', mode='val', **data_kwargs)
        # testloader = data.DataLoader(test_dataset, batch_size=1, drop_last=False, shuffle=False, num_workers=1, pin_memory=True)
        # interp = nn.Upsample(size=(512, 512), mode='bilinear', align_corners=True)

    # elif args.dataset == 'cityscapes':
        # data_loader = get_loader('cityscapes')
        # data_path = get_data_path('cityscapes')
        # test_dataset = data_loader( data_path, img_size=(512, 1024), is_transform=True, split='val')
        # testloader = data.DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)
        # interp = nn.Upsample(size=(512, 1024), mode='bilinear', align_corners=True)
    
    data_list = []
    colorize = Clou_Colorize()
   
    if args.with_mlmt:
        mlmt_preds = np.loadtxt('./mlmt_output/output_ema_p_1_0_voc_5.txt', dtype = float) # best mt 0.05

        mlmt_preds[mlmt_preds>=0.2] = 1
        mlmt_preds[mlmt_preds<0.2] = 0 
 
    for index, batch in enumerate(testloader):
        if index % 100 == 0:
            print('%d processd'%(index))
        #image, label, size, name, _ = batch
        image, size, name = batch		
        size = size[0]
        with torch.no_grad():
             output, _, _, _= model(Variable(image).cuda()) # model test 	 
			 
             #output  = model(Variable(image).cuda()) # model test 			 
			 
        output = interp(output).cpu().data[0].numpy() # resize the output result as the same size as the input image

        # if args.dataset == 'pascal_voc':
            # output = output[:,:size[0],:size[1]]
            # gt = np.asarray(label[0].numpy()[:size[0],:size[1]], dtype=np.int)
        # elif args.dataset == 'pascal_context':
            # gt = np.asarray(label[0].numpy(), dtype=np.int)
        # elif args.dataset == 'cityscapes':
            # gt = np.asarray(label[0].numpy(), dtype=np.int)

        if args.with_mlmt:
            for i in range(args.num_classes):
                output[i]= output[i]*mlmt_preds[index][i]
        
        output = output.transpose(1,2,0)
        output = np.asarray(np.argmax(output, axis=2), dtype=np.int)
       
        if args.save_output_images:
            if args.dataset == 'pascal_voc':
                filename = os.path.join(args.save_dir, '{}.png'.format(name[0]))
                color_file = Image.fromarray(colorize(output).transpose(1, 2, 0), 'RGB')
                color_file.save(filename)
                # if index % 50 == 0:				
                    # print('save images to:' + args.save_dir)			
            elif args.dataset == 'pascal_context':
                filename = os.path.join(args.save_dir, filename[0])
                # scipy.misc.imsave(filename, gt)
        
        #data_list.append([gt.flatten(), output.flatten()])
 
    #filename = os.path.join(args.save_dir, 'result.txt')
    #get_iou(args, data_list, args.num_classes, filename)


if __name__ == '__main__':
    main()
