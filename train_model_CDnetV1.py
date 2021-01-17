
import argparse
import os
import sys
import random
import timeit

import cv2
import numpy as np
import pickle
import scipy.misc

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils import data, model_zoo
from torch.autograd import Variable
import torchvision.transforms as transform

from CDnetv1_model import CDnetV1
#from deeplabv2 import Multi_Split_Res_Deeplab, Split_Res_Deeplab, Split_Res_Deeplab_layer3
#from model.deeplabv3p import Res_Deeplab 

#from discriminator import s4GAN_discriminator, s4GAN_feature_discriminator, entropy_discriminator, Split_feature_discriminator_0

from loss import CrossEntropy2d
from data.voc_dataset import VOCDataSet, VOCGTDataSet, LandDataSet
from data import get_loader, get_data_path
from data.augmentations import *

start = timeit.default_timer()



# # load GF1 data
DATA_LIST_PATH2 = '/home/lab532/Remote_sensing/Semisup_SemSeg_master/GF1_data/percentage/50/train_gf1.txt'
DATA_LIST_PATH3 = '/home/lab532/Remote_sensing/Semisup_SemSeg_master/GF1_data/percentage/5/train_gf1_remain.txt'



# load landast8 data

LAND_DATA_DIR = '/home/lab532/Remote_sensing/Semisup_SemSeg_maste_Landsat8/GF1_data/'
DATA_LIST_PATH = '/home/lab532/Remote_sensing/Semisup_SemSeg_maste_Landsat8/GF1_data/train_landsat8.txt'



# load zy3 data
GF_DATA_DIR = '/home/lab532/Remote_sensing/Unsupervised_SemSeg_master/ZY3_data/'
DATA_LIST_PATH3 = '/home/lab532/Remote_sensing/Unsupervised_SemSeg_master/ZY3_data/train_zy3_list.txt'


#DATA_LIST_PATH3 = '/home/lab532/Remote_sensing/Unsupervised_SemSeg_master/ZY3_data/#percentage/0.5/train_zy3_remain.txt'



CHECKPOINT_DIR = './checkpoints/checkpoints_zy3_CDnetV1/'

#checkpoints_advent_feature/voc_advent_feature_landsat_layer3



THRESHOLD_VALUE= 0.55
GPU_NUMBER=0
os.environ['CUDA_VISIBLE_DEVICES']='1'

IMG_MEAN = np.array((85.27698793,85.21876762,85.17891434), dtype=np.float32) # zy3
#IMG_MEAN2 = np.array((69.44698793,51.68876762,49.67891434), dtype=np.float32) # landsat

NUM_CLASSES = 2 # 21 for PASCAL-VOC / 60 for PASCAL-Context / 19 Cityscapes 
DATASET = 'pascal_voc' #pascal_voc or pascal_context 

SPLIT_ID = './splits/voc/split_0.pkl'

MODEL = 'DeepLab'
BATCH_SIZE = 10
NUM_STEPS = 80000
SAVE_PRED_EVERY = 5000

INPUT_SIZE = '321,321'
IGNORE_LABEL = 255 # 255 for PASCAL-VOC / -1 for PASCAL-Context / 250 for Cityscapes

RESTORE_FROM = '/home/lab532/Remote_sensing/Unsupervised_SemSeg_master/pretrained_models/resnet50-19c8e357.pth'

LEARNING_RATE = 1e-4
LEARNING_RATE_D = 1e-4

POWER = 0.9
WEIGHT_DECAY = 0.0005
MOMENTUM = 0.9
NUM_WORKERS = 4
RANDOM_SEED = 1234

LAMBDA_ST = 1.0
LAMBDA_FM = 0.5
LAMBDA_CE = 0.5
LAMBDA_ADAPT = 0.001
LAMBDA_ENTROPY = 0.01


#LAMBDA_ADAPT = 1.0

THRESHOLD_ST = 0.6 # 0.6 for PASCAL-VOC/Context / 0.7 for Cityscapes

LABELED_RATIO = None  #0.02 # 1/8 labeled data by default




def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--gpu", type=int, default=GPU_NUMBER,
                        help="choose gpu device.")	
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the LANDSAT-8 file listing the images in the dataset.")					
    parser.add_argument("--data-list2", type=str, default=DATA_LIST_PATH2,
                        help="Path to the gf1 file listing the images in the dataset.")		
    parser.add_argument("--data-list3", type=str, default=DATA_LIST_PATH3,
                        help="Path to the gf1 file listing the images in the dataset.")
						
    parser.add_argument("--checkpoint-dir", type=str, default=CHECKPOINT_DIR,
                        help="Where to save checkpoints of the model.")
				
    parser.add_argument("--model", type=str, default=MODEL,
                        help="available options : DeepLab/DRN")
    parser.add_argument("--dataset", type=str, default=DATASET,
                        help="dataset to be used")
    # parser.add_argument("--layer_name", type=str, default=LAYER_NAME,
                        # help="layer_name to be used")
						
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--gf-data-dir", type=str, default=GF_DATA_DIR,
                        help="Path to the directory containing the GF1 dataset.")
    parser.add_argument("--land-data-dir", type=str, default=LAND_DATA_DIR,
                        help="Path to the directory containing the LANDSAT-8 dataset.")
						
    parser.add_argument("--labeled-ratio", type=float, default=LABELED_RATIO,
                        help="ratio of the labeled data to full dataset")
    parser.add_argument("--split-id", type=str, default=SPLIT_ID,
                        help="split order id")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--learning-rate-D", type=float, default=LEARNING_RATE_D,
                        help="Base learning rate for discriminator.")
    parser.add_argument("--lambda-fm", type=float, default=LAMBDA_FM,
                        help="lambda_fm for feature-matching loss.")
    parser.add_argument("--lambda-st", type=float, default=LAMBDA_ST,
                        help="lambda_st for self-training.")
    parser.add_argument("--lambda-adapt", type=float, default=LAMBDA_ADAPT,
                        help="lambda_adapt for self-training.")
    parser.add_argument("--lambda-ce", type=float, default=LAMBDA_CE,
                        help="lambda_adapt for self-training.")						
    parser.add_argument("--lambda-entropy", type=float, default=LAMBDA_ENTROPY,
                        help="lambda_entropy for entropy loss.")		  

                        
    parser.add_argument("--threshold-st", type=float, default=THRESHOLD_ST,
                        help="threshold_st for the self-training threshold.")
    parser.add_argument("--threshold-value", type=float, default=THRESHOLD_VALUE,
                        help="threshold_value for the self-training threshold.")						
						
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--ignore-label", type=float, default=IGNORE_LABEL,
                        help="label value to ignored for loss calculation")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of iterations.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--restore-from-D", type=str, default=None,
                        help="Where restore model parameters from.")
    parser.add_argument("--restore-from-D1", type=str, default=None,
                        help="Where restore model parameters from.")	
    parser.add_argument("--restore-from-D2", type=str, default=None,
                        help="Where restore model parameters from.")	
						
    parser.add_argument("--restore-from-D31", type=str, default=None,
                        help="Where restore model parameters from.")
    parser.add_argument("--restore-from-D32", type=str, default=None,
                        help="Where restore model parameters from.")	
    parser.add_argument("--restore-from-D33", type=str, default=None,
                        help="Where restore model parameters from.")		
    parser.add_argument("--restore-from-D34", type=str, default=None,
                        help="Where restore model parameters from.")		

    parser.add_argument("--restore-from-D41", type=str, default=None,
                        help="Where restore model parameters from.")
    parser.add_argument("--restore-from-D42", type=str, default=None,
                        help="Where restore model parameters from.")	
    parser.add_argument("--restore-from-D43", type=str, default=None,
                        help="Where restore model parameters from.")		
    parser.add_argument("--restore-from-D44", type=str, default=None,
                        help="Where restore model parameters from.")	

    parser.add_argument("--restore-from-D_D31", type=str, default=None,
                        help="Where restore model parameters from.")
    parser.add_argument("--restore-from-D_D32", type=str, default=None,
                        help="Where restore model parameters from.")	
    parser.add_argument("--restore-from-D_D33", type=str, default=None,
                        help="Where restore model parameters from.")		
    parser.add_argument("--restore-from-D_D34", type=str, default=None,
                        help="Where restore model parameters from.")		

    parser.add_argument("--restore-from-D_D41", type=str, default=None,
                        help="Where restore model parameters from.")
    parser.add_argument("--restore-from-D_D42", type=str, default=None,
                        help="Where restore model parameters from.")	
    parser.add_argument("--restore-from-D_D43", type=str, default=None,
                        help="Where restore model parameters from.")		
    parser.add_argument("--restore-from-D_D44", type=str, default=None,
                        help="Where restore model parameters from.")	

						
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")

    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")

    return parser.parse_args()

args = get_arguments()

def loss_calc(pred, label):
    # label = Variable(label.long()).cuda(args.gpu)
    # criterion = CrossEntropy2d(ignore_label=args.ignore_label).cuda(args.gpu)  # Ignore label ??
    label = Variable(label.long()).cuda()
    criterion = CrossEntropy2d(ignore_label=args.ignore_label).cuda()  # Ignore label ??
    return criterion(pred, label)

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr*((1-float(iter)/max_iter)**(power))

def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1 :
        optimizer.param_groups[1]['lr'] = lr * 10

def adjust_learning_rate_D(optimizer, i_iter):
    lr = lr_poly(args.learning_rate_D, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1 :
        optimizer.param_groups[1]['lr'] = lr * 10

def one_hot(label):
    label = label.numpy()  # N,H,W
    one_hot = np.zeros((label.shape[0], args.num_classes, label.shape[1], label.shape[2]), dtype=label.dtype)  # N,C,H,W
    for i in range(args.num_classes):
        one_hot[:,i,...] = (label==i)
    #handle ignore labels
    return torch.FloatTensor(one_hot)

def compute_argmax_map(output):
    output = output.detach().cpu().numpy() #  c,H,W
    output = output.transpose((1,2,0))  # H,W,c
    output = np.asarray(np.argmax(output, axis=2), dtype=np.int) # H,W; obtain the index thatrepresented the max value through the axis==2 (i.e., channel)
    output = torch.from_numpy(output).float()  # numpy-->torch-->torch float 
    return output
     
def find_good_maps(D_outs, pred_all):
    count = 0
    for i in range(D_outs.size(0)):  # N,C
        if D_outs[i] > args.threshold_st:
            count +=1

    if count > 0:
        #print ('Above ST-Threshold : ', count, '/', args.batch_size)
        pred_sel = torch.Tensor(count, pred_all.size(1), pred_all.size(2), pred_all.size(3)) # n,c,h,w
        label_sel = torch.Tensor(count, pred_sel.size(2), pred_sel.size(3)) # n,h,w
        num_sel = 0 
        for j in range(D_outs.size(0)):
            if D_outs[j] > args.threshold_st:
                pred_sel[num_sel] = pred_all[j]  # get the pred_all[*] map large than threshold value 
                label_sel[num_sel] = compute_argmax_map(pred_all[j]) # score map --> label map with channel==1

                num_sel +=1
        #return  pred_sel.cuda(args.gpu), label_sel.cuda(args.gpu), count  
        return  pred_sel.cuda(), label_sel.cuda(), count  		
    else:
        return 0, 0, count 

		
def compute_ignore_mask(pred0, max_pred):
    pred0 = pred0.detach() # c,H,W    
    pred = torch.chunk(torch.squeeze(pred0,0),2,dim=0)
    pred_1 = torch.squeeze(pred[0],0)	# 1,h,w-->h,w
    pred_1 = pred_1.cpu().numpy() 
    pred_1[pred_1 > args.threshold_value] = 0
    pred_1[pred_1 < 1-args.threshold_value] = 0
    pred_1[pred_1 > 0] = 255    #h,w
    max_pred = max_pred.cpu().numpy() 	
    mask = 	max_pred + pred_1
    mask[mask > 2] = 255  	
    mask =torch.from_numpy(mask) #h,w
    
    return mask	


def find_good_maps_new(D_outs, pred_all, pred_all_2):
    count = 0
    for i in range(D_outs.size(0)):  # N,C
        if D_outs[i] > args.threshold_st:
            count +=1

    if count > 0:
        #print ('Above ST-Threshold : ', count, '/', args.batch_size)
        pred_sel = torch.Tensor(count, pred_all.size(1), pred_all.size(2), pred_all.size(3)) # n,c,h,w
        label_sel = torch.Tensor(count, pred_sel.size(2), pred_sel.size(3)) # n,h,w
        num_sel = 0 
        for j in range(D_outs.size(0)):
            if D_outs[j] > args.threshold_st:
                pred_sel[num_sel] = pred_all[j]  # c,h,w; get the pred_all[*] map large than threshold value 
                #label_sel[num_sel] = compute_argmax_map(pred_all[j]) # H,W; score map --> label map with channel==1
                label_sel[num_sel] = compute_ignore_mask( pred_all_2[j], compute_argmax_map(pred_all[j]) )
                num_sel +=1
        #return  pred_sel.cuda(args.gpu), label_sel.cuda(args.gpu), count  
        return  pred_sel.cuda(), label_sel.cuda(), count  		
    else:
        return 0, 0, count 
        
				
def bce_loss(y_pred, y_label):
    y_truth_tensor = torch.FloatTensor(y_pred.size())
    y_truth_tensor.fill_(y_label)
    y_truth_tensor = y_truth_tensor.to(y_pred.get_device())
    return nn.BCEWithLogitsLoss()(y_pred, y_truth_tensor)
    
def prob_2_entropy(prob):
    """ convert probabilistic prediction maps to weighted self-information maps
    """
    n, c, h, w = prob.size()
    return -torch.mul(prob, torch.log2(prob + 1e-30)) / np.log2(c)    



criterion = nn.BCELoss()

def main():
    print (args)

    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)

    cudnn.enabled = True
    #gpu = args.gpu

    # create network
    model = CDnetV1(num_classes=args.num_classes)
	
    # load pretrained parameters
    saved_state_dict = torch.load(args.restore_from)
    new_params = model.state_dict().copy()
    for name, param in new_params.items():
        if name in saved_state_dict and param.size() == saved_state_dict[name].size():
            new_params[name].copy_(saved_state_dict[name])
    model.load_state_dict(new_params)
	
    #model.load_state_dict(torch.load('./checkpoints/checkpoints_zy3_CDnetV1/VOC_2w.pth'))	
	
    model.train()
    #model.cuda(args.gpu)
    model.cuda()
    cudnn.benchmark = True


	
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
		
    # load data and do preprocessing,such as rescale,flip
    if args.dataset == 'pascal_voc':    
        # train_dataset = VOCDataSet(args.land_data_dir, args.data_list, crop_size=input_size,
                        # scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN2)  # landsat-8 data , labeled
        # train_remain_gt_dataset = VOCDataSet(args.gf_data_dir, args.data_list2, crop_size=input_size,
                        # scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN)	 # zy3 data GT,	 labeled				
        train_dataset = VOCDataSet(args.gf_data_dir, args.data_list3, crop_size=input_size,
                        scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN)  # zy3 data for training ,unlabeled
						

    elif args.dataset == 'pascal_context':
        input_transform = transform.Compose([transform.ToTensor(),
            transform.Normalize([.406, .456, .485], [.229, .224, .225])])
        # data_kwargs = {'transform': input_transform, 'base_size': 505, 'crop_size': 321}
        # #train_dataset = get_segmentation_dataset('pcontext', split='train', mode='train', **data_kwargs)
        # data_loader = get_loader('pascal_context')
        # data_path = get_data_path('pascal_context') 
        # train_dataset = data_loader(data_path, split='train', mode='train', **data_kwargs)
        # #train_gt_dataset = data_loader(data_path, split='train', mode='train', **data_kwargs)
        
    # elif args.dataset == 'cityscapes':
        # data_loader = get_loader('cityscapes')
        # data_path = get_data_path('cityscapes')
        # data_aug = Compose([RandomCrop_city((256, 512)), RandomHorizontallyFlip()])
        # train_dataset = data_loader( data_path, is_transform=True, augmentations=data_aug) 
        # #train_gt_dataset = data_loader( data_path, is_transform=True, augmentations=data_aug) 

    train_dataset_size = len(train_dataset)
    print ('dataset size: ', train_dataset_size)

    if args.labeled_ratio is None:
        trainloader = data.DataLoader(train_dataset,
                        batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    else:
        partial_size = int(args.labeled_ratio * train_dataset_size)

    trainloader_iter = iter(trainloader)
	
	
    # trainloader_gt_iter = iter(trainloader_gt)

    # optimizer for segmentation network
    optimizer = optim.SGD(model.parameters(),
                lr=args.learning_rate, momentum=args.momentum,weight_decay=args.weight_decay)  # if weight_decay=0.0，there is no regularization term
    optimizer.zero_grad()


	
    interp = nn.Upsample(size=(input_size[0], input_size[1]), mode='bilinear', align_corners=True)


    for i_iter in  range(args.num_steps+1):
    #for i_iter in  range(60001, args.num_steps+1):
        loss_ce_value = 0
        loss_pred_value = 0		
        loss_aux1_value = 0		
        loss_aux2_value = 0		
        loss_aux3_value = 0		

		
        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter)
        
        try:
            batch = next(trainloader_iter)
        except:
            trainloader_iter = iter(trainloader)
            batch = next(trainloader_iter)

        images, labels, _, _, _ = batch
        images = Variable(images).cuda()
        #feature_img3, feature_img4, pred = model(images)
        pred, pred_aux1, pred_aux2, pred_aux3 = model(images)	
		
        pred = interp(pred)		
        pred_aux1 = interp(pred_aux1)	
        # pred_aux2 = interp(pred_aux2)	
        # pred_aux3 = interp(pred_aux3)	
		
		# Cross entropy loss 
        loss_pred = loss_calc(pred, labels) # Cross entropy loss for labeled data
		
        loss_aux1 = loss_calc(pred_aux1, labels) 		
        # loss_aux2 = loss_calc(pred_aux2, labels) 
        # loss_aux3 = loss_calc(pred_aux3, labels) 


		
        loss_ce = loss_pred + 1*(loss_aux1 ) #+ loss_aux2 +loss_aux3)
		
        loss_ce.backward()
		
        loss_ce_value += loss_ce.item()
        loss_pred_value += loss_pred.item()		
        loss_aux1_value += loss_aux1.item()
        # loss_aux2_value += loss_aux2.item()
        # loss_aux3_value += loss_aux3.item()

		
        optimizer.step()

		
        if i_iter %20 ==0:
            #print('iter={0:5d},l_ce={1:.3f}, l_pred={2:.3f}, l_aux1={3:.3f}, l_aux2={4:.3f}, l_aux3={5:.3f},'.format(i_iter, loss_ce_value, loss_pred_value, loss_aux1_value, loss_aux2_value, loss_aux3_value, loss_aux1_value ))
			
            print('iter={0:5d}, l_ce={1:.3f}, l_pred={2:.3f}, l_aux1={3:.3f}'.format(i_iter, loss_ce_value,  loss_pred_value, loss_aux1_value))
			
        if i_iter >= args.num_steps-1:
            print ('save model ...')
            torch.save(model.state_dict(),os.path.join(args.checkpoint_dir, 'VOC_'+str(args.num_steps)+'.pth'))
            #torch.save(model_D1.state_dict(),os.path.join(args.checkpoint_dir, 'VOC_'+str(args.num_steps)+'_D1.pth'))			
				
			
            break

        if i_iter % args.save_pred_every == 0 and i_iter!=0:
            print ('saving checkpoint  ...')
            torch.save(model.state_dict(),os.path.join(args.checkpoint_dir, 'VOC_'+str(i_iter)+'.pth'))
            #torch.save(model_D1.state_dict(),os.path.join(args.checkpoint_dir, 'VOC_'+str(i_iter)+'_D1.pth'))			
	
			
    end = timeit.default_timer()
    print (end-start,'seconds')

if __name__ == '__main__':
    main()
