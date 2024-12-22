import argparse
import torch
from torch import nn
from torch.autograd import Variable
import torchvision
from torchvision.ops import nms
import os
from PIL import Image
import numpy as np
from skimage.draw import rectangle_perimeter
from skimage.transform import resize

from models.HRRegionNet import HRRegionNet
from utils.utility import csv_preprocess, calc_iou, soft_nms_pytorch

input_size = 512
output_size = input_size // 2

test_tx = torchvision.transforms.Compose([
        torchvision.transforms.Resize((input_size, input_size)),
        torchvision.transforms.Grayscale(num_output_channels=3),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('divece: ', device)

def main(args):
    
    val_list = csv_preprocess(args.csv_path)
    print("found", len(val_list), "of images")
    
    if not (args.log_dir == None):
        print("Load checkpoint from " + args.log_dir)
        checkpoint = torch.load(args.log_dir, map_location="cpu")    

    model = HigherHRCenterNet()
    model = nn.DataParallel(model)
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()
    
    iou_sum = 0.
    
    for i in range(len(val_list)):
        print(args.data_dir + val_list[i][0])
        img = Image.open(args.data_dir + val_list[i][0]).convert("RGB")
    
        image_tensor = test_tx(img)
        image_tensor = image_tensor.unsqueeze_(0)
        inp = Variable(image_tensor)
        inp = inp.to(device, dtype=torch.float)
        predict = model(inp)
        
        iou = _nms(args, img, predict, val_list, i, nms_score=0.4, iou_threshold=0.1)
        print('IoU = ', iou)
        
        iou_sum = iou_sum + iou
        
    print('Average IoU: ', iou_sum / len(val_list))
    
def _nms(args, img, predict, val_list, dindex, nms_score, iou_threshold):
    
    bbox = list()
    score_list = list()
    im_draw = np.asarray(torchvision.transforms.functional.resize(img, (img.size[1], img.size[0]))).copy()
    
    heatmap1 = predict[0].data.cpu().numpy()[0, 0, ...]
    heatmap2 = predict[1].data.cpu().numpy()[0, 0, ...]
    
    heatmap = (resize(heatmap1, (output_size, output_size)) +  heatmap2) / 2
#     heatmap = heatmap1
    
    offset_y = predict[1].data.cpu().numpy()[0, 1, ...]
    offset_x = predict[1].data.cpu().numpy()[0, 2, ...]
    width_map = predict[1].data.cpu().numpy()[0, 3, ...]
    height_map = predict[1].data.cpu().numpy()[0, 4, ...]
    
    for j in np.where(heatmap.reshape(-1, 1) >= nms_score)[0]:

        row = j // output_size 
        col = j - row*output_size
        
        bias_y = offset_y[row, col] * (img.size[0] / output_size)
        bias_x = offset_x[row, col] * (img.size[1] / output_size)

        width = width_map[row, col] * output_size * (img.size[0] / output_size)
        height = height_map[row, col] * output_size * (img.size[1] / output_size)

        score_list.append(heatmap[row, col])

        row = row * (img.size[1] / output_size) + bias_y
        col = col * (img.size[0] / output_size) + bias_x

        top = row - height // 2
        left = col - width // 2
        bottom = row + height // 2
        right = col + width // 2

        start = (top, left)
        end = (bottom, right)

        bbox.append([top, left, bottom, right])
    
    if len(bbox) == 0:
        print('No object was found in the image')
        bbox.append([0, 0, 0, 0])
        score_list.append(0)
        
    _nms_index = torchvision.ops.nms(torch.FloatTensor(bbox), scores=torch.flatten(torch.FloatTensor(score_list)), iou_threshold=iou_threshold)
#     _nms_index = soft_nms_pytorch(torch.FloatTensor(bbox), torch.flatten(torch.FloatTensor(score_list)), thresh=nms_score, cuda=0)
    
    for k in range(len(_nms_index)):
    
        top, left, bottom, right = bbox[_nms_index[k]]
        
        start = (top, left)
        end = (bottom, right)
        
        rr, cc = rectangle_perimeter(start, end=end,shape=(img.size[1], img.size[0]))
        
        im_draw[rr, cc] = (255, 0, 0)
        
    if args.save_img:
        for j in range(1, len(val_list[dindex][1])):
            
            for thickness in range(2):
                x, y, width, height = val_list[dindex][1][j][1:5]

                top = y - height // 2 - thickness
                left = x - width // 2 - thickness
                bottom = y + height // 2 + thickness
                right = x + width // 2 + thickness

                start = (int(top), int(left))
                end = (int(bottom), int(right))

                rr, cc = rectangle_perimeter(start, end=end, shape=(img.size[1], img.size[0]))

                im_draw[rr, cc] = (0, 0, 255)

        print('save image to ', args.output_dir + val_list[dindex][0])
        Image.fromarray(im_draw).save(args.output_dir + val_list[dindex][0])
        
    iou = calc_iou(bbox, _nms_index, val_list, dindex, imshape=(img.size[1], img.size[0]))
        
    return iou


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HRCenterNet.")
    
    parser.add_argument("--data_dir", required=True,
                      help="Path to the testing images folder, preprocessed for torchvision.")
    
    parser.add_argument("--csv_path", required=True,
                       help="Path to the csv file for evaluation")
    
    parser.add_argument("--log_dir", required=True, default=None,
                      help="Where to load for the pretrained model.")
    
    parser.add_argument("--output_dir", default='./',
                      help="Where to save for the outputs.")
    
    parser.add_argument('--save_img', default=False, action='store_true',
                       help="save image draw with the predictions and ground-truth labeling")

    main(parser.parse_args())
