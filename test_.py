import argparse
import torch
from torch.autograd import Variable
import torchvision
from torchvision.ops import nms
import os
from PIL import Image
import numpy as np
from skimage.draw import rectangle_perimeter
import matplotlib.pyplot as plt
import cv2

from models.HRCenterNet_ca_T2_Bottle2neck import HRCenterNet

input_size = 512
output_size = 256

test_tx = torchvision.transforms.Compose([
    torchvision.transforms.Resize((input_size, input_size)),
    torchvision.transforms.ToTensor(),
])

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
print('divece: ', device)


def main():

    log_dir = r'E:\gbw\HRC\HRCenterNet\weight\best.pth.tar'

    if not (log_dir == None):
        print("Load checkpoint from " + log_dir)
        checkpoint = torch.load(log_dir, map_location="cpu")

    model = HRCenterNet()
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()

    out_path = r'E:\gbw\HRC\HRCenterNet\out_val_50'

    if not os.path.exists(out_path):
        os.makedirs(out_path)
        os.makedirs(out_path + '/1/')
        os.makedirs(out_path + '/cropped/')

    # 手动输入单个图像的路径
    image_path = input("请输入图像文件的路径（jpg/png）: ").strip()

    if os.path.isfile(image_path):
        if image_path.lower().endswith(('jpg', 'png')):
            try:
                # 打开图像并转换为RGB模式
                img = Image.open(image_path).convert("RGB")
                image_tensor = test_tx(img)
                image_tensor = image_tensor.unsqueeze_(0)

                inp = Variable(image_tensor)
                inp = inp.to(device, dtype=torch.float)
                predict = model(inp)

                # 进行非最大抑制处理
                out_img, flag = _nms(img, predict, out_path, os.path.basename(image_path), nms_score=0.3, iou_threshold=0.1)
                if flag == 1:
                    output_file_path = os.path.join(out_path, '1', os.path.basename(image_path))
                    print('保存图像至', output_file_path)
                    Image.fromarray(out_img).save(output_file_path)
                else:
                    print('图像处理失败。')
            except Exception as e:
                print(f"处理图像时出错: {e}")
        else:
            print(f"错误: {image_path} 不是支持的图像格式（jpg/png）。")
    else:
        print(f"错误: {image_path} 不是有效的文件路径。")

def _nms(img, predict, out_path, file, nms_score, iou_threshold):
    bbox = list()
    score_list = list()
    im_draw = np.asarray(torchvision.transforms.functional.resize(img, (img.size[1], img.size[0]))).copy()

    heatmap = predict.data.cpu().numpy()[0, 0, ...]
    offset_y = predict.data.cpu().numpy()[0, 1, ...]
    offset_x = predict.data.cpu().numpy()[0, 2, ...]
    width_map = predict.data.cpu().numpy()[0, 3, ...]
    height_map = predict.data.cpu().numpy()[0, 4, ...]

    for j in np.where(heatmap.reshape(-1, 1) >= nms_score)[0]:
        row = j // output_size
        col = j - row * output_size

        bias_x = offset_x[row, col] * (img.size[1] / output_size)
        bias_y = offset_y[row, col] * (img.size[0] / output_size)

        width = width_map[row, col] * output_size * (img.size[1] / output_size)
        height = height_map[row, col] * output_size * (img.size[0] / output_size)

        score_list.append(heatmap[row, col])

        row = row * (img.size[1] / output_size) + bias_y
        col = col * (img.size[0] / output_size) + bias_x

        top = row - width // 2
        left = col - height // 2
        bottom = row + width // 2
        right = col + height // 2

        start = (top, left)
        end = (bottom, right)

        bbox.append([top, left, bottom, right])

    if len(torch.FloatTensor(bbox).shape) == 2:

        flag = 1
        
        _nms_index = torchvision.ops.nms(torch.FloatTensor(bbox), scores=torch.flatten(torch.FloatTensor(score_list)),
                                        iou_threshold=iou_threshold)

        for k in range(len(_nms_index)):
            top, left, bottom, right = bbox[_nms_index[k]]

            start = (top, left)
            end = (bottom, right)

            rr, cc = rectangle_perimeter(start, end=end, shape=(img.size[1], img.size[0]))

            im_draw[rr, cc] = (255, 0, 0)

            cropped = img.crop((left, top, right, bottom))

            cropped.save(str(out_path) + '/cropped/' + file[:-4] + '_' +'%d.jpg' % k)
    else:
        flag = 0
    return im_draw, flag

if __name__ == "__main__":
    main()
