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
from tqdm import tqdm  # 进度条

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
        os.makedirs(out_path + '/0/')
        os.makedirs(out_path + '/cropped/')

    test_path = r'E:\gbw\HRC\HRCenterNet\datasets\xxshanzi\val_50'

    for file in tqdm(os.listdir(test_path)):
        if file[-3:] == 'jpg' or file[-3:] == 'png':

            img = Image.open(str(test_path) + '/' + file).convert("RGB")
            image_tensor = test_tx(img)
            image_tensor = image_tensor.unsqueeze_(0)
            """
            在pytorch中的Variable就是一个存放会变化值的地方，里面的值会不停发生变化,
            pytorch都是有tensor计算的，而tensor里面的参数都是Variable的形式,这正好就符合了反向传播，参数更新的属性
            """
            inp = Variable(image_tensor)
            inp = inp.to(device, dtype=torch.float)
            predict = model(inp)

            out_img, flag = _nms(img, predict, out_path, file, nms_score=0.3, iou_threshold=0.1)
            if flag == 1: 
                print('saving image to ', str(out_path)+'/1/' + file)
                # plt.imshow(out_img)  # 显示图片
                # plt.axis('off')  # 不显示坐标轴
                # plt.show()

                # Image.fromarray的作用：简而言之，就是实现array到image的转换
                Image.fromarray(out_img).save(str(out_path)+'/1/' + file)
            else:
                print('saving image to ', str(out_path)+'/0/' + file)
                # plt.imshow(out_img)  # 显示图片
                # plt.axis('off')  # 不显示坐标轴
                # plt.show()

                # Image.fromarray的作用：简而言之，就是实现array到image的转换
                Image.fromarray(out_img).save(str(out_path)+'/0/' + file)
        else:
            print(str(file)+' 格式不对！')

def _nms(img, predict, out_path, file, nms_score, iou_threshold):
    bbox = list()
    score_list = list()
    im_draw = np.asarray(torchvision.transforms.functional.resize(img, (img.size[1], img.size[0]))).copy()

    heatmap = predict.data.cpu().numpy()[0, 0, ...]
    offset_y = predict.data.cpu().numpy()[0, 1, ...]
    offset_x = predict.data.cpu().numpy()[0, 2, ...]
    width_map = predict.data.cpu().numpy()[0, 3, ...]
    height_map = predict.data.cpu().numpy()[0, 4, ...]

    """
    np.where(condition, x, y)
    满足条件(condition)，输出x，不满足输出y。
    只有条件 (condition)，没有x和y，则输出满足条件 (即非0) 元素的坐标 (等价于numpy.nonzero)。
    这里的坐标以tuple的形式给出，通常原数组有多少维，输出的tuple中就包含几个数组，分别对应符合条件元素的各维坐标。
    """
    # a = heatmap.reshape(-1, 1)
    # b = np.where(heatmap.reshape(-1, 1) >= nms_score)

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
        
        # 首先找出score最高的框,然后依次计算候选框与最高score的IoU,最后删除大于阈值的框，这样做是删除重复多余的框
        _nms_index = torchvision.ops.nms(torch.FloatTensor(bbox), scores=torch.flatten(torch.FloatTensor(score_list)),
                                        iou_threshold=iou_threshold)

        for k in range(len(_nms_index)):
            top, left, bottom, right = bbox[_nms_index[k]]

            start = (top, left)
            end = (bottom, right)

            """
            rectangle_perimeter返回的矩形框上每个像素的坐标
            参数：
                start：矩形内部的原点，也就是左上角
                end：矩形内部的结点，也就是右下角
            """
            rr, cc = rectangle_perimeter(start, end=end, shape=(img.size[1], img.size[0]))

            im_draw[rr, cc] = (255, 0, 0)

            # cv2.rectangle(im_draw, start, end, color=(0, 250, 0), thickness=2)

            # plt.imshow(im_draw)  # 显示图片
            # plt.axis('off')  # 不显示坐标轴
            # plt.show()

            cropped = img.crop((left, top, right, bottom))

            # plt.imshow(cropped)  # 显示图片
            # plt.axis('off')  # 不显示坐标轴
            # plt.show()
            # Image.fromarray(out_img).save(args.output_dir + file)

            cropped.save(str(out_path) + '/cropped/' + file[:-4] + '_' +'%d.jpg' % k)
            # print('saving image to ', str(out_path) + '/cropped/' + file[:-4] + '_' + '%d.jpg' % k)
    else:
        flag = 0
    return im_draw, flag

if __name__ == "__main__":
    main()
