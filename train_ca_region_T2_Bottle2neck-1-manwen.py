"""
    @author mhuazheng 
    @create 2021-10-09 21:07 
    
"""

import torch, gc

gc.collect()
torch.cuda.empty_cache()

import argparse
import torch
import sys
from tqdm import tqdm  # 进度条
import os

from torch.utils.tensorboard import SummaryWriter

from datasets.HanDataset import dataset_generator
from utils.utility import csv_preprocess, _nms_eval_iou
from utils.losses_region import calc_loss
#from models.HRCenterNet_ca_T2_Bottle2neck import HRCenterNet
from models.HRCenterNet_ca_T2_Bottle2neck import HRCenterNet
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

crop_size = 512
output_size = 256

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device: ', device)


def main():
    dataloader = dict()

    # 训练集预处理 
    #train_csv_path = 'data_man/train_580.csv'

    #先在汉字的基础上训练200轮
    #train_csv_path = '/mnt/big_disk/maxiaoxuan/data/handata/train.csv'
    
    train_csv_path = '/mnt/sda/gbw/HRCenterNet/datasets/xxshanzi/train_100.csv'
    
    train_list = csv_preprocess(train_csv_path)
    print("found", len(train_list), "of images for training")

    #train_data_dir = 'data_man/train_580'
    #师兄的
    
    #/mnt/big_disk/maxiaoxuan/data/manwen_demo/train/
    #train_data_dir = '/mnt/big_disk/maxiaoxuan/data/handata/images'
    train_data_dir = '/mnt/sda/gbw/HRCenterNet/datasets/xxshanzi/train_100/'
    train_set = dataset_generator(train_data_dir, train_list, crop_size, 0.5, output_size, train=True)
    bz_bz=16
    # 训练数据集的加载器，自动将数据分割成batch，顺序随机打乱
    dataloader['train'] = torch.utils.data.DataLoader(train_set, batch_size=bz_bz, shuffle=True)

    val = False#？？？？定义的是true 学长原本
    if val:
        # 验证集预处理
        val_csv_path = '/mnt/sda/gbw/HRCenterNet/datasets/xxshanzi/val_50.csv'
        val_list = csv_preprocess(val_csv_path)
        print("found", len(val_list), "of images for validation")

        val_data_dir = '/mnt/sda/gbw/HRCenterNet/datasets/xxshanzi/val_50/'
        val_set = dataset_generator(val_data_dir, val_list, crop_size, 0, output_size, train=False)
        dataloader['val'] = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=False)

    checkpoint = None
    # log_dir = "weights/HRCenterNet.pth.tar"
    # log_dir = "weights/古代汉字/ca_region_T2-b24-lr4_0.8725.pth.tar"
    # log_dir = "weights/ca_region_T2-b24-lr4_0.8725.pth.tar"
    # log_dir = "weights/ca_region_T2_Bottle2neck-b22-lr2_0.882.pth.tar" ??????????
    log_dir = "/mnt/sda/gbw/HRCenterNet_/xxs_best.pth.tar"
    del_key = []
    if not (log_dir == None):
        checkpoint = torch.load(log_dir, map_location=device)

        for key, value in list(checkpoint['model'].items()):
            if "layer1" in key:
                del_key.append(key)
                del checkpoint['model']['%s' % key]

    weight_dir = './weights/323/'
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)

    train(dataloader, checkpoint)


def train(dataloader, checkpoint=None):
    num_epochs = 201
    loss_average = 0.
    metrics = dict()

    model = HRCenterNet()

    # 获取模型权重信息
    model_weights = model.state_dict()
        
    #这里不知道学长在干什么所以删掉
    new_weights = list(checkpoint['model'].items())

    i, j = 0, 0
    for key, value in list(model_weights.items()):
        i += 1
        if "layer1" in key:
            # start = i
            # j += 1
            res_weights = list(model_weights.items())[i - 1]
            new_weights.insert(i - 1, res_weights)

    # res_weights = list(model_weights.items())[start - j:start]

    new_weights = dict(new_weights)

    checkpoint["model"].update(new_weights)
    
        
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-7)
    model = model.to(device)



    if not (checkpoint == None):
        # log_dir = "weights/HRCenterNet.pth.tar"
        # log_dir = "weights/古代汉字/ca_region_T2-b24-lr4_0.8725.pth.tar"
        log_dir = "/mnt/sda/gbw/HRCenterNet/weights/best.pth.tar"
        print("Load checkpoint from " + log_dir)
        model.load_state_dict(checkpoint['model'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        # best_iou = checkpoint['best_iou']

    optimizer.zero_grad()

    best_iou = 0
    for epoch in range(num_epochs):
        loss = 0.

        """
        model.train()
        在使用pytorch构建神经网络的时候，训练过程中会在程序上方添加一句model.train()，作用是启用batch normalization和drop out。
        model.eval():
        测试过程中会使用model.eval()，这时神经网络会不启用 BatchNormalization 和 Dropout
        """
        model.train()

        for batch_idx, sample in enumerate(dataloader['train']):
            inputs = sample['image'].to(device, dtype=torch.float)
            labels = sample['labels'].to(device, dtype=torch.float)

            outputs = model(inputs)

            loss = calc_loss(outputs, labels, metrics)
     
        
              
              
             
            bz=2    
      
            
            
            
            
            
            
            
            loss_average = loss_average + metrics['loss']
            sys.stdout.write('\r')
            sys.stdout.write(
                'Training: Epoch[%3d/%3d] Iter[%3d/%3d] Loss: %.5f heatmap_loss: %.4f size_loss: %.4f offset_loss: %.4f region_loss: %.4f'
                % (epoch + 1, num_epochs, batch_idx, (len(dataloader['train'].dataset) // (bz)) + 1,
                   metrics['loss'], metrics['heatmap'], metrics['size'], metrics['offset'], metrics['region']))
            sys.stdout.write(' average loss: %.5f' % (loss_average / (
                    ((len(dataloader['train'].dataset) // (bz)) + 1) * epoch + (batch_idx + 1))))
            sys.stdout.flush()

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

        print()
        val = False
        #学长设置的是true val = True
        if val:
            avg_iou = evaluate(dataloader, model)
            print('Average IoU: ', avg_iou)
            weight_dir = './weights/320/'
            if avg_iou > best_iou:
                print('IoU improve from', best_iou, 'to', avg_iou)
                best_iou = avg_iou
                print('Saving model to', weight_dir, 'best.pth.tar')
                torch.save({'best_iou': best_iou,
                            'optimizer': optimizer.state_dict(),
                            'model': model.state_dict()},
                           weight_dir + 'best.pth.tar')
        weight_dir = './weights/322/'
        if (epoch % 10) == 0:
            print('Saving model to {}{}.pth.tar'.format(weight_dir, str(epoch)))
            torch.save({'best_iou': best_iou,
                        'optimizer': optimizer.state_dict(),
                        'model': model.state_dict()},
                       weight_dir + str(epoch) + '.pth.tar')


def evaluate(dataloader, model):
    iou_sum = 0.
    model.eval()

    for batch_idx, sample in enumerate(tqdm(dataloader['val'], ascii=True, desc='Evaluation')):
        with torch.no_grad():
            inputs = sample['image'].to(device, dtype=torch.float)
            labels = sample['labels'].to(device, dtype=torch.float)
            outputs = model(inputs)
            img_width, img_height = sample['img_size']
            iou = _nms_eval_iou(labels, outputs, img_width.item(), img_height.item(), output_size, nms_score=0.3,
                                iou_threshold=0.1)

            iou_sum = iou_sum + iou

    return iou_sum / len(dataloader['val'].dataset)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="HRCenterNet.")
    #
    # parser.add_argument("--train_data_dir", required=True,
    #                     help="Path to the training images folder, preprocessed for torchvision.")
    #
    # parser.add_argument("--train_csv_path", required=True,
    #                     help="Path to the csv file for training")
    #
    # parser.add_argument('--val', default=False, action='store_true',
    #                     help="traing with validation")
    #
    # parser.add_argument("--val_data_dir", required=False,
    #                     help="Path to the validated images folder, preprocessed for torchvision.")
    #
    # parser.add_argument("--val_csv_path", required=False,
    #                     help="Path to the csv file for validation")
    #
    # parser.add_argument("--log_dir", required=False, default=None,
    #                     help="Where to load for the pretrained model.")
    #
    # parser.add_argument("--epoch", type=int, default=201,
    #                     help="number of epoch")
    #
    # parser.add_argument("--lr", type=float, default=1e-6,
    #                     help="learning rate")
    #
    # parser.add_argument("--batch_size", type=int, default=16,
    #                     help="number of batch size")
    #
    # parser.add_argument("--crop_ratio", type=float, default=0.5,
    #                     help="crop ration for random crop in data augumentation")
    #
    # # parser.add_argument('--weight_dir', default='/mnt/big_disk/2020/zhengminghua/HRCenterNet/weights/',
    # #                     help="Where to save the weight")
    # parser.add_argument('--weight_dir', default='./weights/',
    #                     help="Where to save the weight")
    #
    # parser.add_argument('--save_epoch', type=int, default=10,
    #                     help="save model weight every number of epoch")

    main()

# python train.py --train_csv_path data/train.csv --train_data_dir data/images --val --val_csv_path data/val.csv --val_data_dir data/images/ --batch_size 8 --epoch 80
# python train.py --train_csv_path data_Demo/2/train.csv --train_data_dir data_Demo/2/images --log_dir weights/HRCenterNet.pth.tar --val_csv_path data_Demo/2/val.csv --val_data_dir data_Demo/2/images --batch_size 8 --epoch 101
# python train.py --train_csv_path data_Demo/1/train.csv --train_data_dir data_Demo/1/images --log_dir weights/HRCenterNet.pth.tar --batch_size 8 --epoch 101


# python train.py --train_csv_path data_Demo/2/train.csv --train_data_dir data_Demo/2/images --val --val_csv_path data_Demo/2/val.csv --val_data_dir data_Demo/2/images --log_dir weights/HRCenterNet.pth.tar --batch_size 8 --epoch 101


# /mnt/big_disk/2020/zhengminghua/HRCenterNet/weights/HRCenterNet.pth.tar
# python train_ca.py --train_csv_path data_Demo/4/train_80.csv --train_data_dir data_Demo/4/train_80 --val --val_csv_path data_Demo/4/val_20.csv --val_data_dir data_Demo/4/val_20 --log_dir /mnt/big_disk/2020/zhengminghua/HRCenterNet/weights/HRCenterNet.pth.tar --batch_size 8 --epoch 81


# python train_region_T2.py --train_csv_path data/train.csv --train_data_dir data/images --val --val_csv_path data/val.csv --val_data_dir data/images/ --log_dir weights/HRCenterNet.pth.tar --batch_size 8 --epoch 80
