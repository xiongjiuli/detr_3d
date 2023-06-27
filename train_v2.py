import os 
import argparse
import torchio as tio
from torchvision import transforms as T
import torch 
from tqdm import tqdm 
import numpy as np
import torch.nn.functional as F 
from IPython import embed
import logging
import torch.utils.data as data
from time import time
from dataloader_v1 import DETR3DDataset, detr_dataset_collate
from torch.utils.data import DataLoader

# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
# from utils import npy2nii
from detr import *

def log_create(log_path):
    # 创建一个logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 创建一个handler，用于将日志输出到控制台
    # ch = logging.StreamHandler()
    # ch.setLevel(logging.INFO)

    # 创建一个handler，用于将日志写入到文件中
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)

    # 定义handler的输出格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    # 将handler添加到logger中
    # logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
def save(step, opt, model, out_path, name='model'):
        data = {
            'step': step,
            'model': model.state_dict(),
            'opt': opt.state_dict()
        }
        torch.save(data, os.path.join(out_path, f'{name}-{step}.pt'))  


def train(config):
    
    detr_loss = build_loss(1).cuda()
    logger = log_create(config.log_path)
    model = DETR(backbone='resnet50', position_embedding='sine', hidden_dim=256, num_classes=1, num_queries=100)
    traindataset = DETR3DDataset(config, mode='train')
    validdataset = DETR3DDataset(config, mode='valid')
    trainloader = DataLoader(traindataset, batch_size=config.train_batch_size, shuffle=True, collate_fn=detr_dataset_collate, drop_last=True)
    validloader = DataLoader(validdataset, batch_size=config.valid_batch_size, shuffle=False, collate_fn=detr_dataset_collate)

    time_cuda = time()
    model.cuda()
    print(time() - time_cuda)
 
    if config.model != 'normal':
        model_path = '/public_bme/data/xiongjl/det/save/best_model-120.pt'
        model.load_state_dict(torch.load(model_path)['model'])

    optimizer = torch.optim.Adam(model.parameters(), config.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config.scheduler_steps, gamma=config.gamma)


    for epoch in range(config.total_steps):

        best_loss = 1e4
        print('Start Validation')
        pbar = tqdm(total=config.train_number//config.train_batch_size, desc=f'Epoch {epoch + 1}/{config.total_steps}',postfix=dict,mininterval=0.3)


        if (epoch) % config.save_freq == 0:
            model.eval()
            val_loss = 0
            os.makedirs(config.save_dir, exist_ok=True)
            save(epoch, optimizer, model, config.save_dir, name=config.save_model_name)
            with torch.no_grad():
                val_step = 1
                for iteration, valid_batch in enumerate(validloader):
                    image = valid_batch['image']
                    label = valid_batch['label']
                    image = image.float().cuda()
                    #----------------------#
                    #   清零梯度
                    #----------------------#
                    optimizer.zero_grad()
                    #----------------------#
                    #   前向传播
                    #----------------------#
                    out = model(image)
                    valid_loss  = detr_loss(out, label)
                    weight_dict = detr_loss.weight_dict
                    # embed()
                    print(f'valid loss_ce loss : {valid_loss["loss_ce"]}\n')
                    print(f'valid class_error loss : {valid_loss["class_error"]}\n')
                    print(f'valid loss_giou loss : {valid_loss["loss_giou"]}\n')
                    print(f'valid loss_bbox loss : {valid_loss["loss_bbox"]}\n')
                    print(f'valid cardinality_error loss : {valid_loss["cardinality_error"]}\n')
                    valid_loss = sum(valid_loss[k] * weight_dict[k] for k in valid_loss.keys() if k in weight_dict)
                    # embed()
                val_loss += valid_loss.item()
                val_step += 1

                # logger.info('Epoch: %d, valid_Loss: %.4f', step, np.mean(valid_loss))

                if best_loss > valid_loss.item():
                    best_loss = valid_loss.item()
                    os.makedirs(config.save_dir, exist_ok=True)
                    save(epoch, optimizer, model, config.save_dir, name=config.best_model_name)
                
                    
                pbar.set_postfix(**{'loss'  : val_loss / (iteration + 1), 
                                    'lr'    : get_lr(optimizer)})
                pbar.update(1)       
                pbar.close()
                print('Finish Validation')

        print('Start Train')       
        model.train()
        pbar = tqdm(total=config.train_number//config.train_batch_size, desc=f'Epoch {epoch + 1}/{config.total_steps}',postfix=dict,mininterval=0.3)
        train_loss = 0
        # train_batch = next(train_loader)
        for iteration, train_batch in enumerate(trainloader):

            images, targets = train_batch['image'], train_batch['label']
            images = images.float().cuda()
            with torch.no_grad():
                #----------------------#
                #   清零梯度
                #----------------------#
                optimizer.zero_grad()
                #----------------------#
                #   前向传播
                #----------------------#
                outputs     = model(images)
                loss_value  = detr_loss(outputs, targets)
                weight_dict = detr_loss.weight_dict
                loss_value  = sum(loss_value[k] * weight_dict[k] for k in loss_value.keys() if k in weight_dict)

                #----------------------#
                #   反向传播
                #----------------------#
                # print(loss_value)
                loss_value.requires_grad = True
                loss_value.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                optimizer.step()
                
            lr_scheduler.step()  
            train_loss += loss_value.item()
            logger.info(f'{iteration}/{epoch} -- train loss : {loss_value}')
            pbar.set_postfix(**{'loss'  : train_loss / (iteration + 1), 
                                'lr'    : get_lr(optimizer)})
            pbar.update(1)

        pbar.close()
        print('Finish Train')
        # pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="detr")
    parser.add_argument('--log_dir', default='./log')
    parser.add_argument('--seed', default=1)
    parser.add_argument('--train_batch_size', default=4)
    parser.add_argument('--valid_batch_size', default=4)
    parser.add_argument('--backbone_name', default='resnet101')
    parser.add_argument('--num_classes', default=1)
    parser.add_argument('--lr', default=1e-4)
    parser.add_argument('--scheduler_steps', default=10000)
    parser.add_argument('--gamma', default=0.1)
    parser.add_argument('--total_steps', default=1000)
    parser.add_argument('--valid_steps', default=1000)
    parser.add_argument('--save_freq', default=20)
    parser.add_argument('--save_dir', default='/public_bme/data/xiongjl/detr/save')
    parser.add_argument('--model', default='normal')
    parser.add_argument('--train_number', default=300)
    parser.add_argument('--valid_number', default=50)

    parser.add_argument('--log_path', default='./log/training_v1_nopool_crop.log')
    parser.add_argument('--best_model_name', default='best_detr_v1_nopool_crop')
    parser.add_argument('--save_model_name', default='detr_v1_nopool_crop')
    
    parser.add_argument('--label_path', default='/public_bme/data/xiongjl/detr/data/annotation_v1.csv')
    parser.add_argument('--image_path', default='/public_bme/data/xiongjl/det/nii_data_resample_seg_crop/')
    parser.add_argument('--crop_length', default=128)
    args = parser.parse_args()
    
    train(args)
                
        
    
