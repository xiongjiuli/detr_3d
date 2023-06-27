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
from IPython import embed


# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
# from utils import npy2nii

from detr import *


# 创建一个logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# 创建一个handler，用于将日志输出到控制台
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# 创建一个handler，用于将日志写入到文件中
fh = logging.FileHandler('/public_bme/data/xiongjl/detr/log/detr_v52nodes.log')
fh.setLevel(logging.INFO)

# 定义handler的输出格式
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
fh.setFormatter(formatter)

# 将handler添加到logger中
logger.addHandler(ch)
logger.addHandler(fh)




def save(step, opt, model, out_path, name='model'):

        data = {
            'step': step,
            'model': model.state_dict(),
            'opt': opt.state_dict()
        }

        torch.save(data, os.path.join(out_path, f'{name}-{step}.pt'))  


def train(config):
    

    detr_loss = build_loss(1).cuda()
    model = DETR(backbone='resnet50', position_embedding='sine', hidden_dim=256, num_classes=1, num_queries=10)

    time_cuda = time()
    model.cuda()
    print(time() - time_cuda)
 
    if config.model != 'normal':
        model_path = '/public_bme/data/xiongjl/det/save/best_model-120.pt'
        model.load_state_dict(torch.load(model_path)['model'])


    optimizer = torch.optim.Adam(model.parameters(), config.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config.scheduler_steps, gamma=config.gamma)
    
    # if config.pretrained_model:
    #     model.load_state_dict(torch.load(config.pretrained_model)['model'], strict=False)
    
    best_loss = 1e4

    with tqdm(total=(config.total_steps)) as pbar:
        for step in range(config.total_steps):

            if (step) % 200 == 0:
            # if (step) % 100 == 0:
                model.eval()
                os.makedirs(config.save_dir, exist_ok=True)
                save(step, optimizer, model, config.save_dir, name=config.save_model_name)


                with torch.no_grad():
                    loss_ce_loss = 0
                    class_error_loss = 0
                    loss_giou_loss = 0
                    loss_bbox_loss = 0
                    cardinality_error_loss = 0
                    for i in range(20):
                        
                        # val_step = 1
                        # embed()
                        inputs, targets = data_set()
                        inputs = inputs.float().cuda()
                        out = model(inputs)
                        valid_loss  = detr_loss(out, targets)
                        
                        loss_ce_loss += valid_loss["loss_ce"].item()
                        class_error_loss += valid_loss["class_error"].item()
                        loss_giou_loss += valid_loss["loss_giou"].item()
                        loss_bbox_loss += valid_loss["loss_bbox"].item()
                        cardinality_error_loss += valid_loss["cardinality_error"].item()
                        weight_dict = detr_loss.weight_dict
                        valid_loss = sum(valid_loss[k] * weight_dict[k] for k in valid_loss.keys() if k in weight_dict)

                    # targets = targets.cuda()
                    # embed()

                        

                    logger.info(f'valid loss_ce loss : {loss_ce_loss/20.}\n')
                    logger.info(f'valid class_error loss : {class_error_loss/20.}\n')
                    logger.info(f'valid loss_giou loss : {loss_giou_loss/20.}\n')
                    logger.info(f'valid loss_bbox loss : {loss_bbox_loss/20.}\n')
                    logger.info(f'valid cardinality_error loss : {cardinality_error_loss/20.}\n')
                    # valid_loss.append(loss.item())
                    # val_step += 1

                    # logger.info('Epoch: %d, valid_Loss: %.4f', step, np.mean(valid_loss))

                if best_loss > valid_loss.item():
                    best_loss = valid_loss.item()
                    os.makedirs(config.save_dir, exist_ok=True)
                    save(step, optimizer, model, config.save_dir, name=config.best_model_name)
                
                
                pbar.set_description(f'{step} - valid: {valid_loss:.4f}\n')
                # writer.add_scalar('valid/loss', np.mean(valid_loss), step)
                            
                                
            model.train()
            # batch_step = 0
            # train_loss = []
            # train_batch = next(train_loader)
            # for inputs, targets in train_dataset:
            loss_ce_loss = 0
            class_error_loss = 0
            loss_giou_loss = 0
            loss_bbox_loss = 0
            cardinality_error_loss = 0
            for i in range(20):
                inputs, targets = data_set()
                inputs = inputs.float().cuda()
                # targets = [target.cuda() for target in targets]


                out = model(inputs)
                train_loss  = detr_loss(out, targets)
                weight_dict = detr_loss.weight_dict

                loss_ce_loss += train_loss["loss_ce"].item()
                class_error_loss += train_loss["class_error"].item()
                loss_giou_loss += train_loss["loss_giou"].item()
                loss_bbox_loss += train_loss["loss_bbox"].item()
                cardinality_error_loss += train_loss["cardinality_error"].item()



                train_loss = sum(train_loss[k] * weight_dict[k] for k in train_loss.keys() if k in weight_dict)
            # train_loss.append(loss.item())
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                lr_scheduler.step()  
            
            # batch_step += 1
            # print(loss.item())
            logger.info(f'train loss_ce loss : {loss_ce_loss/20.}\n')
            logger.info(f'train class_error loss : {class_error_loss/20.}\n')
            logger.info(f'train loss_giou loss : {loss_giou_loss/20.}\n')
            logger.info(f'train loss_bbox loss : {loss_bbox_loss/20.}\n')
            logger.info(f'train cardinality_error loss : {cardinality_error_loss/20.}\n')

            
                 
            # loss /= len(train_loader)  
            # logger.info('Epoch: %d, train_Loss: %.4f', step, np.mean(train_loss))
            # logger.info('average_train_Loss: %.4f', np.mean(train_loss))    
            pbar.set_description(f'{step} - train: {train_loss.item():.4f}\n')
            # writer.add_scalar('train/loss', loss.item(), step)
            pbar.update(1)   
            
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="detr")
    parser.add_argument('--log_dir', default='./log')
    parser.add_argument('--seed', default=1)
    parser.add_argument('--train_batch_size', default=1)
    parser.add_argument('--valid_batch_size', default=1)
    parser.add_argument('--model_type', default=50)
    parser.add_argument('--backbone_name', default='resnet50')
    parser.add_argument('--num_classes', default=1)
    parser.add_argument('--pretrained_model', default='/public_bme/data/meilzj/Tooth/output/panoramic_xray/detects/model-52500.pt')
    parser.add_argument('--lr', default=1e-4)
    parser.add_argument('--scheduler_steps', default=10000)
    parser.add_argument('--gamma', default=0.1)
    parser.add_argument('--total_steps', default=20000)
    parser.add_argument('--valid_steps', default=1000)
    parser.add_argument('--save_freq', default=20)
    parser.add_argument('--save_dir', default='/public_bme/data/xiongjl/detr/save')
    parser.add_argument('--point_weight', default=4)
    parser.add_argument('--model', default='normal')
    parser.add_argument('--log_name', default='./log/training_50')
    parser.add_argument('--best_model_name', default='best_detrmodel50_v5_2nodes')
    parser.add_argument('--save_model_name', default='detrmodel50_v5_2nodes')
    args = parser.parse_args()
    
    train(args)
                
        
    
