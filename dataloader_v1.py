import torch
from torch.utils.data import Dataset, DataLoader
from IPython import embed
import argparse
import pandas as pd
import os
import torchio as tio
import random
import numpy as np
import logging



from PIL import Image
def draw_boxes_on_nii(image, boxes, name):
    # 将nii图像转换为numpy数组
    # print(image.shape)
    if isinstance(image, torch.Tensor):
        nii_data = np.array(image)
    if image.ndim == 3:
        nii_data = image 
    elif image.ndim == 4:
        nii_data = image.squeeze(0)
    elif image.ndim == 5:
        nii_data = image.squeeze(0).squeeze(0)
    else:
        print('DIM ERROR: image.ndim  != 3 or 4 or 5 ')

    # embed()
    # print(boxes)
    boxes = boxes[0]['boxes'] * 128.
    # print(f'boxes is {boxes}')
    # print(f'name is {name}')
    # print(f'nii data shape is {nii_data.shape}')
    new_size = (128, 128)
    # if isinstance(boxes[0], list):
    #     new_size = (300, 300) #说明这个box是整张数据的box
    # else:
    #     if boxes != [[]]:
    #         boxes = boxes[0] #说明这个box是一个patch的box
    #         new_size = (128, 128)

    # 创建一个空白的四维数组
    color_image = np.zeros((*nii_data.shape, 3))
    # 将灰度图像复制到新数组的每个颜色通道中
    color_image[..., 0] = nii_data
    color_image[..., 1] = nii_data
    color_image[..., 2] = nii_data
    # if color_image.shape

    # 遍历 ground_truth_boxes中的每个框

    for box in boxes:
        if len(box) == 0:
            return print('There are no bbox in the image patch')
        else:

            x, y, z, w, h, d = box

            x1 = int(x - w / 2)
            x2 = int(x + w / 2)
            y1 = int(y - h / 2)
            y2 = int(y + h / 2)
            z1 = int(z - d / 2)
            z2 = int(z + d / 2)
            # embed()
            # 在nii数据上绘制框

            color_image[x1:x2+1, y1:y2+1, z1, 0] = 1.0
            color_image[x1:x2+1, y1:y2+1, z2, 0] = 1.0
            color_image[x1:x2+1, y1, z1:z2+1, 0] = 1.0
            color_image[x1:x2+1, y2, z1:z2+1, 0] = 1.0
            color_image[x1, y1:y2+1, z1:z2+1, 0] = 1.0
            color_image[x2, y1:y2+1, z1:z2+1, 0] = 1.0

    slice_gt_array = []
    for box in boxes:
        # 获取框的坐标
        c_x, c_y, c_z = map(int, box[0:3])
        # embed()
        slice_x = color_image[c_x, :, :, :]
        slice_y = color_image[:, c_y, :, :]
        slice_z = color_image[:, :, c_z, :]

        slice_gt_array.append(slice_x)
        slice_gt_array.append(slice_y)
        slice_gt_array.append(slice_z)
        # embed()
    # 将数组转换为图像

    images = [Image.fromarray((arr * 255 ).astype(np.uint8)) for arr in slice_gt_array] #!

    # 将图片缩放到相同的大小
    # new_size = (128, 128)
    images = [image.resize(new_size) for image in images]

    # 计算新图片的大小
    width, height = new_size
    gap = 10
    number = len(images)
    merged_width = (width + gap) * number - gap

    # 创建一个新的空白图片，用于存放合并后的图片
    # embed()
    merged_image = Image.new('RGB', (int(merged_width), int(height)), (255, 255, 255))

    # 将六张图片合并到新图片中
    x, y = 0, 0
    for i in range(len(images)):
        merged_image.paste(images[i], (x, y))
        x += (width + gap)
        # 保存合并后的图片
    merged_image.save(os.path.join('/public_bme/data/xiongjl/detr/result_image', f"{name}.png"))


def creat_logging(log_name='log/dataloader.log'):
# 创建一个logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 创建一个handler，用于将日志输出到控制台
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # 创建一个handler，用于将日志写入到文件中
    fh = logging.FileHandler(log_name)
    fh.setLevel(logging.INFO)

    # 定义handler的输出格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    # 将handler添加到logger中
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


def select(image_name):
    image_list =\
       ['1.3.6.1.4.1.14519.5.2.1.6279.6001.102681962408431413578140925249',\
        '1.3.6.1.4.1.14519.5.2.1.6279.6001.119304665257760307862874140576',\
        '1.3.6.1.4.1.14519.5.2.1.6279.6001.121993590721161347818774929286',\
        '1.3.6.1.4.1.14519.5.2.1.6279.6001.122621219961396951727742490470',\
        '1.3.6.1.4.1.14519.5.2.1.6279.6001.132817748896065918417924920957',\
        '1.3.6.1.4.1.14519.5.2.1.6279.6001.145759169833745025756371695397',\
        '1.3.6.1.4.1.14519.5.2.1.6279.6001.146987333806092287055399155268',\
        '1.3.6.1.4.1.14519.5.2.1.6279.6001.178680586845223339579041794709',\
        '1.3.6.1.4.1.14519.5.2.1.6279.6001.279300249795483097365868125932',\
        '1.3.6.1.4.1.14519.5.2.1.6279.6001.313334055029671473836954456733',\
        '1.3.6.1.4.1.14519.5.2.1.6279.6001.313835996725364342034830119490',\
        '1.3.6.1.4.1.14519.5.2.1.6279.6001.613212850444255764524630781782',\
        '1.3.6.1.4.1.14519.5.2.1.6279.6001.624425075947752229712087113746',\
        '1.3.6.1.4.1.14519.5.2.1.6279.6001.850739282072340578344345230132',]
    if image_name in image_list:
        image_name = '1.3.6.1.4.1.14519.5.2.1.6279.6001.107351566259572521472765997306'
        return image_name
    else:
        return image_name


def crop_padding(image, start_point, size):
    # 计算裁剪区域的坐标范围
    x_min = start_point[0] - size[0] // 2
    x_max = start_point[0] + size[0] // 2
    y_min = start_point[1] - size[1] // 2
    y_max = start_point[1] + size[1] // 2
    z_min = start_point[2] - size[2] // 2
    z_max = start_point[2] + size[2] // 2
    
    # 计算需要填充的大小
    pad_x_min = max(0, -x_min)
    pad_x_max = max(0, x_max - image.shape[0])
    pad_y_min = max(0, -y_min)
    pad_y_max = max(0, y_max - image.shape[1])
    pad_z_min = max(0, -z_min)
    pad_z_max = max(0, z_max - image.shape[2])
    
    # 对图像进行填充
    padded_image = np.pad(image, ((pad_x_min, pad_x_max), (pad_y_min, pad_y_max), (pad_z_min, pad_z_max)), mode='constant', constant_values=0)
    
    # 裁剪图像
    cropped_image = padded_image[x_min+pad_x_min:x_max+pad_x_min, y_min+pad_y_min:y_max+pad_y_min, z_min+pad_z_min:z_max+pad_z_min]
    
    return cropped_image


def get_filenames(image_path):
    filenames = []
    for filename in os.listdir(image_path):
        filenames.append(filename.split('_')[0])
    return filenames


def read_csv(csv_path):
    data = pd.read_csv(csv_path)
    result_dict = {}
    for _, row in data.iterrows():
        seriesuid = row['seriesuid']
        x = row['x']
        y = row['y']
        z = row['z']
        w = row['w']
        h = row['h']
        d = row['d']
        if seriesuid not in result_dict:
            result_dict[seriesuid] = []
        result_dict[seriesuid].append([x, y, z, w, h, d])
    return result_dict


def process_boxes(boxes, coord, crop_length):
    result = []
    for box in boxes:
        x, y, z, w, h, d = box
        x -= coord[0]
        y -= coord[1]
        z -= coord[2]
        if (x - w/2 < 0 or y - h/2 < 0 or z - d/2 < 0) or (x + w/2 >= 128 or y + h/2 >= 128 or z + d/2 >= 128):
            continue
        result.append([x / np.float32(crop_length), y / np.float32(crop_length), z / np.float32(crop_length), w / np.float32(crop_length), h / np.float32(crop_length), d / np.float32(crop_length)])
    return result


def crop_image(nii_rootpath, image_name, labels, crop_length):
    # 读取图像
    image = tio.ScalarImage(f'{nii_rootpath}{image_name}_croplung.nii.gz')
    logger = creat_logging()
    # print(f"labels is {labels}")
    # draw_boxes_on_nii(image.data, labels, f'{image_name}_whole')

    # 生成一个随机数
    p = random.random()
    image_shape = image.shape[1:]
    idx = random.randint(0, len(labels) - 1)
    label = labels[idx]
    random.seed(0)

    if p < 0.8:
        # 有选择的裁剪
        x, y, z, w, h, d = label
        x_min = int(x - w / 2)
        y_min = int(y - h / 2)
        z_min = int(z - d / 2)
        # logger.info(f'x_crop between {(max(0, x_min - crop_length), min(x_min, image_shape[0] - crop_length))},\n\
        #               y_crop between {(max(0, y_min - crop_length), min(y_min, image_shape[1] - crop_length))},\n\
        #               z_crop between {(max(0, z_min - crop_length), min(z_min, image_shape[2] - crop_length))},\n\
        #               label is {label}, x_min is {x_min}, y_min is {y_min}, z_min is {z_min}, image shape is {image.shape}')
        # 考虑到万一边截止的范围比开始的要小的话，就去强行变化范围
        x_sta = max(0, x_min - crop_length)
        x_stop = min(x_min, image_shape[0] - crop_length)
        y_sta = max(0, y_min - crop_length)
        y_stop = min(y_min, image_shape[1] - crop_length)
        z_sta = max(0, z_min - crop_length)
        z_stop = min(z_min, image_shape[2] - crop_length)
        if x_sta >= x_stop:
            x_sta = x_stop - 10
        if y_sta >= y_stop:
            y_sta = y_stop - 10
        if z_sta >= z_stop:
            z_sta = z_stop - 10
        x_crop = random.randint(x_sta, x_stop)
        y_crop = random.randint(y_sta, y_stop)
        z_crop = random.randint(z_sta, z_stop)

    else:
        if image_shape[0] - crop_length <= 0:
            x_crop = 0
        else:
            x_crop = random.randint(0, image_shape[0] - crop_length)
        if image_shape[1] - crop_length <= 0:
            y_crop = 0
        else:
            y_crop = random.randint(0, image_shape[1] - crop_length)
        if image_shape[2] - crop_length <= 0:
            z_crop = 0
        else:
            z_crop = random.randint(0, image_shape[2] - crop_length)
        
    # 考虑到万一整个的image最短边小于被crop的长度的话，就去padding
    if (image_shape[0] - crop_length) <= 0 or x_crop < 0:
        x_crop = 0
    elif (image_shape[1] - crop_length) <= 0 or y_crop < 0:
        y_crop = 0
    elif (image_shape[2] - crop_length) <= 0 or z_crop < 0:
        z_crop = 0
    
    # 确定这个被crop图像的start point
    start_point = (x_crop, y_crop, z_crop)

    if (image_shape[0] - crop_length) <= 0 or (image_shape[1] - crop_length) <= 0 or (image_shape[2] - crop_length) <= 0:
        image_crop = crop_padding(image.data[0, :, :, :], start_point, size=(crop_length, crop_length, crop_length))
    else:
        image_crop = image.data[0, x_crop : x_crop + crop_length,\
                                   y_crop : y_crop + crop_length,\
                                   z_crop : z_crop + crop_length,]
    # embed()
    labels_ = process_boxes(labels, (x_crop, y_crop, z_crop), crop_length) 
    # logger.info(f'name is {image_name},\n start_point is {start_point}, origin label is {labels}, after is {labels_}')
    return image_crop, labels_



class DETR3DDataset(Dataset):

    def __init__(self, args, mode='train'):
        self.label_path = args.label_path
        self.image_path = args.image_path
        self.crop_length = args.crop_length
        self.mode = mode
        self.labels_dict = read_csv(self.label_path)
        self.setup()
       
    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        image_name = self.names[idx]
        image_name = select(image_name)
        labels = self.labels_dict[image_name]
        image_crop, bboxes = crop_image(self.image_path, image_name, labels, crop_length=self.crop_length)
        # embed()

        dct = {}
        dct['image'] = image_crop
        dct['label'] = bboxes
        dct['image_name'] = image_name
        return dct
    
    def setup(self):
        self.names = []
        names = get_filenames(self.image_path)
        random.seed(0)
        train_names = names[ : int(len(names) * 0.7)]
        valid_names = names[int(len(names) * 0.7) : int(len(names) * 0.8)]
        test_names = names[int(len(names) * 0.8) : ]
        print(f'valid names is {valid_names}')
        print(f'test names is {test_names}')

        if self.mode == 'valid':
            self.names = valid_names
        elif self.mode == 'test':
            self.names = test_names
        elif self.mode == 'train':
            np.random.shuffle(train_names) 
            self.names = train_names
        else:
            print('MODE ERROR: the mode should be "train" or "test" or "valid"!!!!!!!!')   


def detr_dataset_collate(batch):
    targets = []
    names = []
    images = []
    # print(f'the batch in fn is {batch}')
    for dct in batch:
        img, bbox, name = dct['image'], dct['label'], dct['image_name']
        # print(f'img, bbox, name is {img}, {bbox}, {name}')
        batch_dct = {}
        batch_classes = []
        batch_boxes = []
        if bbox == []:
            batch_dct['labels'] = torch.tensor([1])
            batch_dct['boxes'] = torch.tensor([[0, 0, 0, 0, 0, 0]])
            targets.append(batch_dct)
        else:
            for box in bbox:
                batch_boxes.append(box)
                batch_classes.append(0)
            batch_dct['labels'] = torch.tensor(batch_classes)
            batch_dct['boxes'] = torch.tensor(batch_boxes)
            targets.append(batch_dct)
        names.append(name)
        images.append(img)

    stacked_array = np.stack(images)
    result = np.expand_dims(stacked_array, axis=1)
    result = torch.from_numpy(result).type(torch.FloatTensor)

    dct = {}
    dct['image'] = result
    dct['label'] = targets
    dct['image_name'] = names
    return dct


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="detr3d dataloader")
    parser.add_argument('--label_path', default='/public_bme/data/xiongjl/detr/data/annotation_v1.csv')
    parser.add_argument('--image_path', default='/public_bme/data/xiongjl/det/nii_data_resample_seg_crop/')
    parser.add_argument('--crop_length', default=128)

    args = parser.parse_args()
    
    dataset = DETR3DDataset(args)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=detr_dataset_collate)
    # embed()
    for dct in dataloader:
        # pass
        image, bboxes, name = dct['image'], dct['label'], dct['image_name']
        draw_boxes_on_nii(image, bboxes, name)
    # train(args)




                
