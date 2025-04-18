import torch
from skimage import io
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms
import re
import csv
from PIL import Image
import glob
import numpy as np

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]


def make_list(csv_list, img_folder_list):
    '''
        input:
            csv_list:['./Dataset/ICDAR 2013 word recognition/sorted_data/train_sorted.csv', './Dataset/ICDAR 2015 word recognition/sorted_data/train_sorted.csv', 
                        './Dataset/COCO-Text/sorted_data/train_words_sorted.csv']
            img_folder_list:['./Dataset/ICDAR 2013 word recognition/sorted_data/train_sorted', './Dataset/ICDAR 2015 word recognition/sorted_data/train_sorted', 
                        './Dataset/COCO-Text/sorted_data/train_words_sorted']

        output: ['./Dataset/ICDAR 2013 word recognition/sorted_data/train_sorted/word_1.png', './Dataset/ICDAR 2013 word recognition/sorted_data/train_sorted/word_2.png', 
                    './Dataset/ICDAR 2015 word recognition/sorted_data/train_sorted/word_1.png', './Dataset/ICDAR 2015 word recognition/sorted_data/train_sorted/word_2.png', 
                    './Dataset/COCO-Text/sorted_data/train_words_sorted/109_3352.png', './Dataset/COCO-Text/sorted_data/train_words_sorted/138_14739.png']
    '''

    img_list = []

    for i in range(len(csv_list)):
        csv_file = csv_list[i]
        with open(csv_file) as f:
            reader = csv.reader(f)
            l = [row for row in reader]
            img_text = [list(x) for x in zip(*l)]
            img_names = img_text[0][1:]
        
            if 'MJSynth' in csv_file:
                for img_name in img_names:
                    img_path = f'{img_folder_list[i]}{img_name[1:]}'
                    img_list.append(img_path)
            
            else:
                for img_name in img_names:
                    img_path = f'{img_folder_list[i]}/{img_name}'
                    img_list.append(img_path)

    
    return img_list


def resize_w_pad(img, target_w, target_h):
    '''
    Resize PIL image while maintaining aspect ratio
    '''
    # get new size
    target_ratio = target_h / target_w
    img_ratio = img.size(1) / img.size(2) 
    if target_ratio > img_ratio:
        # fixed with width
        new_w = target_w
        new_h = round(new_w * img_ratio)
    else:
        # fixed with height
        new_h = target_h
        new_w = round(new_h / img_ratio)
    # resize to new size
    tr = transforms.Resize((new_h, new_w))
    img = tr(img)
    
    # padding to target size
    horizontal_pad = (target_w - new_w) / 2
    vertical_pad = (target_h - new_h) / 2
    left = horizontal_pad if horizontal_pad % 1 == 0 else horizontal_pad + 0.5
    right = horizontal_pad if horizontal_pad % 1 == 0 else horizontal_pad - 0.5
    top = vertical_pad if vertical_pad % 1 == 0 else vertical_pad + 0.5
    bottom = vertical_pad if vertical_pad % 1 == 0 else vertical_pad - 0.5

    padding = (int(left), int(top), int(right), int(bottom))
    img = transforms.Pad(padding)(img)

    return img


def pad_to_square(image, fill_color=(255, 255, 255)):
    # 画像のサイズを取得
    img_width, img_height = image.size
    # 新しいサイズを決定
    new_size = max(img_width, img_height)
    # 新しい画像を作成
    new_image = Image.new('RGB', (new_size, new_size), fill_color)
    # 元の画像を新しい画像の中央にペースト
    new_image.paste(image, (int((new_size - img_width) / 2), int((new_size - img_height) / 2)))

    # new_image.save('/home/miyatamoe/ドキュメント/研究（支部大会2024）/SIBU_font_image/padimg_test/pad.png')
    return new_image


class Auto_Encoder_Dataset(Dataset):

    def __init__(self, img_list, target_w_size=64, target_h_size=64, transform=None):
        self.img_list = img_list
        self.target_w_size = target_w_size
        self.target_h_size = target_h_size
        self.transform = transform
    
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()


        image = Image.open(self.img_list[idx]).convert('RGBA')
        background = Image.new('RGB', image.size, (255, 255, 255))
        # 元の画像を背景画像に合成
        background.paste(image, (0, 0), image)

        # 画像を正方形にパディング
        image = pad_to_square(background)
        # 画像をテンソルに変換
        transform = transforms.Compose([
            transforms.Resize((self.target_w_size, self.target_h_size)), # 画像サイズを統一
            transforms.ToTensor()  # テンソルに変換
        ])
        input_image = transform(image)
        label_image = transform(image)

        return input_image, label_image
