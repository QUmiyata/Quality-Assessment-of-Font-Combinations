import torch
import sys
import os
import random
import numpy as np
import torch
import glob
import re

import FontAutoencoder
import pretrain_utils
import dataset

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../lib'))
sys.path.append(parent_dir)
from createImage import createInputOutput, visualize_pca_images, plot_loss
# import lib.utils as utils
from torchvision import utils


def fix_seed(seed):
    # Pythonのランダムシード設定
    random.seed(seed)
    # NumPyのランダムシード設定
    np.random.seed(seed)
    # PyTorchのランダムシード設定（CPU）
    torch.manual_seed(seed)
    # CUDAを使用する場合の設定（GPU）
    torch.cuda.manual_seed_all(seed)
    # CUDNNの設定（再現性確保）
    torch.backends.cudnn.deterministic = True  # 決定論的な演算を強制
    torch.backends.cudnn.benchmark = False  # 最適化を無効にして再現性を確保

def natural_sort_key(s):
    # 数字を抽出して、それを整数として扱う
    return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', s)]


# parameter from config
MAX_EPOCH = 50000
EARLY_STOPPING_PATIENCE = 50
LEARNING_RATE = 1e-4
BATCH_SIZE = 16
DECODER_TYPE = 'deconv'
RANDOM_SEED = 1

ae_epoch = 15000

# fix random numbers, set cudnn option
fix_seed(RANDOM_SEED)

# initialize the model and criterion, optimizer, earlystopping
device = torch.device('cuda:0')
# font_autoencoder = FontAutoencoder.Autoencoder().to(device)
font_autoencoder = FontAutoencoder.Autoencoder()
font_autoencoder.load_state_dict(torch.load(f'/home/miyatamoe/ドキュメント/研究/久保田さん/result/models/model_AE_epoch{ae_epoch}.pth'))
font_autoencoder = font_autoencoder.to(device)
criterion = torch.nn.L1Loss().to(device)
optimizer = torch.optim.Adam(font_autoencoder.parameters(), lr=LEARNING_RATE)
earlystopping = pretrain_utils.EarlyStopping(patience=EARLY_STOPPING_PATIENCE, delta=0)

# set up the data loader
img_paths_test = sorted(glob.glob('/home/miyatamoe/ドキュメント/研究/久保田さん/npy/*.npy', recursive=True), key=natural_sort_key)
test_set = dataset.ImageDataset(img_paths_test)
testloader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers = os.cpu_count(), pin_memory=True)


# testing
test_loss, all_latent_epoch, all_inputs_epoch = pretrain_utils.val(testloader, font_autoencoder, criterion, device)
print(f'[test]  {test_loss:7.4f}')

visualize_pca_images(all_latent_epoch, all_inputs_epoch, epoch=f'{ae_epoch}_test')

# # 例: 261個のlatentベクトル、それぞれにIDを付ける
# latent_dict = {}
# for ID in range(1, 262):
#     latent_dict[ID] = all_latent_epoch[ID-1]  # IDとlatentを辞書に格納

# # 保存するファイルパス
# save_path = f'/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォント画像の研究/AutoEncoder/久保田さん/result/latent/latent_with_ids_aeepoch{ae_epoch}.pth'
# # 辞書を保存
# torch.save(latent_dict, save_path)
# print(f'Latent tensors with IDs saved to {save_path}')