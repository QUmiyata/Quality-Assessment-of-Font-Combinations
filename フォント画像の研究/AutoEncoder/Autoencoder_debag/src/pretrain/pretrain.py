import torch
import sys
import os
import random
import numpy as np
import torch
import glob

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


# parameter from config
MAX_EPOCH = 90
EARLY_STOPPING_PATIENCE = 50
LEARNING_RATE = 1e-4
BATCH_SIZE = 16
DECODER_TYPE = 'deconv'
RANDOM_SEED = 1

# fix random numbers, set cudnn option
fix_seed(RANDOM_SEED)

# initialize the model and criterion, optimizer, earlystopping
device = torch.device('cuda:0')
font_autoencoder = FontAutoencoder.Autoencoder().to(device)
criterion = torch.nn.L1Loss().to(device)
optimizer = torch.optim.Adam(font_autoencoder.parameters(), lr=LEARNING_RATE)
earlystopping = pretrain_utils.EarlyStopping(patience=EARLY_STOPPING_PATIENCE, delta=0)

# set up the data loader
img_paths_train = glob.glob('/home/miyatamoe/ドキュメント/研究/久保田さん/npy/*.npy', recursive=True)
train_set = dataset.ImageDataset(img_paths_train)
trainloader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers = os.cpu_count(), pin_memory=True)


loss_history = []
# train and save results
for epoch in range(1, MAX_EPOCH + 1):
    print('-'*130)
    print('Epoch {}/{}'.format(epoch, MAX_EPOCH))

    # training and validation
    train_loss, all_latent_epoch, all_inputs_epoch, all_outputs_epoch = pretrain_utils.train(trainloader, font_autoencoder, criterion, optimizer, device)
    print(f'[train]  {train_loss:7.4f}')
    loss_history.append(train_loss)

    if (epoch%10) == 0:
        checkpoint = f'/home/miyatamoe/ドキュメント/研究/久保田さん/result/models'
        os.makedirs(checkpoint, exist_ok=True)
        checkpoint_name = checkpoint + f'/model_AE_epoch{epoch}.pth'
        torch.save(font_autoencoder.state_dict(), checkpoint_name)

        createInputOutput(all_inputs_epoch, all_outputs_epoch, epoch)
        visualize_pca_images(all_latent_epoch, all_outputs_epoch, epoch)
        plot_loss(loss_history, epoch)
