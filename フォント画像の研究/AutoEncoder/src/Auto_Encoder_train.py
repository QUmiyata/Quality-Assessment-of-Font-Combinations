'''
    Auto-Encoderを学習させるためのプログラム
'''

import os
import torch
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import utils
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from tools.pytorchtools import EarlyStopping
import tools.Auto_Encoder_utils as tools
from Auto_Encoder import AutoEncoder
import argparse
import glob


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available())
print(device)

# np.random.seed(0)
# torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)


def fix_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)  # fix the initial value of the network weight
    torch.cuda.manual_seed(seed)  # for cuda
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True  # choose the determintic algorithm




def get_args():
    parser = argparse.ArgumentParser(description='sample',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--batch', type=int, default=64, help='Batch size')
    parser.add_argument('--epoch', type=int, default=100, help='Number of epoch')
    parser.add_argument('--seed', type=int, default=0, help='seed number')
    # parser.add_argument('--csv_list', nargs="*", type=str, default=['/home/miyatamoe/ドキュメント/研究（支部大会2024）/SIBU_font_image/datasets/csv/train/train_fonts_no_0_sorted.csv'], help='csv list')
    parser.add_argument('--img_folder', type=str, default='/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォント画像の研究/dataset/images/original_crello', help='image folder path')
    # parser.add_argument('--csv_list', nargs="*", type=str, default=['./Dataset/sample/ICDAR 2013 word recognition/sorted_data/train_sorted.csv', './Dataset/sample/ICDAR 2015 word recognition/sorted_data/train_sorted.csv', './Dataset/sample/COCO-Text/sorted_data/train_words_sorted.csv', './Dataset/sample/IIIT5K-Word_V3.0/IIIT5K/sorted_data/train_sorted.csv', './Dataset/sample/Total-Text/sorted_data/train_sorted.csv'], help='csv list')
    # parser.add_argument('--img_folder_list', nargs="*", type=str, default=['./Dataset/sample/ICDAR 2013 word recognition/sorted_data/train_sorted', './Dataset/sample/ICDAR 2015 word recognition/sorted_data/train_sorted', './Dataset/sample/COCO-Text/sorted_data/train_words_sorted', './Dataset/sample/IIIT5K-Word_V3.0/IIIT5K/sorted_data', './Dataset/sample/Total-Text/sorted_data/train_sorted'], help='image folder path')
    # parser.add_argument('--early_stopping', type=int, default=30, help='Early stopping epoch')

    return parser.parse_args()


# # validation
# def validate(net, dataloader, dataset, criterion, epoch, save_dir_input, save_dir_output):
#     net.eval()
#     with torch.no_grad():
#         total_loss = 0.0
#         i = 0

#         if ((epoch+1)%100) == 0:
#         # if ((epoch+1)%3) == 0:
#             new_save_dir_input = save_dir_input + '/epoch' + str(epoch+1)
#             new_save_dir_output = save_dir_output + '/epoch' + str(epoch+1)
#             os.makedirs(new_save_dir_input, exist_ok=True)
#             os.makedirs(new_save_dir_output, exist_ok=True)

#         for data in dataloader:
#             inputs, labels = data
#             inputs = inputs.to(device)
#             labels = labels.to(device)

#             outputs = net(inputs)
#             loss = criterion(outputs, labels)
#             total_loss+=loss.item()*inputs.shape[0]

#             if ((epoch+1)%100) == 0:
#             # if ((epoch+1)%3) == 0:
#                 img_name_input = new_save_dir_input +'/input_back_image' + str(i) + '.png'
#                 img_name_output = new_save_dir_output + '/output_image' + str(i) + '.png'
#                 utils.save_image(inputs, img_name_input, normalize=True)
#                 utils.save_image(outputs, img_name_output, normalize=True)
#                 i+=1 

#     avg_loss = total_loss / len(dataset)
#     return avg_loss


# training
def train(net, traindataloader, traindataset, criterion, optimizer, epochs, save_dir_input_train, save_dir_output_train, checkpoint_name):

    train_loss_history = []
    epoch_history = []

    i = 0

    for epoch in (range(epochs)):

        loss_item = 0
        print("Now epoch : %d/%d" %(epoch,epochs))

        if ((epoch+1)%100) == 0:
        # if ((epoch+1)%3) == 0:
            new_save_dir_input_train = save_dir_input_train + '/epoch' + str(epoch+1)
            new_save_dir_output_train = save_dir_output_train + '/epoch' + str(epoch+1)
            os.makedirs(new_save_dir_input_train, exist_ok=True)
            os.makedirs(new_save_dir_output_train, exist_ok=True)


        net.train()

        for data in tqdm(traindataloader, leave=False):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            loss_item+=loss.item()*inputs.shape[0]

            if ((epoch+1)%100) == 0:
            # if ((epoch+1)%3) == 0:
                img_name_input = new_save_dir_input_train +'/input_image' + str(i) + '.png'
                img_name_outputs = new_save_dir_output_train + '/output_in_image' + str(i) + '.png'

                utils.save_image(inputs, img_name_input, normalize=True)
                utils.save_image(outputs, img_name_outputs, normalize=True)

                i+=1

            # 100epochごとに学習モデルの保存
            if ((epoch+1)%100) == 0:
                checkpoint = f'./result/model/train_model/sample/Image_Encoder/Auto_Encoder/{save_name}/per_epoch'
                os.makedirs(checkpoint, exist_ok=True)
                checkpoint_name = checkpoint + f'/model_AE_epoch{epoch+1}.pth'
                torch.save(net.state_dict(), checkpoint_name)

        avg_loss_train = loss_item/len(traindataset)

        train_loss_history.append(avg_loss_train)
        epoch_history.append(epoch)

    print('Finished Training')

    torch.save(net.state_dict(), checkpoint_name)

    return train_loss_history, epoch_history, epochs



if __name__ == '__main__':
    args = get_args()

    fix_seed(args.seed)

    im_list = []
    for ID in range(1, 262):
        im_sublist = sorted(glob.glob(f'{args.img_folder}/ID{ID}/*.png', recursive=True))
        im_list.append(im_sublist)
    # target_size = args.img_size
    traindataset = tools.Auto_Encoder_Dataset(im_list, 64, 64)

    traindataloader = DataLoader(dataset=traindataset, batch_size = args.batch, shuffle=True, num_workers=0)

    net = AutoEncoder()
    net.to(device)

    save_name = f'batch{str(args.batch)}_lr0.0001'

    # save_dir_input_train = f'./train_val_output/Image_Encoder/Auto_Encoder/{save_name}/train/input'
    # save_dir_label_train = f'./train_val_output/Image_Encoder/Auto_Encoder/{save_name}/train/label'
    # save_dir_output_train = f'./train_val_output/Image_Encoder/Auto_Encoder/{save_name}/train/output'

    # save_dir_input_val = f'./train_val_output/Image_Encoder/Auto_Encoder/{save_name}/validation/input'
    # save_dir_label_val = f'./train_val_output/Image_Encoder/Auto_Encoder/{save_name}/validation/label'
    # save_dir_output_val = f'./train_val_output/Image_Encoder/Auto_Encoder/{save_name}/validation/output'

    save_dir_input_train = f'./result/images/train_output/Image_Encoder/Auto_Encoder/{save_name}/train/input'
    save_dir_label_train = f'./result/images/train_output/Image_Encoder/Auto_Encoder/{save_name}/train/label'
    save_dir_output_train = f'./result/images/train_output/Image_Encoder/Auto_Encoder/{save_name}/train/output'
    

    criterion = nn.L1Loss() 
    optimizer = optim.Adam(net.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    epochs = args.epoch

    # checkpoint = f'./train_model/Image_Encoder/Auto_Encoder/{save_name}'
    checkpoint = f'./result/model/train_model/sample/Image_Encoder/Auto_Encoder/{save_name}'
    os.makedirs(checkpoint, exist_ok=True)
    checkpoint_name = checkpoint + '/checkpoint_model.pth'
    # early_stopping = EarlyStopping(patience=args.early_stopping, verbose=True, path=checkpoint_name)

    train_loss_history, epoch_history, epochs = train(net, traindataloader, traindataset, criterion, optimizer, epochs, save_dir_input_train, save_dir_output_train, checkpoint_name)
    
    fig1 = plt.figure()
    plt.plot(epoch_history, train_loss_history, label='train')
    plt.legend()
    plt.grid()
    plt.xlabel('epoch')
    plt.title("loss")
    graph_path = f'./result/images/graph/Image_Encoder/Auto_Encoder/{save_name}/Auto_Encoder.png'
    os.makedirs(f'./result/images/graph/Image_Encoder/Auto_Encoder/{save_name}', exist_ok=True)
    fig1.savefig(graph_path)