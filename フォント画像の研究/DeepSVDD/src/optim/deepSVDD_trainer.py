from base.base_trainer import BaseTrainer
from base.base_dataset import BaseADDataset
from base.base_net import BaseNet
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import roc_auc_score

import logging
import time
import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import pickle

import os
import sys
# # プロジェクトのルートディレクトリを取得
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
# # プロジェクトのルートディレクトリをPythonのパスに追加
sys.path.append(root_dir)
from Transformer.src.dataloader import Dataloader_TF
from Transformer.src.Transformer import Transformer

from DeepSVDD.src.result.image.create_result_images import plot_LossCurve, plot_Features_train, plot_Features_test, create_fontlist_txt_MSE, plot_before_train_cls_tokens, plot_before_test_cls_tokens, plot_Features_test_and_train
from DeepSVDD.src.result.number.calculate_trainfeature_border import calculate_trainfeature_border

from torchvision import transforms
from PIL import Image
# def pad_to_square(image, fill_color=(255, 255, 255)):
#     # 画像のサイズを取得
#     img_width, img_height = image.size
#     # 新しいサイズを決定
#     new_size = max(img_width, img_height)
#     # 新しい画像を作成
#     new_image = Image.new('RGB', (new_size, new_size), fill_color)
#     # 元の画像を新しい画像の中央にペースト
#     new_image.paste(image, (int((new_size - img_width) / 2), int((new_size - img_height) / 2)))

#     # new_image.save('/home/miyatamoe/ドキュメント/研究（支部大会2024）/SIBU_font_image/padimg_test/pad.png')
#     return new_image

# def transform_image(image_path):
#     # 画像を開く
#     image = Image.open(image_path).convert('RGBA')
#     # image.save('/home/miyatamoe/ドキュメント/研究（支部大会2024）/SIBU_font_image/padimg_test/img.png')
#     background = Image.new('RGB', image.size, (255, 255, 255))
#     # 元の画像を背景画像に合成
#     background.paste(image, (0, 0), image)
#     # background.save('/home/miyatamoe/ドキュメント/研究（支部大会2024）/SIBU_font_image/padimg_test/img_neo.png')

#     # 画像を正方形にパディング
#     image = pad_to_square(background)
#     # 画像をテンソルに変換
#     transform = transforms.Compose([
#         transforms.Resize((64, 64)), # 画像サイズを統一
#         transforms.ToTensor()  # テンソルに変換
#     ])
#     image_tensor = transform(image)
#     return image_tensor

dataset_name = 'crello'

option = 'nofont1kind_arranged'
tfepoch = 10000 #transformerのepoch
mlpepoch = 90

class DeepSVDDTrainer(BaseTrainer):

    def __init__(self, objective, R, c, nu: float, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 150,
                 lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
                 n_jobs_dataloader: int = 0):
        super().__init__(optimizer_name, lr, n_epochs, lr_milestones, batch_size, weight_decay, device,
                         n_jobs_dataloader)

        assert objective in ('one-class', 'soft-boundary'), "Objective must be either 'one-class' or 'soft-boundary'."
        self.objective = objective

        # Deep SVDD parameters
        self.R = torch.tensor(R, device=self.device)  # radius R initialized with 0 by default.
        self.c = torch.tensor(c, device=self.device) if c is not None else None
        self.nu = nu

        # Optimization parameters
        self.warm_up_n_epochs = 10  # number of training epochs for soft-boundary Deep SVDD before radius R gets updated

        # Results
        self.train_time = None
        self.test_auc = None
        self.test_time = None
        self.test_scores = None

    def train(self, dataset: BaseADDataset, net: BaseNet, deep_SVDD, cfg):
        # transformerの準備
        autoen_ep = 400
        # num_epochs = 200
        # 保存したファイルから辞書を読み込む
        clstoken_dict_load_path = f'/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォント画像の研究/Transformer/result/A-Zstack/cls_tokens/AEepoch{autoen_ep}/train/clstoken_with_ids_tfepoch{tfepoch}_nofont1kind_arranged.pth'
        clstoken_dict_loaded = torch.load(clstoken_dict_load_path)

        logger = logging.getLogger()

        # Set device for network
        net = net.to(self.device)

        # Get train data loader
        if (dataset_name == 'crello'):
            train_data_path = '/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォント画像の研究/dataset/csv/train/train_no_font1kind_arranged_sorted.csv'
            train_random_data_path = '/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォント画像の研究/dataset/csv/train/train_fonts_null.csv'
            train_loader = Dataloader_TF(train_data_path, train_random_data_path, batch_size=self.batch_size, is_training=True)
        else:
            train_loader, _ = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Set optimizer (Adam optimizer for now)
        optimizer = optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay,
                               amsgrad=self.optimizer_name == 'amsgrad')

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)

        # Initialize hypersphere center c (if c not loaded)
        if self.c is None:
            logger.info('Initializing center c...')
            print('Initializing center c...')
            self.c = self.init_center_c(train_loader, net)
            logger.info('Center c initialized.')
            print('Center c initialized.')

        # Training
        logger.info('Starting training...')
        print('Starting training...')
        start_time = time.time()
        net.train()

        before_train_cls_tokens = []
        loss_epochs = []
        for epoch in range(self.n_epochs):
            features_epoch = []

            scheduler.step()
            if epoch in self.lr_milestones:
                logger.info('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))
                print('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))

            loss_epoch = 0.0
            n_batches = 0
            epoch_start_time = time.time()

            if (dataset_name == 'crello'):
                for font_list, len_list, id_list, label_list in tqdm(train_loader.yield_batched_data()):
                    font_list = torch.tensor(font_list, dtype=torch.int64).to(self.device)
                    len_list = torch.tensor(len_list, dtype=torch.int64).to(self.device)
                    label_list = torch.tensor(label_list, dtype=torch.float32).to(self.device)

                    # font_listを画像に変換して配列に格納
                    image_font_list = []  # font_listの画像版を格納するlist
                    for id in id_list:
                        # image_path = f'/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォント画像の研究/dataset/images/original_crello/ID{font}.png'
                        # image_tensor = transform_image(image_path)
                        # image_font_list.append(image_tensor.tolist())
                        image_font_list.append(clstoken_dict_loaded[id].tolist())
                    
                    cls_tokens = torch.tensor(image_font_list).to(self.device)

                    # with torch.no_grad():

                        # モデルに入力
                        # cls_tokens = image_font_list
                    if (epoch == 0):
                        before_train_cls_tokens.extend(cls_tokens.tolist())

                        # inputs = cls_tokens.to(self.device)
                    inputs = cls_tokens

                    # Zero the network parameter gradients
                    optimizer.zero_grad()

                    # Update network parameters via backpropagation: forward + backward + optimize
                    outputs = net(inputs)
                    # 特徴ベクトルを収集
                    for feature in outputs.tolist():
                        features_epoch.append(feature)

                    dist = torch.sum((outputs - self.c) ** 2, dim=1)
                    if self.objective == 'soft-boundary':
                        scores = dist - self.R ** 2
                        loss = self.R ** 2 + (1 / self.nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
                    else:
                        loss = torch.mean(dist)
                    loss.backward()
                    optimizer.step()

                    # Update hypersphere radius R on mini-batch distances
                    if (self.objective == 'soft-boundary') and (epoch >= self.warm_up_n_epochs):
                        self.R.data = torch.tensor(get_radius(dist, self.nu), device=self.device)

                    loss_epoch += loss.item()
                    n_batches += 1
        
            else:
                for data in train_loader:
                    inputs, _, _ = data
                    inputs = inputs.to(self.device)

                    # Zero the network parameter gradients
                    optimizer.zero_grad()

                    # Update network parameters via backpropagation: forward + backward + optimize
                    outputs = net(inputs)
                    dist = torch.sum((outputs - self.c) ** 2, dim=1)
                    if self.objective == 'soft-boundary':
                        scores = dist - self.R ** 2
                        loss = self.R ** 2 + (1 / self.nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
                    else:
                        loss = torch.mean(dist)
                    loss.backward()
                    optimizer.step()

                    # Update hypersphere radius R on mini-batch distances
                    if (self.objective == 'soft-boundary') and (epoch >= self.warm_up_n_epochs):
                        self.R.data = torch.tensor(get_radius(dist, self.nu), device=self.device)

                    loss_epoch += loss.item()
                    n_batches += 1

            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            logger.info('  Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}'
                        .format(epoch + 1, self.n_epochs, epoch_train_time, loss_epoch / n_batches))
            print('  Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}'
                        .format(epoch + 1, self.n_epochs, epoch_train_time, loss_epoch / n_batches))
            
            loss_epochs.append(loss_epoch / n_batches)


            if (dataset_name == 'crello'):
                # 学習前の特徴分布の表示
                if (epoch == 0):
                    plot_before_train_cls_tokens(before_train_cls_tokens, c=self.c)
                # 1epochごとに損失曲線と特徴分布の保存
                plot_LossCurve(epoch, loss_epochs)
                plot_Features_train(epoch, features_epoch, c=self.c)

                # 例えば、train関数内のエポックごとの保存
                train_features_save_path = f'/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォント画像の研究/DeepSVDD/result/features/train/{option}/tfepoch{tfepoch}/train_features_epoch{epoch+1}.pkl'
                with open(train_features_save_path, 'wb') as f:
                    pickle.dump(features_epoch, f)


                # # 1epochごとにネットワークの保存
                # net_path = f'/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォント画像の研究/DeepSVDD/result/models/A-Zstack/pth/net_MLP_epoch{epoch+1}.pth'
                # torch.save(net.state_dict(), net_path)

                # Save model
                deep_SVDD.save_model(export_model=f'/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォント画像の研究/DeepSVDD/result/models/A-Zstack/original/{option}/tfepoch{tfepoch}' + f'/model_MLP_{epoch+1}.tar', c=self.c.cpu().data.numpy().tolist())


        self.train_time = time.time() - start_time
        logger.info('Training time: %.3f' % self.train_time)
        print('Training time: %.3f' % self.train_time)

        logger.info('Finished training.')
        print('Finished training.')

        return net


    def test(self, dataset: BaseADDataset, net: BaseNet):
        val_or_tes = 'validation'
        
        logger = logging.getLogger()

        # Set device for network
        net = net.to(self.device)

        # Get test data loader
        if (dataset_name == 'crello'):
            # test_data_path = '/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォント画像の研究/dataset/csv/test/test_no_font1kind_arranged_sorted.csv'
            # test_random_data_path = '/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォント画像の研究/dataset/csv/test/test_no_font1kind_random_arranged_sorted.csv'
            test_data_path = f'/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォント画像の研究/dataset/csv/{val_or_tes}/{val_or_tes}_no_font1kind_arranged_sorted.csv'
            test_random_data_path = f'/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォント画像の研究/dataset/csv/{val_or_tes}/{val_or_tes}_no_font1kind_random_arranged_sorted.csv'
            test_loader = Dataloader_TF(test_data_path, test_random_data_path, batch_size=self.batch_size, is_training=False)
        else:
            _, test_loader = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Testing
        logger.info('Starting testing...')
        print('Starting testing...')
        start_time = time.time()
        idx_label_score = []
        net.eval()
        with torch.no_grad():
            if (dataset_name == 'crello'):
                # transformerの準備
                autoen_ep = 400
                # num_epochs = 200
                # 保存したファイルから辞書を読み込む
                # clstoken_dict_load_path = f'/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォント画像の研究/Transformer/result/A-Zstack/cls_tokens/AEepoch{autoen_ep}/test/clstoken_with_ids_tfepoch{tfepoch}_nofont1kind_arranged.pth'
                clstoken_dict_load_path = f'/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォント画像の研究/Transformer/result/A-Zstack/cls_tokens/AEepoch{autoen_ep}/{val_or_tes}/clstoken_with_ids_tfepoch{tfepoch}_nofont1kind_arranged.pth'
                clstoken_dict_loaded = torch.load(clstoken_dict_load_path)

                before_test_cls_tokens = []
                test_features = []
                font_list_all = []
                len_list_all = []
                id_list_all = []
                label_list_all = []
                for font_list, len_list, id_list, label_list in tqdm(test_loader.yield_batched_data()):
                    font_list_all.extend(font_list)
                    len_list_all.extend(len_list)
                    id_list_all.extend(id_list)
                    label_list_all.extend(label_list)

                    font_list = torch.tensor(font_list, dtype=torch.int64).to(self.device)
                    len_list = torch.tensor(len_list, dtype=torch.int64).to(self.device)
                    label_list = torch.tensor(label_list, dtype=torch.int64).to(self.device)

                    # font_listを画像に変換して配列に格納
                    image_font_list = []  # font_listの画像版を格納するlist
                    for id in id_list:
                        # image_path = f'/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォント画像の研究/dataset/images/original_crello/ID{font}.png'
                        # image_tensor = transform_image(image_path)
                        # image_font_list.append(image_tensor.tolist())
                        image_font_list.append(clstoken_dict_loaded[id].tolist())
                    
                    cls_tokens = torch.tensor(image_font_list).to(self.device)

                    # モデルに入力
                    # cls_tokens = image_font_list
                    before_test_cls_tokens.extend(cls_tokens.tolist())

                    # inputs = cls_tokens.to(self.device)
                    inputs = cls_tokens

                    outputs = net(inputs)
                    # 特徴ベクトルを収集
                    for feature in outputs.tolist():
                        test_features.append(feature)
                    
                    dist = torch.sum((outputs - self.c) ** 2, dim=1)
                    if self.objective == 'soft-boundary':
                        scores = dist - self.R ** 2
                    else:
                        scores = dist

                    # Save triples of (idx, label, score) in a list
                    idx_label_score += list(zip(id_list,
                                                label_list.cpu().numpy().tolist(),
                                                scores.cpu().data.numpy().tolist()))
                    
            else:
                for data in test_loader:
                    inputs, labels, idx = data
                    inputs = inputs.to(self.device)
                    outputs = net(inputs)
                    dist = torch.sum((outputs - self.c) ** 2, dim=1)
                    if self.objective == 'soft-boundary':
                        scores = dist - self.R ** 2
                    else:
                        scores = dist

                    # Save triples of (idx, label, score) in a list
                    idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                                labels.cpu().data.numpy().tolist(),
                                                scores.cpu().data.numpy().tolist()))

        self.test_time = time.time() - start_time
        logger.info('Testing time: %.3f' % self.test_time)
        print('Testing time: %.3f' % self.test_time)

        self.test_scores = idx_label_score

        # Compute AUC
        _, labels, scores = zip(*idx_label_score)
        labels = np.array(labels)
        scores = np.array(scores)

        self.test_auc = roc_auc_score(labels, scores)
        logger.info('Test set AUC: {:.2f}%'.format(100. * self.test_auc))
        print('Test set AUC: {:.2f}%'.format(100. * self.test_auc))

        # if (dataset_name == 'crello'):
        #     # テキスト出力用font list作成
        #     idx = 0
        #     test_font_list_all = []
        #     for length in len_list_all:
        #         test_font_list_all.append(font_list_all[idx : idx + length])
        #         idx += length
        #     # test時の特徴分布表示
        #     plot_before_test_cls_tokens(before_test_cls_tokens, label_list_all, c=self.c)
        #     plot_Features_test(test_features, test_font_list_all, label_list_all, n_epochs=self.n_epochs, c=self.c)
        #     create_fontlist_txt_MSE(test_scores=self.test_scores, n_epochs=self.n_epochs)
        #     plot_Features_test_and_train(test_features, test_font_list_all, label_list_all, n_epochs=self.n_epochs, c=self.c)
        #     calculate_trainfeature_border(c=self.c, device=self.device)

        #     # # 例えば、train関数内のエポックごとの保存
        #     test_features_save_path = f'/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォント画像の研究/DeepSVDD/result/features/test/{option}/tfepoch{tfepoch}/test_features_epoch{mlpepoch}.pkl'
        #     with open(test_features_save_path, 'wb') as f:
        #         pickle.dump(test_features, f)

        logger.info('Finished testing.')
        print('Finished testing.')

    def init_center_c(self, train_loader, net: BaseNet, eps=0.1):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
        n_samples = 0
        if net.rep_dim is None:
            net.rep_dim = 512  # 例えば、出力層の次元数など
        c = torch.zeros(net.rep_dim, device=self.device)

        net.eval()
        with torch.no_grad():
            if (dataset_name == 'crello'):
                # transformerの準備
                autoen_ep = 400
                num_epochs = 200
                # 保存したファイルから辞書を読み込む
                clstoken_dict_load_path = f'/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォント画像の研究/Transformer/result/A-Zstack/cls_tokens/AEepoch{autoen_ep}/train/clstoken_with_ids_tfepoch{tfepoch}_nofont1kind_arranged.pth'
                clstoken_dict_loaded = torch.load(clstoken_dict_load_path)

                for font_list, len_list, id_list, label_list in tqdm(train_loader.yield_batched_data()):
                    font_list = torch.tensor(font_list, dtype=torch.int64).to(self.device)
                    len_list = torch.tensor(len_list, dtype=torch.int64).to(self.device)
                    label_list = torch.tensor(label_list, dtype=torch.float32).to(self.device)

                    # font_listを画像に変換して配列に格納
                    image_font_list = []  # font_listの画像版を格納するlist
                    for id in id_list:
                        # image_path = f'/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォント画像の研究/dataset/images/original_crello/ID{font}.png'
                        # image_tensor = transform_image(image_path)
                        image_font_list.append(clstoken_dict_loaded[id].tolist())
                    
                    cls_tokens = torch.tensor(image_font_list).to(self.device)

                    # モデルに入力
                    # cls_tokens = image_font_list
                    # inputs = cls_tokens.to(self.device)
                    inputs = cls_tokens

                    outputs = net(inputs)
                    n_samples += outputs.shape[0]
                    c += torch.sum(outputs, dim=0)

            else:
                for data in train_loader:
                    # get the inputs of the batch
                    inputs, _, _ = data
                    inputs = inputs.to(self.device)
                    outputs = net(inputs)
                    n_samples += outputs.shape[0]
                    c += torch.sum(outputs, dim=0)

        c /= n_samples

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        return c


def get_radius(dist: torch.Tensor, nu: float):
    """Optimally solve for radius R via the (1-nu)-quantile of distances."""
    return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)
