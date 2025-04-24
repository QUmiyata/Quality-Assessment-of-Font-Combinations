import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional
import matplotlib.pyplot as plt
from dataloader import Dataloader_TF
from Transformer import Transformer
import random
from tqdm import tqdm
import numpy as np
import csv
from result.createPCA_fromClstokens import createPCA_fromClstokens
# from AutoEncoder.src.Auto_Encoder import AutoEncoder

import logging
# ログ設定
logging.basicConfig(
    # filename='/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォント画像の研究/Transformer/result/A-Zstack/log/train_original_log.log',  # 単一フォント除外なし
    filename='/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォント画像の研究/Transformer/result/A-Zstack/log/train_nofont1kind_arranged_log.log',  # 単一フォント除外あり・フォントIDの並び替えあり
    level=logging.INFO,  # ログレベルをINFOに設定
    format='%(asctime)s - %(levelname)s - %(message)s'  # ログメッセージのフォーマット
)

def fix_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)  # fix the initial value of the network weight
    torch.cuda.manual_seed(seed)  # for cuda
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True  # choose the determintic algorithm

def pad_data(font_list, max_len):     # font_listに含まれる各fontsリストの長さがmax_len未満の場合、不足する部分をゼロで埋める
        padded_fonts = []
        for fonts in font_list:
            padded_fonts.append(fonts + [0] * (max_len - len(fonts)))
        return padded_fonts


# fontIDをカウントして配列に格納する関数
def count_fontID_frequency(file_path, fontID_list):
    # Download data
    list = []
    with open(file_path) as f:
        reader = csv.reader(f)
        next(reader)  # ヘッダー行をスキップ
        for r in tqdm(reader, desc="[Download data]"):
            list.append(r[2:])
    # 1つのリストにする
    flat_list = [item for sublist in list for item in sublist]
    # 整数のリストに変換
    integer_list = [int(s) for s in flat_list]

    for i in integer_list:
        fontID_list[i] = fontID_list[i] + 1

    # total_sum = sum(fontID_list)
    return fontID_list


def get_indices_sorted_by_values_exclude_first(lst):
    if len(lst) <= 1:
        raise ValueError("リストには2つ以上の要素が必要です。")
    # 最初の要素を除いたリストを取得
    sublist = lst[1:]
    # (値, インデックス) のタプルのリストを作成
    indexed_list = list(enumerate(sublist, start=1))
    # 値で降順にソート
    sorted_list = sorted(indexed_list, key=lambda x: x[1], reverse=True)
    # ソート後のインデックスを抽出
    sorted_indices = [index for index, value in sorted_list]
    return [0]+sorted_indices


# シード固定
fix_seed(0)

# デバイスの設定
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# ハイパーパラメータの設定
# # ## 単一フォント除外なし・フォントIDの並び替えなし
# train_data_path = "/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォント画像の研究/dataset/csv/train/old/train_fonts_no_0_sorted.csv"
# train_random_data_path = "/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォント画像の研究/dataset/csv/train/old/train_fonts_random_no_0_range±3.csv"
# validation_data_path = "/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォント画像の研究/dataset/csv/validation/old/validation_fonts_no_0_sorted.csv"
# validation_random_data_path = "/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォント画像の研究/dataset/csv/validation/old/validation_fonts_random_no_0_range±3.csv"
# 単一フォント除外あり・フォントIDの並び替えあり
train_data_path = "/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォント画像の研究/dataset/csv/train/train_no_font1kind_arranged_sorted.csv"
train_random_data_path = "/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォント画像の研究/dataset/csv/train/train_no_font1kind_random_arranged_sorted.csv"
validation_data_path = "/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォント画像の研究/dataset/csv/validation/validation_no_font1kind_arranged_sorted.csv"
validation_random_data_path = "/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォント画像の研究/dataset/csv/validation/validation_no_font1kind_random_arranged_sorted.csv"
batch_size = 256
enc_layers = 2
max_seq_len = 44
d_model = 128
dim_feedforward = 256
num_epochs = 10000
lr = 1e-5 
warmup_steps = 4000

autoen_ep = 400

# データローダーの初期化
dataloader = Dataloader_TF(train_data_path, train_random_data_path, batch_size, is_training=True)
validation_dataloader = Dataloader_TF(validation_data_path, validation_random_data_path, batch_size, is_training=False)

# モデルの初期化
model = Transformer(num_classes=2, embed_dim=512, depth=enc_layers)
model = model.to(device) 

# 保存したファイルから辞書を読み込む
latent_dict_load_path = f'/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォント画像の研究/AutoEncoder/久保田さん/result/latent/latent_with_ids_aeepoch{autoen_ep}.pth'
latent_dict_loaded = torch.load(latent_dict_load_path)
# latent_id_5 = latent_dict_loaded[5]
# print(f"Latent for ID 5: {latent_id_5.shape}")


# IDの置き換え用リスト
train_no_font1kind_path = "/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォントIDの研究/dataset/csv/train/old/train_no_font1kind_sorted.csv"
train_fontID_frequency = [0] * 262
count_fontID_frequency(train_no_font1kind_path, train_fontID_frequency)
train_fontID_frequency_index = get_indices_sorted_by_values_exclude_first(train_fontID_frequency)

# 損失関数と最適化アルゴリズムの設定
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=lr)
# scheduler = TransformerLR(optimizer, d_model=d_model, warmup_steps=warmup_steps)

# 損失と認識率を保存するリスト
losses = []
accuracies = []
validation_losses = []
validation_accuracies = []

validation_avg_loss_min = 100 # validation lossの最小値を記録

for epoch in range(num_epochs):
    ########## train ###########
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    font_list_epoch = []
    len_list_epoch = []
    id_list_epoch = []
    label_list_epoch = []
    predicted_epoch = []
    cls_tokens_epoch = []

    for font_list, len_list, id_list, label_list in tqdm(dataloader.yield_batched_data()):
        # 成功例、失敗例出力用のlist
        font_list_epoch.extend(font_list)
        len_list_epoch.extend(len_list)
        id_list_epoch.extend(id_list)
        label_list_epoch.extend(label_list)

        font_list = torch.tensor(font_list, dtype=torch.int64).to(device) 
        len_list = torch.tensor(len_list, dtype=torch.int64).to(device)
        label_list = torch.tensor(label_list, dtype=torch.float32).to(device) #binary

        # font_listを画像に変換して配列に格納
        image_font_list = []  # font_listの画像版を格納するlist
        for font in font_list:
            # image_path = f'/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/dataset/images/original_crello/ID{font}.png'
            # image_tensor = transform_image(image_path)

            # image_font_list.append(latent_dict_loaded[font.item()].tolist())  # 単一フォント除外なし・フォントの並び替えなし

            # 単一フォント除外あり・フォントID並び替えあり
            original_ID = train_fontID_frequency_index[font.item()]
            image_font_list.append(latent_dict_loaded[original_ID].tolist())
        
        image_font_list = torch.tensor(image_font_list).to(device)

        # モデルに入力
        outputs, cls_tokens = model(image_font_list, len_list=len_list)
        cls_tokens_epoch.extend(cls_tokens.tolist())

        # 損失の計算と逆伝搬
        loss = criterion(outputs, label_list)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()
        total_loss += loss.item() * len(font_list)

        # 予測値の取得
        predicted = torch.round(outputs)
        predicted_epoch.extend(predicted.tolist())

        total += label_list.size(0)
        correct += (predicted == label_list).sum().item()

    # テキスト出力用font_list作成
    idx = 0
    font_list_for_predict = []
    for length in len_list_epoch:
        font_list_for_predict.append(font_list_epoch[idx : idx + length])
        idx += length
    font_list_for_predict = pad_data(font_list_for_predict, max_seq_len)
    font_list_for_predict = torch.tensor(font_list_for_predict, dtype=torch.int64).to(device) 

    # match_file_name = f'/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/Transformer/result/texts/Autoencoder_epoch{autoen_ep}/train/train_match_±3_epoch{num_epochs}.txt'
    # mismatch_file_name = f'/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/Transformer/result/texts/Autoencoder_epoch{autoen_ep}/train/train_mismatch_±3_epoch{num_epochs}.txt'

    # #正しく判定しているデータはtrain_matchへ、そうでないものはtrain_mismatchへ保存
    # if epoch == num_epochs-1:
    #     with open(match_file_name, 'w') as match_file, \
    #         open(mismatch_file_name, 'w') as mismatch_file:
    #         for i in range(len(label_list)):
    #             # フォントリストから要素が0でないものをフィルタリング
    #             filtered_font_list = [str(x.item()) for x in font_list_for_predict[i].flatten() if x.item() != 0]
    #             if filtered_font_list:  # フィルタリング後に要素があるか確認
    #                 font_list_str = ', '.join(filtered_font_list)
    #                 if predicted[i] == label_list[i]:
    #                     match_file.write(f"ID: {id_list[i]}, Predicted: {predicted[i].item()}, Actual: {label_list[i].item()}, Font List: [{font_list_str}]\n")
    #                 else:
    #                     mismatch_file.write(f"ID: {id_list[i]}, Predicted: {predicted[i].item()}, Actual: {label_list[i].item()}, Font List: [{font_list_str}]\n")

    avg_loss = total_loss / len(dataloader.idx_list)
    losses.append(avg_loss)

    accuracy = correct / total
    accuracies.append(accuracy)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    logging.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")


    ###### validation ########
    # バリデーションデータを用いた評価
    model.eval()
    validation_correct = 0
    validation_total = 0
    validation_loss = 0.0

    validation_font_list_epoch = []
    validation_len_list_epoch = []
    validation_id_list_epoch = []
    validation_label_list_epoch = []
    validation_predicted_epoch = []
    validation_cls_tokens_epoch = []


    with torch.no_grad():
        for font_list, len_list, id_list, label_list in validation_dataloader.yield_batched_data():
            # 成功例、失敗例出力用のlist
            validation_font_list_epoch.extend(font_list)
            validation_len_list_epoch.extend(len_list)
            validation_id_list_epoch.extend(id_list)
            validation_label_list_epoch.extend(label_list)

            font_list = torch.tensor(font_list, dtype=torch.int64).to(device) 
            len_list = torch.tensor(len_list, dtype=torch.int64).to(device)
            label_list = torch.tensor(label_list, dtype=torch.float32).to(device)   #binary

            # font_listを画像に変換して配列に格納
            image_font_list = []  # font_listの画像版を格納するlist
            for font in font_list:
                # image_path = f'/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/dataset/images/original_crello/ID{font}.png'
                # image_tensor = transform_image(image_path)

                # image_font_list.append(latent_dict_loaded[font.item()].tolist())  # 単一フォント除外なし

                # arranged_ID = train_fontID_frequency_index.index(font.item())

                # 単一フォント除外あり・フォントID並び替えあり
                original_ID = train_fontID_frequency_index[font.item()]
                image_font_list.append(latent_dict_loaded[original_ID].tolist())
            
            image_font_list = torch.tensor(image_font_list).to(device)

            # モデルに入力
            outputs, cls_tokens = model(image_font_list, len_list=len_list)
            validation_cls_tokens_epoch.extend(cls_tokens.tolist())

            # 損失の計算
            loss = criterion(outputs, label_list)
            validation_loss += loss.item() * len(font_list)

            # 予測値の取得
            predicted = torch.round(outputs)
            validation_predicted_epoch.extend(predicted)

            validation_total += label_list.size(0)
            validation_correct += (predicted == label_list).sum().item()


    # テキスト出力用font list作成
    idx = 0
    validation_font_list_for_predict = []
    for length in validation_len_list_epoch:
        validation_font_list_for_predict.append(validation_font_list_epoch[idx : idx + length])
        idx += length
    validation_font_list_for_predict = pad_data(validation_font_list_for_predict, max_seq_len)
    validation_font_list_for_predict = torch.tensor(validation_font_list_for_predict, dtype=torch.int64).to(device) 


            # #正しく判定しているデータはvalidation_matchへ、そうでないものはvalidation_mismatchへ保存
            # if epoch == num_epochs-1:
            #     with open('/home/miyatamoe/ドキュメント/研究（支部大会2024）/siku/実験後予測結果/支部大会原稿用/validation_match_no_font1kind_epoch100.txt', 'a') as validation_match_file, \
            #         open('/home/miyatamoe/ドキュメント/研究（支部大会2024）/siku/実験後予測結果/支部大会原稿用/validation_mismatch_no_font1kind_epoch100.txt', 'a') as validation_mismatch_file:
            #         for i in range(len(label_list)):
            #             # フォントリストから要素が0でないものをフィルタリング
            #             filtered_font_list = [str(x.item()) for x in font_list_for_predict[i].flatten() if x.item() != 0]
            #             if filtered_font_list:  # フィルタリング後に要素があるか確認
            #                 font_list_str = ', '.join(filtered_font_list)
            #                 if predicted[i] == label_list[i]:
            #                     validation_match_file.write(f"ID: {id_list[i]}, Predicted: {predicted[i].item()}, Actual: {label_list[i].item()}, Font List: [{font_list_str}]\n")
            #                 else:
            #                     validation_mismatch_file.write(f"ID: {id_list[i]}, Predicted: {predicted[i].item()}, Actual: {label_list[i].item()}, Font List: [{font_list_str}]\n")

            
    validation_avg_loss = validation_loss / len(validation_dataloader.idx_list)
    validation_losses.append(validation_avg_loss)
    validation_accuracy = validation_correct / validation_total
    validation_accuracies.append(validation_accuracy)
    print(f"Validation Loss: {validation_avg_loss:.4f}, Validation Accuracy: {validation_accuracy:.4f}")
    logging.info(f"Validation Loss: {validation_avg_loss:.4f}, Validation Accuracy: {validation_accuracy:.4f}")

    # 学習モデルの保存（validation lossが一番小さいとき）
    if validation_avg_loss <= validation_avg_loss_min:
        model_path = f'/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォント画像の研究/Transformer/result/A-Zstack/models/AEepoch{autoen_ep}/model_TF_ValLossMin_epoch{num_epochs}_nofont1kind_arranged.pth'
        torch.save(model.state_dict(), model_path)
        validation_avg_loss_min = validation_avg_loss

        createPCA_fromClstokens(predicted_epoch, label_list_epoch, cls_tokens_epoch, num_epochs, epoch, mode='train', isValLossMin=True)
        createPCA_fromClstokens(validation_predicted_epoch, validation_label_list_epoch, validation_cls_tokens_epoch, num_epochs, epoch, mode='validation', isValLossMin=True)

        # validation lossが一番小さいときの損失と認識率のグラフ表示
        epochs = range(1, (epoch+1) + 1)
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, losses, '-o', markersize=2, label='Train Loss', color='tab:blue')
        plt.plot(epochs, validation_losses, '-o', markersize=2, label='Validation Loss', color='tab:orange')
        plt.title('Loss per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(epochs, accuracies, '-o', markersize=2, label='Train Accuracy', color='tab:blue')
        plt.plot(epochs, validation_accuracies, '-o', markersize=2, label='Validation Accuracy', color='tab:orange')
        plt.title('Accuracy per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        result_image_path = f'/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォント画像の研究/Transformer/result/A-Zstack/images/AEepoch{autoen_ep}/TF_2class_TraAndVal_ValLossMin_epoch{num_epochs}_nofont1kind_arranged.png'
        plt.savefig(result_image_path)  # 画像として保存
        plt.close()

        #正しく判定しているデータはtrain_matchへ、そうでないものはtrain_mismatchへ保存
        train_match_file_name = f'/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォント画像の研究/Transformer/result/A-Zstack/texts/AEepoch{autoen_ep}/ValidationLossMin/train/train_ValMin_match_±3_epoch{num_epochs}_nofont1kind_arranged.txt'
        train_mismatch_file_name = f'/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォント画像の研究/Transformer/result/A-Zstack/texts/AEepoch{autoen_ep}/ValidationLossMin/train/train_ValMin_mismatch_±3_epoch{num_epochs}_nofont1kind_arranged.txt'

        with open(train_match_file_name, 'w') as train_match_file, \
            open(train_mismatch_file_name, 'w') as train_mismatch_file:
            for i in range(len(label_list_epoch)):
                # フォントリストから要素が0でないものをフィルタリング
                train_filtered_font_list = [str(x.item()) for x in font_list_for_predict[i].flatten() if x.item() != 0]
                if train_filtered_font_list:  # フィルタリング後に要素があるか確認
                    train_font_list_str = ', '.join(train_filtered_font_list)
                    if predicted_epoch[i] == label_list_epoch[i]:
                        train_match_file.write(f"ID: {id_list_epoch[i]}, Predicted: {predicted_epoch[i]}, Actual: {label_list_epoch[i]}, Font List: [{train_font_list_str}]\n")
                    else:
                        train_mismatch_file.write(f"ID: {id_list_epoch[i]}, Predicted: {predicted_epoch[i]}, Actual: {label_list_epoch[i]}, Font List: [{train_font_list_str}]\n")


        #正しく判定しているデータはvalidation_matchへ、そうでないものはvalidation_mismatchへ保存
        validation_match_file_name = f'/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォント画像の研究/Transformer/result/A-Zstack/texts/AEepoch{autoen_ep}/ValidationLossMin/validation/validation_ValMin_match_±3_epoch{num_epochs}_nofont1kind_arranged.txt'
        validation_mismatch_file_name = f'/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォント画像の研究/Transformer/result/A-Zstack/texts/AEepoch{autoen_ep}/ValidationLossMin/validation/validation_ValMin_mismatch_±3_epoch{num_epochs}_nofont1kind_arranged.txt'

        with open(validation_match_file_name, 'w') as validation_match_file, \
            open(validation_mismatch_file_name, 'w') as validation_mismatch_file:
            for i in range(len(validation_label_list_epoch)):
                # フォントリストから要素が0でないものをフィルタリング
                validation_filtered_font_list = [str(x.item()) for x in validation_font_list_for_predict[i].flatten() if x.item() != 0]
                if validation_filtered_font_list:  # フィルタリング後に要素があるか確認
                    validation_font_list_str = ', '.join(validation_filtered_font_list)
                    if validation_predicted_epoch[i] == validation_label_list_epoch[i]:
                        validation_match_file.write(f"ID: {validation_id_list_epoch[i]}, Predicted: {validation_predicted_epoch[i]}, Actual: {validation_label_list_epoch[i]}, Font List: [{validation_font_list_str}]\n")
                    else:
                        validation_mismatch_file.write(f"ID: {validation_id_list_epoch[i]}, Predicted: {validation_predicted_epoch[i]}, Actual: {validation_label_list_epoch[i]}, Font List: [{validation_font_list_str}]\n")

    # 学習モデルの保存（10epochごと）
    if ((epoch+1)%10) == 0:
        model_per10epoch_path = f'/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォント画像の研究/Transformer/result/A-Zstack/models/AEepoch{autoen_ep}/per10epoch/model_TF_epoch{epoch+1}_nofont1kind_arranged.pth'
        torch.save(model.state_dict(), model_per10epoch_path)


        createPCA_fromClstokens(predicted_epoch, label_list_epoch, cls_tokens_epoch, num_epochs, epoch, mode='train')
        createPCA_fromClstokens(validation_predicted_epoch, validation_label_list_epoch, validation_cls_tokens_epoch, num_epochs, epoch, mode='validation')

        # 100epochごとに損失と認識率のグラフ表示
        epochs = range(1, (epoch+1) + 1)
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, losses, '-o', markersize=2, label='Train Loss', color='tab:blue')
        plt.plot(epochs, validation_losses, '-o', markersize=2, label='Validation Loss', color='tab:orange')
        plt.title('Loss per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(epochs, accuracies, '-o', markersize=2, label='Train Accuracy', color='tab:blue')
        plt.plot(epochs, validation_accuracies, '-o', markersize=2, label='Validation Accuracy', color='tab:orange')
        plt.title('Accuracy per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        result_image_path = f'/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォント画像の研究/Transformer/result/A-Zstack/images/AEepoch{autoen_ep}/per10epoch/TF_2class_TraAndVal_epoch{epoch+1}_nofont1kind_arranged.png'
        plt.savefig(result_image_path)  # 画像として保存
        plt.close()


        #正しく判定しているデータはtrain_matchへ、そうでないものはtrain_mismatchへ保存
        train_match_file_name = f'/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォント画像の研究/Transformer/result/A-Zstack/texts/AEepoch{autoen_ep}/per10epoch/train/train_ValMin_match_±3_epoch{epoch+1}_nofont1kind_arranged.txt'
        train_mismatch_file_name = f'/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォント画像の研究/Transformer/result/A-Zstack/texts/AEepoch{autoen_ep}/per10epoch/train/train_ValMin_mismatch_±3_epoch{epoch+1}_nofont1kind_arranged.txt'

        with open(train_match_file_name, 'w') as train_match_file, \
            open(train_mismatch_file_name, 'w') as train_mismatch_file:
            for i in range(len(label_list_epoch)):
                # フォントリストから要素が0でないものをフィルタリング
                train_filtered_font_list = [str(x.item()) for x in font_list_for_predict[i].flatten() if x.item() != 0]
                if train_filtered_font_list:  # フィルタリング後に要素があるか確認
                    train_font_list_str = ', '.join(train_filtered_font_list)
                    if predicted_epoch[i] == label_list_epoch[i]:
                        train_match_file.write(f"ID: {id_list_epoch[i]}, Predicted: {predicted_epoch[i]}, Actual: {label_list_epoch[i]}, Font List: [{train_font_list_str}]\n")
                    else:
                        train_mismatch_file.write(f"ID: {id_list_epoch[i]}, Predicted: {predicted_epoch[i]}, Actual: {label_list_epoch[i]}, Font List: [{train_font_list_str}]\n")

        #正しく判定しているデータはvalidation_matchへ、そうでないものはvalidation_mismatchへ保存
        match_file_name = f'/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォント画像の研究/Transformer/result/A-Zstack/texts/AEepoch{autoen_ep}/per10epoch/validation/validation_match_±3_epoch{epoch+1}_nofont1kind_arranged.txt'
        mismatch_file_name = f'/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォント画像の研究/Transformer/result/A-Zstack/texts/AEepoch{autoen_ep}/per10epoch/validation/validation_mismatch_±3_epoch{epoch+1}_nofont1kind_arranged.txt'

        with open(match_file_name, 'w') as validation_match_file, \
            open(mismatch_file_name, 'w') as validation_mismatch_file:
            for i in range(len(label_list)):
                # フォントリストから要素が0でないものをフィルタリング
                validation_filtered_font_list = [str(x.item()) for x in validation_font_list_for_predict[i].flatten() if x.item() != 0]
                if validation_filtered_font_list:  # フィルタリング後に要素があるか確認
                    validation_font_list_str = ', '.join(validation_filtered_font_list)
                    if validation_predicted_epoch[i] == validation_label_list_epoch[i]:
                        validation_match_file.write(f"ID: {validation_id_list_epoch[i]}, Predicted: {validation_predicted_epoch[i]}, Actual: {validation_label_list_epoch[i]}, Font List: [{validation_font_list_str}]\n")
                    else:
                        validation_mismatch_file.write(f"ID: {validation_id_list_epoch[i]}, Predicted: {validation_predicted_epoch[i]}, Actual: {validation_label_list_epoch[i]}, Font List: [{validation_font_list_str}]\n")


print("Training finished.")
logging.info("Training finished.")
