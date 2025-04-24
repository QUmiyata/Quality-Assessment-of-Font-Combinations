import torch
import torch.nn as nn
from torch.nn import functional
from dataloader import Dataloader_TF
from Transformer import Transformer
import numpy as np
from result.createPCA_fromClstokens import createPCA_fromClstokens
from tqdm import tqdm
import csv

# def fix_seed(seed):
#     np.random.seed(seed)
#     torch.manual_seed(seed)  # fix the initial value of the network weight
#     torch.cuda.manual_seed(seed)  # for cuda
#     torch.cuda.manual_seed_all(seed)  # for multi-GPU
#     torch.backends.cudnn.deterministic = True  # choose the determintic algorithm

# # シード固定
# fix_seed(0)

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


# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mode = 'validation'
option = '_nofont1kind_arranged'
# # ハイパーパラメータの設定
if (option == '_nofont1kind_arranged'):
    if (mode == 'train'):
        test_data_path = "/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォントIDの研究/dataset/csv/train/train_no_font1kind_arranged_sorted.csv"
        test_random_data_path = "/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォントIDの研究/dataset/csv/train/train_no_font1kind_random_arranged_sorted.csv"
    elif (mode == 'test'):
        test_data_path = "/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォントIDの研究/dataset/csv/test/test_no_font1kind_arranged_sorted.csv"
        test_random_data_path = "/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォントIDの研究/dataset/csv/test/test_no_font1kind_random_arranged_sorted.csv"
    elif (mode == 'validation'):
        test_data_path = "/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォントIDの研究/dataset/csv/validation/validation_no_font1kind_arranged_sorted.csv"
        test_random_data_path = "/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォントIDの研究/dataset/csv/validation/validation_no_font1kind_random_arranged_sorted.csv"
if (option == '_original'):
    test_data_path = "/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォントIDの研究/dataset/csv/train/old/train_fonts_no_0_sorted.csv"
    test_random_data_path = "/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォントIDの研究/dataset/csv/train/old/train_fonts_random_no_0_range±3.csv"
    # test_data_path = "/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォントIDの研究/dataset/csv/test/old/test_fonts_no_0_sorted.csv"
    # test_random_data_path = "/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォントIDの研究/dataset/csv/test/old/test_fonts_random_no_0_range±3.csv"
batch_size = 128
enc_layers = 2
disc_layers = 2
d_input = 128
max_seq_len = 44
d_model = 128
nhead = 8
dim_feedforward = 256
dropout = 0.5

num_epochs = 50

############ test ##############

# データローダーの初期化
test_dataloader = Dataloader_TF(test_data_path, test_random_data_path, batch_size, is_training=False)

# モデルの初期化
model = Transformer(num_classes=2, embed_dim=512, depth=enc_layers)

# モデルの準備
model.load_state_dict(torch.load(f'/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォントIDの研究/Transformer/result/models/model_TF_ValLossMin_epoch10000_no_font1kind_arranged.pth'))
# model.load_state_dict(torch.load(f'/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォントIDの研究/Transformer/result/models/AEepoch400/model_TF_ValLossMin_epoch3000.pth'))
model = model.to(device)

# # IDの置き換え用リスト
# train_no_font1kind_path = "/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォントIDの研究/dataset/csv/train/old/train_no_font1kind_sorted.csv"
# train_fontID_frequency = [0] * 262
# count_fontID_frequency(train_no_font1kind_path, train_fontID_frequency)
# train_fontID_frequency_index = get_indices_sorted_by_values_exclude_first(train_fontID_frequency)

criterion = nn.BCELoss()
# 損失と認識率を保存するリスト
test_losses = []
test_accuracies = []

test_font_list_epoch = []
test_len_list_epoch = []
test_id_list_epoch = []
test_label_list_epoch = []
test_predicted_epoch = []
test_cls_tokens_epoch = []
all_cls_tokens = []

# テストデータを用いた評価
model.eval()
test_correct = 0
test_total = 0
test_loss = 0.0
with torch.no_grad():
    for font_list, len_list, id_list, label_list in test_dataloader.yield_batched_data():
        # 成功例、失敗例出力用のlist
        test_font_list_epoch.extend(font_list)
        test_len_list_epoch.extend(len_list)
        test_id_list_epoch.extend(id_list)
        test_label_list_epoch.extend(label_list)

        font_list = torch.tensor(font_list, dtype=torch.int64).to(device)
        len_list = torch.tensor(len_list, dtype=torch.int64).to(device)
        label_list = torch.tensor(label_list, dtype=torch.float32).to(device)

        # 1-hotベクトルに変換
        one_hot_font_list = nn.functional.one_hot(font_list, num_classes=262)
        one_hot_font_list = one_hot_font_list.float()
        # Input embedding
        int_one_hot_font_list = one_hot_font_list.to(device)
        # モデルに入力
        outputs, cls_tokens = model(int_one_hot_font_list, len_list=len_list)

        all_cls_tokens.extend(cls_tokens)
        test_cls_tokens_epoch.extend(cls_tokens.tolist())

        # 損失の計算
        loss = criterion(outputs, label_list)
        test_loss += loss.item() * len(font_list)

        # 予測値の取得
        predicted = torch.round(outputs)
        test_predicted_epoch.extend(predicted.tolist())

        test_total += label_list.size(0)
        test_correct += (predicted == label_list).sum().item()

        # テキスト出力用font list作成
        idx = 0
        font_list_for_predict = []
        for length in len_list:
            font_list_for_predict.append(font_list[idx : idx + length].tolist())
            idx += length
        font_list_for_predict = pad_data(font_list_for_predict, max_seq_len)
        font_list_for_predict = torch.tensor(font_list_for_predict, dtype=torch.int64).to(device) 
        
        # #正しく判定しているデータはtest_matchへ、そうでないものはtest_mismatchへ保存
        # with open(f'/home/miyatamoe/ドキュメント/研究（支部大会2024）/SIBU_font_image/results/texts/test/test_match_±3_epoch{num_epochs}.txt', 'a') as match_file, \
        #      open(f'/home/miyatamoe/ドキュメント/研究（支部大会2024）/SIBU_font_image/results/texts/test/test_mismatch_±3_epoch{num_epochs}.txt', 'a') as mismatch_file:
        #     for i in range(len(label_list)):
        #         # フォントリストから要素が0でないものをフィルタリング
        #         filtered_font_list = [str(x.item()) for x in font_list_for_predict[i].flatten() if x.item() != 0]
        #         if filtered_font_list:  # フィルタリング後に要素があるか確認
        #             font_list_str = ', '.join(filtered_font_list)
        #             if predicted[i] == label_list[i]:
        #                 match_file.write(f"ID: {id_list[i]}, Predicted: {predicted[i].item()}, Actual: {label_list[i].item()}, Font List: [{font_list_str}]\n")
        #             else:
        #                 mismatch_file.write(f"ID: {id_list[i]}, Predicted: {predicted[i].item()}, Actual: {label_list[i].item()}, Font List: [{font_list_str}]\n")


    test_avg_loss = test_loss / len(test_dataloader.idx_list)
    test_losses.append(test_avg_loss)
    test_accuracy = test_correct / test_total
    test_accuracies.append(test_accuracy)
    print(f"Test Loss: {test_avg_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    createPCA_fromClstokens(test_predicted_epoch, test_label_list_epoch, test_cls_tokens_epoch, num_epochs, epoch=0, mode=mode, isValLossMin=True)

    # cls_tokenの保存
    clstoken_dict = {}
    for idx in range(len(all_cls_tokens)):
        clstoken_dict[test_id_list_epoch[idx]] = all_cls_tokens[idx]  # IDとlatentを辞書に格納
    # 保存するファイルパス
    # save_path = f'/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォントIDの研究/Transformer/result/cls_tokens/{mode}/clstoken_with_ids_tfepoch{num_epochs}{option}.pth'
    save_path = f'/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォントIDの研究/Transformer/result/cls_tokens/{mode}/clstoken_with_ids_tf_vallossminepoch10000{option}.pth'
    # 辞書を保存
    torch.save(clstoken_dict, save_path)