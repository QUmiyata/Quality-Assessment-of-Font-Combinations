import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import csv
from PIL import Image
import numpy as np

# fontIDをカウントして配列に格納する関数
def count_fontID_frequency(file_path, fontID_list):
    # Download data
    list = []
    with open(file_path) as f:
        reader = csv.reader(f)
        next(reader)  # ヘッダー行をスキップ
        for r in reader:
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

@torch.no_grad()
def create_Autoencoder_plot(latent, indices_to_plot_images, indices_to_plot_labels, index, score):
    # # 出力テンソルを0-255の範囲にスケール
    # output_255 = output * 255
    # output_255_int = output_255.to(torch.uint8)  # uint8に変換

    # # 出力テンソルから文字A（最初のチャネル）を抽出
    # A_images = output_255_int[:, 0, :, :]  # shape = [261, 64, 64]

    latent_num = np.array(latent)
    # 特徴量の標準化（PCAの前処理）
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(latent_num)

    # PCAで次元削減（4096次元 -> 2次元）
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(features_scaled)

    # PCA結果を0から1にスケーリング（MinMaxScaler）
    scaler_minmax = MinMaxScaler()
    pca_result_scaled = scaler_minmax.fit_transform(pca_result)

    # 可視化
    fig, ax = plt.subplots(figsize=(20, 20))

    # PCA結果の範囲を軸に合わせる
    ax.set_xlim(pca_result_scaled[:, 0].min(), pca_result_scaled[:, 0].max())
    ax.set_ylim(pca_result_scaled[:, 1].min(), pca_result_scaled[:, 1].max())

    for i in range(pca_result_scaled.shape[0]):
        # PCAで得られた座標
        x, y = pca_result_scaled[i, 0], pca_result_scaled[i, 1]

        # 画像を指定されたインデックスに対してのみプロット
        if i in indices_to_plot_images:
            original_ID = train_fontID_frequency_index[int(i)] ## font_idが文字かもしれないので確認！！！
            # 画像のパスを取得して画像を読み込む
            image_path = f'/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォント画像の研究/AutoEncoder/久保田さん/font_crello_AtoZ_grayscale/ID{original_ID}/ID{original_ID}_A.png'  # Font_listから画像のパスを取得
            image = Image.open(image_path).convert('L')  # 画像を開く
            image = np.array(image)

            # 画像をOffsetImageとしてセット
            imagebox = OffsetImage(image, zoom=0.5, alpha=0.7)  # zoom=1.0にしてサイズを調整
            ab = AnnotationBbox(imagebox, (x, y), frameon=False, xycoords='data', boxcoords="axes fraction")
            ax.add_artist(ab)
        elif i in indices_to_plot_labels:
            # original_ID = train_fontID_frequency_index[int(i)] ## font_idが文字かもしれないので確認！！！
            # 指定されたインデックスに対しては数字（ラベル）をプロット
            # 画像をプロットしない場合は、数字（ラベル）だけをプロット
            ax.plot(x, y, 'o', markersize=5, color='black')  # 点をプロット
            # ax.text(x, y, str(i), fontsize=12, ha='center', va='center', color='black')

    # タイトルとラベル
    ax.set_title(f'PCA of A (Character A) Images - Epoch {num_epoch}')
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')

    # プロット表示
    plt.grid(True)
    
    # 画像保存
    save_path = f"/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォント画像の研究/DeepSVDD/result/images/A-Zstack/Combinations_from_score/nofont1kind_arranged/tfepoch{tfepoch}/epoch{num_epoch}/Autoencoder_plot/{score}_{index}.png"  # 保存先のパス
    plt.savefig(save_path, dpi=300)  # 画像を指定したパスに保存
    print(f"画像を保存しました: {save_path}")

    # プロットを閉じる
    plt.close()



option = 'nofont1kind_arranged'
autoen_ep = 400
num_epoch = 70
tfepoch = 70

# IDの置き換え用リスト
train_no_font1kind_path = "/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォントIDの研究/dataset/csv/train/old/train_no_font1kind_sorted.csv"
train_fontID_frequency = [0] * 262
count_fontID_frequency(train_no_font1kind_path, train_fontID_frequency)
train_fontID_frequency_index = get_indices_sorted_by_values_exclude_first(train_fontID_frequency)

# latent読み込み
latent_dict_load_path = f'/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォント画像の研究/AutoEncoder/久保田さん/result/latent/latent_with_ids_aeepoch{autoen_ep}.pth'
latent_dict_loaded = torch.load(latent_dict_load_path)
all_latent_list = []
for ID in range(1, 262):
    all_latent_list.append(latent_dict_loaded[ID].tolist())

# 1. CSVファイルを読み込む
path = f"/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォント画像の研究/dataset/csv/test/scores/{option}/tfepoch{tfepoch}/test_scores_and_fontlist_epoch{num_epoch}.csv"

indices_to_plot_labels = list(range(1, 262))

Index_list = []
Label_list = []
Score_list = []
Font_list = []
with open(path) as f:
    reader = csv.reader(f)
    next(reader)
    for r in reader:
        Index_list.append(r[0])
        Label_list.append(r[1])
        Score_list.append(r[2])
        Font_list.append(r[3:])

# 4. CSVの行をループして処理
for index, score, font_ids in zip(Index_list, Score_list, Font_list):
    int_font_ids = list(map(int, font_ids))
    create_Autoencoder_plot(latent=all_latent_list, indices_to_plot_images=int_font_ids, indices_to_plot_labels=indices_to_plot_labels, index=index, score=score)