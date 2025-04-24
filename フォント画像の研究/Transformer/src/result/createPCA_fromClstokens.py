import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np

option = '_original'

@torch.no_grad()
def createPCA_fromClstokens(predicted_list_epoch, label_list_epoch, cls_tokens_epoch, num_epochs, epoch, mode, isValLossMin=False):
    cls_tokens_epoch_num = np.array(cls_tokens_epoch)
    # 特徴量の標準化（PCAの前処理）
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(cls_tokens_epoch_num)

    # PCAで次元削減（512次元 -> 2次元）
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(features_scaled)

    # # PCA結果を0から1にスケーリング（MinMaxScaler）
    # scaler_minmax = MinMaxScaler()
    # pca_result_scaled = scaler_minmax.fit_transform(pca_result)

    # 可視化
    fig, ax = plt.subplots(figsize=(20, 8))

    # PCA結果の範囲を軸に合わせる
    ax.set_xlim(pca_result[:, 0].min(), pca_result[:, 0].max())
    ax.set_ylim(pca_result[:, 1].min(), pca_result[:, 1].max())

    # for i in range(pca_result.shape[0]):
    #     # PCAで得られた座標
    #     x, y = pca_result[i, 0], pca_result[i, 1]

        # # 数字をPCAで得られた座標にプロット
        # ax.text(x, y, str(i + 131), fontsize=3, ha='center', va='center', alpha=0.7)


    colors = []
    for predicted, label in zip(predicted_list_epoch, label_list_epoch):
        if (predicted == label):
            colors.append('blue')
        else:
            colors.append('red')

    plt.scatter(x=pca_result[:, 0], y=pca_result[:, 1], c=colors, s=2)

    # タイトルとラベル
    ax.set_title('PCA of A (Character A) Images')
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')

    # プロット表示
    plt.grid(True)
    
    # 画像保存
    if (isValLossMin):
        save_path = f"/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォント画像の研究/Transformer/result/A-Zstack/images/cls_tokens/{mode}/ValLossMin/cls_tokens_pca_epoch{num_epochs}_ValLossMin{option}.png"  # 保存先のパス
    else:
        save_path = f"/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォント画像の研究/Transformer/result/A-Zstack/images/cls_tokens/{mode}/cls_tokens_pca_epoch{epoch+1}{option}.png"  # 保存先のパス
    plt.savefig(save_path, dpi=300)  # 画像を指定したパスに保存
    # print(f"画像を保存しました: {save_path}")

    # プロットを閉じる
    plt.close()
