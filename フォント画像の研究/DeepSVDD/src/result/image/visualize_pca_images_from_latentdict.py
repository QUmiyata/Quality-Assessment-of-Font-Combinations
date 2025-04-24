import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np

@torch.no_grad()
def visualize_pca_images_from_latentdict():
    ae_epoch = 400
    # 保存したファイルから辞書を読み込む
    latent_dict_load_path = f'/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォント画像の研究/AutoEncoder/久保田さん/result/latent/latent_with_ids_aeepoch{ae_epoch}.pth'
    latent_dict_loaded = torch.load(latent_dict_load_path)

    latent = []
    for id in range(131, 262):
        latent.append(latent_dict_loaded[id].tolist())

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
    fig, ax = plt.subplots(figsize=(20, 8))

    # PCA結果の範囲を軸に合わせる
    ax.set_xlim(pca_result_scaled[:, 0].min(), pca_result_scaled[:, 0].max())
    ax.set_ylim(pca_result_scaled[:, 1].min(), pca_result_scaled[:, 1].max())

    for i in range(pca_result_scaled.shape[0]):
        # PCAで得られた座標
        x, y = pca_result_scaled[i, 0], pca_result_scaled[i, 1]

        # 数字をPCAで得られた座標にプロット
        ax.text(x, y, str(i + 131), fontsize=3, ha='center', va='center', alpha=0.7)

    # タイトルとラベル
    ax.set_title('PCA of A (Character A) Images')
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')

    # プロット表示
    plt.grid(True)
    
    # 画像保存
    save_path = f"/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォント画像の研究/AutoEncoder/久保田さん/result/images/feature/number/latent_pca_epoch{ae_epoch}_number_131to261.png"  # 保存先のパス
    plt.savefig(save_path, dpi=300)  # 画像を指定したパスに保存
    print(f"画像を保存しました: {save_path}")

    # プロットを閉じる
    plt.close()


visualize_pca_images_from_latentdict()