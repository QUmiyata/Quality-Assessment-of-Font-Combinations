import torch
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

@torch.no_grad()
def createInputOutput(input, output, epoch):
    # 0〜1の範囲を0〜255にスケール
    input_255 = input * 255
    input_255_int = input_255.to(torch.uint8)  # uint8に変換

    # 保存先ディレクトリ（必要に応じて作成）
    input_dir = f"/home/miyatamoe/ドキュメント/研究/久保田さん/result/images/input/epoch{epoch}"
    os.makedirs(input_dir, exist_ok=True)

    # バッチ内の各画像（16個の画像）について処理
    for i in range(input_255_int.shape[0]):  # バッチの数（16個）
        # 1つの大きな画像を作成 (高さ128、幅832の空のテンソル)
        final_image = torch.zeros((2 * 64, 13 * 64), dtype=torch.uint8)  # 縦64×2, 横64×13

        # 各バッチの26チャネル（文字A〜Z）を2列×13列に並べる
        for j in range(input_255_int.shape[1]):  # 26チャネル（A〜Z）
            # 出力テンソルのサイズは(64, 64)
            image_tensor = input_255_int[i, j]  # (64, 64) のテンソル

            # 最終画像内の配置場所（縦2列、横13列）
            row = j // 13  # 行の位置（0または1）
            col = j % 13   # 列の位置（0〜12）

            # 各画像をfinal_imageに配置
            final_image[row * 64: (row + 1) * 64, col * 64: (col + 1) * 64] = image_tensor

        # final_imageをNumPy配列に変換
        final_image_np = final_image.numpy()

        # NumPy配列から画像を作成
        final_image_pil = Image.fromarray(final_image_np)

        # 画像を保存（ファイル名にバッチインデックスを含める）
        final_image_pil.save(f"{input_dir}/batch_{i}_combined_image.png")


    # 0〜1の範囲を0〜255にスケール
    output_255 = output * 255
    output_255_int = output_255.to(torch.uint8)  # uint8に変換

    # 保存先ディレクトリ（必要に応じて作成）
    output_dir = f"/home/miyatamoe/ドキュメント/研究/久保田さん/result/images/output/epoch{epoch}"
    os.makedirs(output_dir, exist_ok=True)

    # バッチ内の各画像（16個の画像）について処理
    for i in range(output_255_int.shape[0]):  # バッチの数（16個）
        # 1つの大きな画像を作成 (高さ128、幅832の空のテンソル)
        final_image = torch.zeros((2 * 64, 13 * 64), dtype=torch.uint8)  # 縦64×2, 横64×13

        # 各バッチの26チャネル（文字A〜Z）を2列×13列に並べる
        for j in range(output_255_int.shape[1]):  # 26チャネル（A〜Z）
            # 出力テンソルのサイズは(64, 64)
            image_tensor = output_255_int[i, j]  # (64, 64) のテンソル

            # 最終画像内の配置場所（縦2列、横13列）
            row = j // 13  # 行の位置（0または1）
            col = j % 13   # 列の位置（0〜12）

            # 各画像をfinal_imageに配置
            final_image[row * 64: (row + 1) * 64, col * 64: (col + 1) * 64] = image_tensor

        # final_imageをNumPy配列に変換
        final_image_np = final_image.numpy()

        # NumPy配列から画像を作成
        final_image_pil = Image.fromarray(final_image_np)

        # 画像を保存（ファイル名にバッチインデックスを含める）
        final_image_pil.save(f"{output_dir}/batch_{i}_combined_image.png")

    print("画像の保存が完了しました。")


@torch.no_grad()
def visualize_pca_images(latent, output, epoch):
    # # 保存先ディレクトリ（必要に応じて作成）
    # input_dir = f"/home/miyatamoe/ドキュメント/研究/久保田さん/result/images/feature"
    # os.makedirs(input_dir, exist_ok=True)

    # 出力テンソルを0-255の範囲にスケール
    output_255 = output * 255
    output_255_int = output_255.to(torch.uint8)  # uint8に変換

    # 出力テンソルから文字A（最初のチャネル）を抽出
    A_images = output_255_int[:, 0, :, :]  # shape = [261, 64, 64]

    latent_num = latent.detach().cpu().numpy()
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

        # 画像を取り出してプロット
        image = A_images[i].cpu().numpy()  # バッチiの文字Aの画像（64x64）

        # 画像をOffsetImageとしてセット
        imagebox = OffsetImage(image, zoom=0.2, alpha=0.7)  # zoom=1.0にしてサイズを調整
        ab = AnnotationBbox(imagebox, (x, y), frameon=False, xycoords='data', boxcoords="axes fraction")
        ax.add_artist(ab)

    # タイトルとラベル
    ax.set_title('PCA of A (Character A) Images')
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')

    # プロット表示
    plt.grid(True)
    
    # 画像保存
    save_path = f"/home/miyatamoe/ドキュメント/研究/久保田さん/result/images/feature/latent_pca_epoch{epoch}_square.png"  # 保存先のパス
    plt.savefig(save_path, dpi=300)  # 画像を指定したパスに保存
    print(f"画像を保存しました: {save_path}")

    # プロットを閉じる
    plt.close()


def plot_loss(train_loss_history, epoch):
    '''Plot and save the training loss history.'''
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(train_loss_history)), train_loss_history, label="Train Loss", color='blue', linestyle='-', marker='')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    
    # Save the plot
    plt.savefig(f"/home/miyatamoe/ドキュメント/研究/久保田さん/result/images/loss/trainloss_epoch{epoch}.png")
    plt.close()