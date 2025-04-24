import os
import numpy as np
from PIL import Image

for ID in range(1, 262):
    # 保存されたNumPy配列のパス
    npy_file_path = f'/home/miyatamoe/ドキュメント/研究/久保田さん/npy/ID{ID}stacked_images.npy'

    # NumPy配列を読み込み
    image_stack = np.load(npy_file_path)

    # スタックされた画像を1枚ずつ復元して保存
    for idx, img_array in enumerate(image_stack):
        # img_array の形状が (H, W, 1) なら squeeze して (H, W) にする
        if img_array.ndim == 3 and img_array.shape[-1] == 1:
            img_array = np.squeeze(img_array, axis=-1)  # (H, W, 1) -> (H, W)

        # PIL画像に変換
        img = Image.fromarray(img_array)

        # 保存先のディレクトリとファイル名を作成
        save_folder = f'/home/miyatamoe/ドキュメント/研究/久保田さん/font_crello_AtoZ_reconstructed(debug)/ID{ID}/'
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        save_path = f'{save_folder}ID{ID}_{idx+1}.png'

        # 画像を保存
        img.save(save_path)

    print(f"ID{ID}の画像が復元され、保存されました。")
