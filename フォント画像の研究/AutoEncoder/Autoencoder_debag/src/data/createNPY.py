import os
import numpy as np
from PIL import Image

for ID in range(1, 262):
    # 画像ファイルが格納されているディレクトリのパス
    folder_path = f'/home/miyatamoe/ドキュメント/研究/久保田さん/font_crello_AtoZ_grayscale/ID{ID}/'

    # PNGファイルをリストアップ
    image_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]

    # 画像をチャンネル方向に重ねるためのリスト
    image_stack = []

    # 画像を順番に読み込み、NumPy配列としてリストに追加
    for image_file in sorted(image_files):
        image_path = os.path.join(folder_path, image_file)
        
        # 画像を開く
        img = Image.open(image_path)
        
        # 画像をNumPy配列に変換 (H, W, C) 形式にする
        img_array = np.array(img)
        
        # チャンネル方向に追加
        image_stack.append(img_array)

    # リストをNumPy配列に変換 (N, H, W, C) 形式にする
    image_stack = np.stack(image_stack, axis=0)

    # NumPy配列を保存 (例えば、npz形式で保存)
    np.save(f'/home/miyatamoe/ドキュメント/研究/久保田さん/npy/ID{ID}stacked_images.npy', image_stack)

    # 保存完了
    print(f"画像が保存されました。保存先: 'stacked_images.npy'")
