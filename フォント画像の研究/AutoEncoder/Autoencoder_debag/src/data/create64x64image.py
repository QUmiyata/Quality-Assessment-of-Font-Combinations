# 画像加工
from torchvision import transforms
from PIL import Image
import glob
import os

def pad_to_square(image, fill_color=(255, 255, 255)):
    # 画像のサイズを取得
    img_width, img_height = image.size
    # 新しいサイズを決定
    new_size = max(img_width, img_height)
    # 新しい画像を作成
    new_image = Image.new('RGB', (new_size, new_size), fill_color)
    # 元の画像を新しい画像の中央にペースト
    new_image.paste(image, (int((new_size - img_width) / 2), int((new_size - img_height) / 2)))
    return new_image

def transform_image(image_path):
    # 画像を開く
    image = Image.open(image_path).convert('RGBA')
    background = Image.new('RGB', image.size, (255, 255, 255))
    # 元の画像を背景画像に合成
    background.paste(image, (0, 0), image)
    # 画像を正方形にパディング
    image = pad_to_square(background)
    # 画像をテンソルに変換
    transform = transforms.Compose([
        transforms.Resize((64, 64)), # 画像サイズを統一
        transforms.ToTensor()  # テンソルに変換
    ])
    image_tensor = transform(image)
    return image_tensor


img_folder = '/home/miyatamoe/ドキュメント/研究/久保田さん/font_crello_AtoZ_original'
alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

im_list = []
for ID in range(1, 262):
    im_sublist = sorted(glob.glob(f'{img_folder}/ID{ID}/*.png', recursive=True))
    im_list.append(im_sublist)

for ID in range(1, 262):
    for idx, char in enumerate(alphabet):
        image = Image.open(im_list[ID-1][idx]).convert('RGBA')
        background = Image.new('RGB', image.size, (255, 255, 255))
        # 元の画像を背景画像に合成
        background.paste(image, (0, 0), image)

        # 画像を正方形にパディング
        image = pad_to_square(background)

        # grayscaleに変更
        image_L = image.convert("L")

        image_resized = image_L.resize((64, 64))

        # 保存先のフォルダを作成（存在しなければ）
        save_folder = f'/home/miyatamoe/ドキュメント/研究/久保田さん/font_crello_AtoZ_grayscale/ID{ID}'
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        # 保存先のファイルパスを作成（例: 'grayscale_images' フォルダ）
        save_path = f'{save_folder}/ID{ID}_{char}.png'
        
        # 画像を保存
        image_resized.save(save_path)