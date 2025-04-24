# import pandas as pd
# import csv
# from PIL import Image
# import os

# # 1. CSVファイルを読み込む
# path = "/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォント画像の研究/dataset/csv/test/scores/epoch50/test_scores_and_fontlist.csv"

# # 2. 画像フォルダのパス
# image_folder = "/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォント画像の研究/AutoEncoder/久保田さん/font_crello_AtoZ_grayscale/"  # 画像ファイルが格納されているフォルダ

# # 3. 出力フォルダ
# output_folder = "/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォント画像の研究/DeepSVDD/result/images/A-Zstack/Combinations_from_score/"
# os.makedirs(output_folder, exist_ok=True)

# Index_list = []
# Label_list = []
# Score_list = []
# Font_list = []
# with open(path) as f:
#     reader = csv.reader(f)
#     next(reader)
#     for r in reader:
#         Index_list.append(r[0])
#         Label_list.append(r[1])
#         Score_list.append(r[2])
#         Font_list.append(r[3:])

# # 4. CSVの行をループして処理
# for index, score, font_ids in zip(Index_list, Score_list, Font_list):

#     # 画像を順番に取得してリストに追加
#     images = []
#     for font_id in font_ids:
#         image_filename = f"ID{font_id}/ID{font_id}_A.png"  # 例: 1_A.png
#         image_path = os.path.join(image_folder, image_filename)
        
#         if os.path.exists(image_path):
#             image = Image.open(image_path)
#             images.append(image)
#         else:
#             print(f"警告: {image_filename} が見つかりません。")

#     # 画像があれば連結して保存
#     if images:
#         # 画像を横に連結
#         widths, heights = zip(*(img.size for img in images))
#         total_width = sum(widths)
#         max_height = max(heights)
        
#         # 新しい空の画像を作成
#         new_image = Image.new('RGB', (total_width, max_height))
        
#         # 画像を配置
#         x_offset = 0
#         for img in images:
#             new_image.paste(img, (x_offset, 0))
#             x_offset += img.width

#         # 出力ファイル名を作成
#         output_filename = f"{score}_{index}.png"
#         output_path = os.path.join(output_folder, output_filename)

#         # 保存
#         new_image.save(output_path)
#         print(f"画像 {output_filename} を保存しました。")

import pandas as pd
import csv
from PIL import Image, ImageDraw, ImageFont
import os

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

# IDの置き換え用リスト
train_no_font1kind_path = "/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォントIDの研究/dataset/csv/train/old/train_no_font1kind_sorted.csv"
train_fontID_frequency = [0] * 262
count_fontID_frequency(train_no_font1kind_path, train_fontID_frequency)
train_fontID_frequency_index = get_indices_sorted_by_values_exclude_first(train_fontID_frequency)


option = 'nofont1kind_arranged'
num_epoch = 80
tfepoch = 10000

# 1. CSVファイルを読み込む
path = f"/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォント画像の研究/dataset/csv/test/scores/{option}/tfepoch{tfepoch}/test_scores_and_fontlist_epoch{num_epoch}.csv"
# path = f"/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォント画像の研究/dataset/csv/test/scores/{option}/miss/tfepoch{tfepoch}/test_scores_and_fontlist_epoch{num_epoch}.csv"

# 2. 画像フォルダのパス
image_folder = "/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォント画像の研究/AutoEncoder/久保田さん/font_crello_AtoZ_grayscale/"  # 画像ファイルが格納されているフォルダ

# 3. 出力フォルダ
output_folder = f"/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォント画像の研究/DeepSVDD/result/images/A-Zstack/Combinations_from_score/{option}/tfepoch{tfepoch}/epoch{num_epoch}"
# output_folder = f"/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォント画像の研究/DeepSVDD/result/images/A-Zstack/Combinations_from_score/{option}/miss/tfepoch{tfepoch}/epoch{num_epoch}/score_fontlist_arrangedID/"

os.makedirs(output_folder, exist_ok=True)

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

    # 画像を順番に取得してリストに追加
    images = []
    for font_id in font_ids:
        # 単一フォント除外あり・フォントID並び替えあり
        original_ID = train_fontID_frequency_index[int(font_id)] ## font_idが文字かもしれないので確認！！！
        image_filename = f"ID{original_ID}/ID{original_ID}_A.png"  # 例: 1_A.png

        # image_filename = f"ID{font_id}/ID{font_id}_A.png"  # 例: 1_A.png
        image_path = os.path.join(image_folder, image_filename)
        
        if os.path.exists(image_path):
            image = Image.open(image_path)

            # 画像にID番号を右上に追加
            draw = ImageDraw.Draw(image)
            # font = ImageFont.load_default()  # デフォルトフォントを使用
            # フォントを小さく設定（ここでサイズを調整）
            try:
                # フォントを小さく設定、サイズは8
                font = ImageFont.truetype("/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf", size=1)
            except IOError:
                font = ImageFont.load_default()  # フォントが読み込めない場合、デフォルトフォントを使用

            # text = str(original_ID)  # 描画するID番号
            text = font_id  # 描画するID番号

            # テキストのバウンディングボックスを計算
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            # 文字を描画する位置
            position = (1, 0.5)  # 右上から少し余裕を取る

            # テキストを描画
            draw.text(position, text, font=font, fill="black")  # 文字色は白

            images.append(image)
        else:
            print(f"警告: {image_filename} が見つかりません。")

    # 画像があれば連結して保存
    if images:
        # 画像を横に連結
        widths, heights = zip(*(img.size for img in images))
        total_width = sum(widths)
        max_height = max(heights)
        
        # 新しい空の画像を作成
        new_image = Image.new('RGB', (total_width, max_height))
        
        # 画像を配置
        x_offset = 0
        for img in images:
            new_image.paste(img, (x_offset, 0))
            x_offset += img.width

        # 出力ファイル名を作成
        output_filename = f"{score}_{index}.png"
        output_path = os.path.join(output_folder, output_filename)

        # 保存
        new_image.save(output_path)
        print(f"画像 {output_filename} を保存しました。")
