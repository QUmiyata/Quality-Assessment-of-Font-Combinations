import torch
import csv
from tqdm import tqdm
import itertools

# fontIDを使用頻度順に並び替える（IDを付け替える）
# train, validation, testのプロデータをロード
train_data_path = "/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォントIDの研究/dataset/csv/train/old/train_no_font1kind_sorted.csv"
# train_data_path = "/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォントIDの研究/dataset/csv/train/old data/train_fonts_no_0.csv"
# validation_data_path = "/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォントIDの研究/dataset/csv/validation/validation_no_font1kind_sorted.csv"
# test_data_path = "/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォントIDの研究/dataset/csv/test/test_no_font1kind_sorted.csv"

# ロードしたデータから長さ262の配列のfontIDの位置に，そのfontIDの頻度を格納する
fontID_frequency = [0] * 262
train_fontID_frequency = [0] * 262
validation_fontID_frequency = [0] * 262
test_fontID_frequency = [0] * 262

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

count_fontID_frequency(train_data_path, train_fontID_frequency)
# count_fontID_frequency(validation_data_path, validation_fontID_frequency)
# count_fontID_frequency(test_data_path, test_fontID_frequency)

fontID_frequency = [a + b + c for a, b, c in zip(train_fontID_frequency, validation_fontID_frequency, test_fontID_frequency)]

train_fontID_frequency_index = get_indices_sorted_by_values_exclude_first(train_fontID_frequency)
print(get_indices_sorted_by_values_exclude_first(train_fontID_frequency))

print(train_fontID_frequency)


# # 対象のリスト
# elements = train_fontID_frequency_index

# # ログファイルのパス
# log_file_path = '/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォントIDの研究/dataset/log/arrange_fontID.txt'  # 実際のファイルパスに変更してください

# # ログファイルに書き込む
# with open(log_file_path, 'w') as log_file:
#     for idx, element in enumerate(elements):
#         # log_file.write(f"Index: {idx}, Element: {element}\n")
#         log_file.write(f"Element: {element} → Index: {idx}\n")

# print(f"ログファイルが作成されました: {log_file_path}")
