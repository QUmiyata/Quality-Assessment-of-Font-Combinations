# 使用頻度を参照したいデータをロードして、fontID->要素番号(fontIDをその格納されている要素番号と等しいIDに変換する)を満たすような配列を用意
# fontIDを書き換えたいデータと用意した配列を見比べながら新しいcsvファイルを作成
# （good_data_sequencesに変更前のfontlistを全部格納．各組み合わせの要素数だけループして１文字ずつ変換してsublistにappend．変換の際は用意した配列からIDを探して要素番号を取得し，それをappend．要素数だけループしたら外枠のlistにappend．これをlen(good_data_sequences)だけループ．）


### 使用頻度を参照したいデータをロードして、fontID->要素番号(fontIDをその格納されている要素番号と等しいIDに変換する)を満たすような配列を用意 ###
import csv
from tqdm import tqdm
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

train_data_path = "/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォントIDの研究/dataset/csv/train/old/train_no_font1kind_sorted.csv" # 固定！
# validation_data_path = "/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォントIDの研究/dataset/csv/validation/validation_no_font1kind_sorted.csv"
# test_data_path = "/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォントIDの研究/dataset/csv/test/test_no_font1kind_sorted.csv"

# ロードしたデータから長さ262の配列のfontIDの位置に，そのfontIDの頻度を格納する
# fontID_frequency = [0] * 262
train_fontID_frequency = [0] * 262
# validation_fontID_frequency = [0] * 262
# test_fontID_frequency = [0] * 262

count_fontID_frequency(train_data_path, train_fontID_frequency)
# count_fontID_frequency(validation_data_path, validation_fontID_frequency)
# count_fontID_frequency(test_data_path, test_fontID_frequency)

# fontID_frequency = [a + b + c for a, b, c in zip(train_fontID_frequency, validation_fontID_frequency, test_fontID_frequency)]
# index_list = get_indices_sorted_by_values_exclude_first(fontID_frequency)
index_list = get_indices_sorted_by_values_exclude_first(train_fontID_frequency)
print(index_list)


### good_data_sequencesに変更前のfontlistを全部格納 ###
good_data_path = "/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォントIDの研究/dataset/csv/train/old/train_no_font1kind_sorted.csv"
# 既存の良いデータのセットを読み込む
id_list = []
len_list = []
good_data_sequences = []
with open(good_data_path, newline='') as f:
    reader = csv.reader(f)
    next(reader)  # ヘッダー行をスキップ
    for row in reader:
        id_list.append(row[0])
        len_list.append(row[1])
        sequence = [int(num) for num in row[2:]]  # idとlengthの要素を飛ばす
        good_data_sequences.append(sequence)

### これをlen(good_data_sequences)だけループ ###
arranged_font_list = []
for num in range(len(good_data_sequences)):
    ### 各組み合わせの要素数だけループして１文字ずつ変換してsublistにappend ###
    sub_list = []
    for ori_fontID in good_data_sequences[num]:
        sub_list.append(index_list.index(ori_fontID))
    arranged_font_list.append(sorted(sub_list))

### 新しいcsvファイルを作成 ###
header = ['id', 'length', 'font']
with open('/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォントIDの研究/dataset/csv/train/train_no_font1kind_arranged_sorted2.csv', 'w') as f:
  writer = csv.writer(f)
  writer.writerow(header)
  for i, j, k in zip(id_list, len_list, arranged_font_list):
    body =  [i, j] + k
    writer.writerow(body)
f.close()