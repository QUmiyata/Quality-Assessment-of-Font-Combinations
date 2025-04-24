import csv

good_data_path = f"/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォントIDの研究/dataset/csv/train/train_no_font1kind_arranged_sorted.csv"
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

# print(good_data_sequences)
print(len(good_data_sequences))

# count = sum(1 for sublist in good_data_sequences if 3 in sublist)
# print(count)

#両方
count = sum(1 for sublist in good_data_sequences if 1 in sublist and 3 in sublist)
print(count)
