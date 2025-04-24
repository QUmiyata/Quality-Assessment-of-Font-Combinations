import torch
import csv

# # サンプルのリスト（リストのリスト）
# list1 = [[1, 2, 3], [3, 4], [6, 7, 8]]
# list2 = [[3, 4], [1, 5, 6], [7, 8]]

train_path = '/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォントIDの研究/dataset/csv/train/train_no_font1kind_arranged_sorted.csv'
test_path = '/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォントIDの研究/dataset/csv/test/test_no_font1kind_arranged_sorted.csv'

train_font_list = []
with open(train_path) as f:
    reader = csv.reader(f)
    for r in reader:
        train_font_list.append(r[2:])
train_font_list = train_font_list[1:]

test_font_list = []
with open(test_path) as f:
    reader = csv.reader(f)
    for r in reader:
        test_font_list.append(r[2:])
test_font_list = test_font_list[1:]

# サブリストをセットに変換（順番を気にせず一致する要素を確認できるようにする）
traint_list = [sublist for sublist in train_font_list]
test_list = [sublist for sublist in test_font_list]

# 重複のカウント    
common_sublist_count = sum([1 for s2 in test_list if any(s2 == s1 for s1 in traint_list)])

# 結果を表示
print(f"リスト全体の重複数: {common_sublist_count}")
