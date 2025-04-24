import csv
import random

mode = 'test'

good_data_path = f"/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォントIDの研究/dataset/csv/{mode}/{mode}_no_font1kind_arranged_sorted.csv"
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


train_high_path = '/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォントIDの研究/dataset/csv/train/train_no_font1kind_arranged_sorted.csv'
val_high_path = '/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォントIDの研究/dataset/csv/validation/validation_no_font1kind_arranged_sorted.csv'
test_high_path = '/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォントIDの研究/dataset/csv/test/test_no_font1kind_arranged_sorted.csv'

train_id_list = []
train_len_list = []
train_font_list = []
train_label_list = []
with open(train_high_path) as f:
    reader = csv.reader(f)
    next(reader)
    for r in reader:
        train_id_list.append(r[0])
        train_len_list.append(r[1])
        train_font_list.append(r[2:])
        train_label_list.append(0)

val_id_list = []
val_len_list = []
val_font_list = []
val_label_list = []
with open(val_high_path) as f:
    reader = csv.reader(f)
    next(reader)
    for r in reader:
        val_id_list.append(r[0])
        val_len_list.append(r[1])
        val_font_list.append(r[2:])
        val_label_list.append(0)

test_id_list = []
test_len_list = []
test_font_list = []
test_label_list = []
with open(test_high_path) as f:
    reader = csv.reader(f)
    next(reader)
    for r in reader:
        test_id_list.append(r[0])
        test_len_list.append(r[1])
        test_font_list.append(r[2:])
        test_label_list.append(0)

# 'id', 'length', 'font'の文字、最初の意味のないラベルを削除
id_list_all = []  
len_list_all = []
font_list_all = []

id_list_all.extend(train_id_list)
id_list_all.extend(val_id_list)
id_list_all.extend(test_id_list)

len_list_all.extend(train_len_list)
len_list_all.extend(val_len_list)
len_list_all.extend(test_len_list)

font_list_all.extend(train_font_list)
font_list_all.extend(val_font_list)
font_list_all.extend(test_font_list)

font_list_all_int = [[int(x) for x in sublist] for sublist in font_list_all]

header = ['id', 'length', 'font']

# ランダムなフォントリストを作成
random_font_list = [] # 出力するランダムリスト
random_id_list = [] # 出力するランダムリスト
random_len_list = [] # 出力するランダムリスト
for num in range(len(good_data_sequences)):
  original_font_list = good_data_sequences[num] # 元のフォントリスト
  original_unique_font_list = sorted(list(set(good_data_sequences[num]))) # 元のフォントリストの重複を除いたリスト
  random_font_sub_list = [] # 各組み合わせのランダム版を格納するフォントリスト

  for i in range(len(original_unique_font_list)):
    if (original_unique_font_list[i] == 1):
      random_fontID = random.randint(2, 4)
    elif (original_unique_font_list[i] == 2 or original_unique_font_list[i] == 3):
      while(1):
        random_fontID = random.randint(1, original_unique_font_list[i]+3)
        if (original_unique_font_list[i] != random_fontID): break
    elif (original_unique_font_list[i] == 259 or original_unique_font_list[i] == 260):
      while(1):
        random_fontID = random.randint(original_unique_font_list[i]-3, 261)
        if (original_unique_font_list[i] != random_fontID): break
    elif (original_unique_font_list[i] == 261):
      while(1):
        random_fontID = random.randint(258, 260)
        if (original_unique_font_list[i] != random_fontID): break
    else:
      while(1):
        random_fontID = random.randint(original_unique_font_list[i]-3, original_unique_font_list[i]+3)
        if (original_unique_font_list[i] != random_fontID): break
    
    for j in range(original_font_list.count(original_unique_font_list[i])):
      random_font_sub_list.append(random_fontID)
    
  # while ((sorted(random_font_sub_list) in font_list_all) or (len(set(random_font_sub_list))==1)): # 良いデータと被っていないか確認 + フォントが1種類じゃないか
  while ((sorted(random_font_sub_list) in font_list_all_int) or (len(set(random_font_sub_list))==1)): # 良いデータと被っていないか確認 + フォントが1種類じゃないか
    random_font_sub_list = []
    for i in range(len(original_unique_font_list)):
      if (original_unique_font_list[i] == 1):
        random_fontID = random.randint(2, 4)
      elif (original_unique_font_list[i] == 2 or original_unique_font_list[i] == 3):
        while(1):
          random_fontID = random.randint(1, original_unique_font_list[i]+3)
          if (original_unique_font_list[i] != random_fontID): break
      elif (original_unique_font_list[i] == 259 or original_unique_font_list[i] == 260):
        while(1):
          random_fontID = random.randint(original_unique_font_list[i]-3, 261)
          if (original_unique_font_list[i] != random_fontID): break
      elif (original_unique_font_list[i] == 261):
        while(1):
          random_fontID = random.randint(258, 260)
          if (original_unique_font_list[i] != random_fontID): break
      else:
        while(1):
          random_fontID = random.randint(original_unique_font_list[i]-3, original_unique_font_list[i]+3)
          if (original_unique_font_list[i] != random_fontID): break
        
      for j in range(original_font_list.count(original_unique_font_list[i])):
        random_font_sub_list.append(random_fontID)
    
  random_font_list.append(sorted(random_font_sub_list))
  random_id_list.append(id_list[num])
  random_len_list.append(len_list[num])


with open(f'/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォントIDの研究/dataset/csv/{mode}/{mode}_no_font1kind_random_arranged_sorted.csv', 'w') as f:
  writer = csv.writer(f)
  writer.writerow(header)
  for i, j, k in zip(random_id_list, random_len_list, random_font_list):
    body =  ["random"+i, j] + k
    writer.writerow(body)
f.close()



# プロの組み合わせの各リストを取り出す -> 重複を除いたリストを作成 -> このリストの長さでループ -> 各要素を+-3のうちからランダムに選んで元のリストのlist.count(number)で調べる
# -> その長さ分ランダム数字をコピー -> 事前に作成しておいたlist[プロの各リスト]に書いて、外側の配列に追加