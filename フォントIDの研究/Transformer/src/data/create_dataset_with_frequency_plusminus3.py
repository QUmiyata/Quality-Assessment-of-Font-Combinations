import csv
import random

def judge_can_random_list(good_data_sequences, sub_list):
  flg = 0
  if (len(set(sub_list))==1):
    range_list = [sub_list[0] + i for i in range(-3, 4) if sub_list[0] + i != sub_list[0]]
    for num in range_list:
      lis = [num] * len(sub_list)
      if (lis in good_data_sequences):
        flg = 1
  if (flg==1): return False
  else: return True


good_data_path = "/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォントIDの研究/dataset/csv/test/test_no_font1kind_arranged_sorted.csv"
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

count = 0
c_len1 = 0
for sublist in good_data_sequences:
  flg = 0
  if (len(set(sublist))==1):
    c_len1 += 1
    range_list = [sublist[0] + i for i in range(-3, 4) if sublist[0] + i != sublist[0]]
    for num in range_list:
      lis = [num] * len(sublist)
      if (lis in good_data_sequences):
        flg = 1
  if (flg == 1):
    count += 1
print(count)

header = ['id', 'length', 'font']

# ランダムなフォントリストを作成
random_font_list = [] # 出力するランダムリスト
random_id_list = [] # 出力するランダムリスト
random_len_list = [] # 出力するランダムリスト
for num in range(len(good_data_sequences)):
  original_font_list = good_data_sequences[num] # 元のフォントリスト
  original_unique_font_list = list(set(good_data_sequences[num])) # 元のフォントリストの重複を除いたリスト
  random_font_sub_list = [] # 各組み合わせのランダム版を格納するフォントリスト

  if (judge_can_random_list(good_data_sequences, original_font_list)):
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
      
    while ((sorted(random_font_sub_list) in good_data_sequences) or (len(set(random_font_sub_list))==1)): # 良いデータと被っていないか確認 + フォントが1種類じゃないか
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


with open('/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォントIDの研究/dataset/csv/test/test_no_font1kind_random_arranged_sorted.csv', 'w') as f:
  writer = csv.writer(f)
  writer.writerow(header)
  for i, j, k in zip(random_id_list, random_len_list, random_font_list):
    body =  ["random"+i, j] + k
    writer.writerow(body)
f.close()



# プロの組み合わせの各リストを取り出す -> 重複を除いたリストを作成 -> このリストの長さでループ -> 各要素を+-3のうちからランダムに選んで元のリストのlist.count(number)で調べる
# -> その長さ分ランダム数字をコピー -> 事前に作成しておいたlist[プロの各リスト]に書いて、外側の配列に追加