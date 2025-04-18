import csv
######## Image のとき ###########
# CSVファイルのパス
Image_file_path = "/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/2つの研究の比較/data_score_id/Image_test_scores_and_fontlist_epoch70.csv"
# 閾値を設定（例: 0.0021）
Image_threshold_90per = 0.017138535156846046
# 一時的にデータを格納するリスト
Image_data = []
# CSVファイルを開いて処理
with open(Image_file_path, newline='', encoding='utf-8') as Image_csvfile:
    reader = csv.reader(Image_csvfile)
    header = next(reader)  # ヘッダーをスキップ

    for row in reader:
        index = row[0]  # ID（Index列）
        score = float(row[2])  # Score列（3列目）
        Image_data.append((score, index))  # (Score, Index) のタプルを保存
# Score の昇順（小さい順）でソート
Image_data.sort()
# 閾値でリストを分ける
Image_above_threshold = [index for score, index in Image_data if score > Image_threshold_90per]
Image_below_threshold = [index for score, index in Image_data if score <= Image_threshold_90per]

# 結果を出力
# print("画像の時のThreshold未満のID:", Image_below_threshold)
# print("\n画像の時のThreshold以上のID:", Image_above_threshold)
print("\n\nデータ数合計:", len(set(Image_above_threshold)) + len(set(Image_below_threshold)))



######## ID のとき ###########
# CSVファイルのパス
ID_file_path = "/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/2つの研究の比較/data_score_id/ID_test_scores_and_fontlist_epoch60.csv"
# 閾値を設定（例: 0.0021）
ID_threshold_90per = 0.006911535747349262
# 一時的にデータを格納するリスト
ID_data = []
# CSVファイルを開いて処理
with open(ID_file_path, newline='', encoding='utf-8') as ID_csvfile:
    reader = csv.reader(ID_csvfile)
    header = next(reader)  # ヘッダーをスキップ

    for row in reader:
        index = row[0]  # ID（Index列）
        score = float(row[2])  # Score列（3列目）
        ID_data.append((score, index))  # (Score, Index) のタプルを保存
# Score の昇順（小さい順）でソート
ID_data.sort()
# 閾値でリストを分ける
ID_above_threshold = [index for score, index in ID_data if score > ID_threshold_90per]
ID_below_threshold = [index for score, index in ID_data if score <= ID_threshold_90per]
# 結果を出力
# print("\nIDの時のThreshold未満のID:", ID_below_threshold)
# print("\nIDの時のThreshold以上のID:", ID_above_threshold)
print("\n\nデータ数合計:", len(set(ID_above_threshold)) + len(set(ID_below_threshold)))


# Image_data と ID_data を辞書に変換
Image_score_dict = {index: score for score, index in Image_data}
ID_score_dict = {index: score for score, index in ID_data}

# 確認
# print("Image_score_dict:", list(Image_score_dict.items())[:5])  # 最初の5件を表示
# print("ID_score_dict:", list(ID_score_dict.items())[:5])  # 最初の5件を表示


# ########## 画像のみで正解 ###############
only_Image_good_ids = list(set(Image_below_threshold) - set(ID_below_threshold))
# print("画像のみで正解:", only_Image_good_ids)
random_count = sum(1 for item in only_Image_good_ids if item.startswith('random'))
print("\nデータ数:", len(only_Image_good_ids), "中，ランダムデータ数:", random_count)
sorted_only_Image_good_ids = sorted(only_Image_good_ids, key=lambda id: Image_score_dict.get(id, float('inf')))
# print("\nソート済み only_Image_good_ids:", sorted_only_Image_good_ids)
sorted_only_Image_good_ids_tuple = sorted(
    [(id, Image_score_dict.get(id, float('inf'))) for id in only_Image_good_ids],
    key=lambda x: x[1]  # score（タプルの2番目の要素）でソート
)
# print("\nソート済みタプル only_Image_good_ids_tuple:", sorted_only_Image_good_ids_tuple)

# ########### IDのみで正解 #################
only_ID_good_ids = list(set(ID_below_threshold) - set(Image_below_threshold))
# print("IDのみで不正解:", only_ID_good_ids)
random_count = sum(1 for item in only_ID_good_ids if item.startswith('random'))
print("\nデータ数:", len(only_ID_good_ids), "中，ランダムデータ数:", random_count)
sorted_only_ID_good_ids_tuple = sorted(
    [(id, ID_score_dict.get(id, float('inf'))) for id in only_ID_good_ids],
    key=lambda x: x[1]  # score（タプルの2番目の要素）でソート
)
# print("\nソート済みタプル only_ID_good_ids_tuple:", sorted_only_ID_good_ids_tuple)

# ########### どちらも正解 ##################
both_good_ids = list(set(Image_below_threshold) & set(ID_below_threshold))
# print("どちらも正解:", both_good_ids)
random_count = sum(1 for item in both_good_ids if item.startswith('random'))
print("\nデータ数:", len(both_good_ids), "中，ランダムデータ数:", random_count)
sorted_both_good_ids_tuple = sorted(
    [(id, Image_score_dict.get(id, float('inf'))) for id in both_good_ids],
    key=lambda x: x[1]  # score（タプルの2番目の要素）でソート
)
# print("\nソート済みタプル both_good_ids_tuple:", sorted_both_good_ids_tuple)
# print("\nソート済みタプル データ数:", len(both_good_ids), len(sorted_both_good_ids_tuple))

# ########### どちらも不正解 ##################
both_bad_ids = list(set(Image_above_threshold) & set(ID_above_threshold))
# print("どちらも不正解:", both_bad_ids)
random_count = sum(1 for item in both_bad_ids if item.startswith('random'))
print("\nデータ数:", len(both_bad_ids), "中，ランダムデータ数:", random_count)
sorted_both_bad_ids_tuple = sorted(
    [(id, Image_score_dict.get(id, float('inf'))) for id in both_bad_ids],
    key=lambda x: x[1]  # score（タプルの2番目の要素）でソート
)
# print("\nソート済みタプル both_bad_ids_tuple:", sorted_both_bad_ids_tuple)
# print("\nソート済みタプル データ数:", len(both_bad_ids), len(sorted_both_bad_ids_tuple))
