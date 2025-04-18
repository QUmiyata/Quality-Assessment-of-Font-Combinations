import csv
import matplotlib.pyplot as plt
import numpy as np

ids = []

# CSVファイルのパス
ID_csv_file = "/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/2つの研究の比較/data_score_id/ID_test_scores_and_fontlist_epoch130.csv"
# scoreを格納する辞書
ID_score_dict = {}
# CSVファイルを読み込む
with open(ID_csv_file, newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        id_name = row[0]
        id_score = float(row[2])  # スコアを浮動小数点数に変換
        ID_score_dict[id_name] = id_score
# 結果を確認
# print(len(ID_score_dict))

# CSVファイルのパス
Image_csv_file = "/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/2つの研究の比較/data_score_id/Image_test_scores_and_fontlist_epoch90.csv"
# scoreを格納する辞書
Image_score_dict = {}
# CSVファイルを読み込む
with open(Image_csv_file, newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        image_name = row[0]
        ids.append(image_name)
        image_score = float(row[2])  # スコアを浮動小数点数に変換
        Image_score_dict[image_name] = image_score

# 結果を確認
# print(len(Image_score_dict))
# print(len(ids))
# print(ids)


# 散布図用のデータを準備
x_red, y_red = [], []  # 'random' なし（赤）
x_blue, y_blue = [], []  # 'random' あり（青）

left_top = []  # 左上データ（低x, 高y）
right_bottom = []  # 右下データ（高x, 低y）

for image_id in ids:
    if image_id in ID_score_dict and image_id in Image_score_dict:
        x_val = Image_score_dict[image_id]
        y_val = ID_score_dict[image_id]

        # 左上のデータ（Image Scoreが小さく、ID Scoreが大きい）
        if x_val < 10**-2 and y_val > 10**-2:
            left_top.append((image_id, x_val, y_val))
        # 右下のデータ（Image Scoreが大きく、ID Scoreが小さい）
        if x_val > 10**0 and y_val < 2.2e-3:
            right_bottom.append((image_id, x_val, y_val))


        if image_id.startswith("random"):
            x_blue.append(x_val)
            y_blue.append(y_val)
        else:
            x_red.append(x_val)
            y_red.append(y_val)
print(len(x_red), len(y_red))
print(len(x_blue), len(y_blue))

# グラフを描画
plt.figure(figsize=(8, 6))
plt.scatter(x_red, y_red, color="red", alpha=0.5, s=3, label="Non-Random ID")
plt.scatter(x_blue, y_blue, color="blue", alpha=0.5, s=3, label="Random ID")

# plt.xlabel("Image Score")
# plt.ylabel("ID Score")
# plt.title("Scatter Plot of Image Score vs ID Score")
# plt.legend()
# plt.grid(True)

print(f"Left Top Data (低 x, 高 y): {len(left_top)}")
for i in left_top[:5]:  # 最初の5個だけ表示
    print(i)
print(f"Right Bottom Data (高 x, 低 y): {len(right_bottom)}")
# for i in right_bottom[:5]:  # 最初の5個だけ表示
for i in right_bottom: 
    print(i)
# 軸を対数スケールに設定
plt.xscale("log")  # X軸を対数スケール
plt.yscale("log")  # Y軸を対数スケール

# # Y軸の目盛を増やす
# y_ticks = np.logspace(-2.5, -1, num=5)  # 10^(-2.5) から 10^(-1) の範囲で5個の目盛
# plt.yticks(y_ticks, [f"{t:.1e}" for t in y_ticks])  
y_ticks = np.concatenate([
                            # np.arange(2.0, 10.0, 1.0) * 1e-3, 
                          np.arange(1.0, 2.0, 1.0) * 1e-2, 
                          np.arange(1.0, 2.0, 1.0) * 1e-1])
plt.yticks(y_ticks, [f"{t:.1e}" for t in y_ticks])

plt.xlabel("Image Score (log scale)")
plt.ylabel("ID Score (log scale)")
plt.title("Scatter Plot of Image Score vs ID Score (Log Scale)")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)  # 対数目盛用のグリッド

# グラフを表示
# plt.savefig("/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/2つの研究の比較/result/img/ICDAR_pro_rand_compare_new.png")
# SVG形式で保存
# plt.savefig("/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/2つの研究の比較/result/img/ICDAR_pro_rand_compare_log_new.svg", format="svg")
# plt.show()