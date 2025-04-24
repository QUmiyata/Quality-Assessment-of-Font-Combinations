import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import random
import csv
import pandas as pd
from collections import defaultdict
import pickle


option = 'nofont1kind_arranged'
tfepoch = 10000

def plot_LossCurve(epoch, loss_epochs):
    epochs_plot = range(1, (epoch+1) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_plot, loss_epochs, '-o', markersize=2, label='Train Loss', color='tab:blue')
    plt.title('Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    result_image_path = f'/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォント画像の研究/DeepSVDD/result/images/A-Zstack/Loss_Curve/{option}/tfepoch{tfepoch}/train_LossCurve_epoch{epoch+1}.png'
    plt.savefig(result_image_path)  # 画像として保存
    plt.close()

def plot_Features_train(epoch, features_epoch, c):
    # 特徴空間可視化
    features_epoch.append(c.tolist())
    data = np.array(features_epoch)  # numpy配列に変換

    # PCAで次元削減（512次元 -> 2次元）
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(data)
    colors = ['blue'] * (len(reduced_data) - 1) + ['orange']  # 最初の要素は赤、他は青
    plt.figure(figsize=(8, 6))
    # 2次元プロット
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=colors, s=2)
    plt.title("2D Visualization of 512D Vectors (PCA)  Train")
    tra_feature_PCA = f'/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォント画像の研究/DeepSVDD/result/images/A-Zstack/Features/train/{option}/tfepoch{tfepoch}/PCA/train_Feature_PCA_epoch{epoch+1}.png'
    plt.savefig(tra_feature_PCA)
    plt.close()

    # t-SNEによる次元削減（512次元 -> 2次元）
    tsne = TSNE(n_components=2, random_state=42)  # random_stateは再現性のため
    reduced_data = tsne.fit_transform(data)
    colors = ['blue'] * (len(reduced_data) - 1) + ['orange']  # 最初の要素はオレンジ、他は青
    plt.figure(figsize=(8, 6))
    # 2次元プロット
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=colors, s=2)
    plt.title("2D Visualization of 512D Vectors (t-SNE) Train")
    tra_feature_tSNE = f'/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォント画像の研究/DeepSVDD/result/images/A-Zstack/Features/train/{option}/tfepoch{tfepoch}/tSNE/train_Feature_tSNE_epoch{epoch+1}.png'
    plt.savefig(tra_feature_tSNE)
    plt.close()

def plot_Features_test(test_features, test_font_list_all, label_list_all, n_epochs, c):
    n_epochs = 130 #test時のみ

    test_features_plot = [] 
    test_features_plot.extend(test_features)
    # プロットの色を指定
    colors = []
    for lebel in label_list_all:
        if (lebel == 0):
            colors.append('blue')
        else:
            colors.append('green')
    colors.append('orange') 

    # 特徴空間可視化
    test_features_plot.append(c.tolist())
    data = np.array(test_features_plot)  # numpy配列に変換
    # PCAで次元削減（512次元 -> 2次元）
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(data)
    plt.figure(figsize=(8, 6))
    # 2次元プロット
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=colors, s=2)
    plt.title("2D Visualization of 512D Vectors (PCA)  Test")
    tes_feature_PCA = f'/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォント画像の研究/DeepSVDD/result/images/A-Zstack/Features/test/{option}/tfepoch{tfepoch}/PCA/test_Feature_PCA_epoch{n_epochs}.png'
    plt.savefig(tes_feature_PCA) 
    plt.close()

    # label == 1 のデータに対してランダムに10個を選んで番号を表示
    label_1_indices_PCA = [i for i, label in enumerate(label_list_all) if label == 1]
    random_indices_PCA = random.sample(label_1_indices_PCA, min(10, len(label_1_indices_PCA)))  # ランダムに最大10個選ぶ
    # 追加の画像を作成して保存
    plt.figure(figsize=(8, 6))
    # 2次元プロット
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=colors, s=2)
    # 選ばれたインデックスに番号を付けて表示
    for i in random_indices_PCA:
        plt.text(reduced_data[i, 0], reduced_data[i, 1], str(i), color='red', fontsize=12)
    plt.title("2D Visualization of 512D Vectors (PCA) Test(+number)")
    test_feature_PCA_plusNumber = f'/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォント画像の研究/DeepSVDD/result/images/A-Zstack/Features/test/{option}/tfepoch{tfepoch}/PCA/test_Feature_PCA_epoch{n_epochs}_plusNumber.png'
    plt.savefig(test_feature_PCA_plusNumber)
    plt.close()

    # t-SNEによる次元削減（512次元 -> 2次元）
    tsne = TSNE(n_components=2, random_state=42)  # random_stateは再現性のため
    reduced_data = tsne.fit_transform(data)
    plt.figure(figsize=(8, 6))
    # 2次元プロット
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=colors, s=2)
    plt.title("2D Visualization of 512D Vectors (t-SNE) Test")
    test_feature_tSNE = f'/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォント画像の研究/DeepSVDD/result/images/A-Zstack/Features/test/{option}/tfepoch{tfepoch}/tSNE/test_Feature_tSNE_epoch{n_epochs}.png'
    plt.savefig(test_feature_tSNE)
    plt.close()

    # label == 1 のデータに対してランダムに10個を選んで番号を表示
    label_1_indices_tSNE = [i for i, label in enumerate(label_list_all) if label == 1]
    random_indices_tSNE = random.sample(label_1_indices_tSNE, min(10, len(label_1_indices_tSNE)))  # ランダムに最大10個選ぶ
    # 追加の画像を作成して保存
    plt.figure(figsize=(8, 6))
    # 2次元プロット
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=colors, s=2)
    # 選ばれたインデックスに番号を付けて表示
    for i in random_indices_tSNE:
        plt.text(reduced_data[i, 0], reduced_data[i, 1], str(i), color='red', fontsize=12)
    plt.title("2D Visualization of 512D Vectors (t-SNE) Test(+number)")
    test_feature_tSNE_plusNumber = f'/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォント画像の研究/DeepSVDD/result/images/A-Zstack/Features/test/{option}/tfepoch{tfepoch}/tSNE/test_Feature_tSNE_epoch{n_epochs}_plusNumber.png'
    plt.savefig(test_feature_tSNE_plusNumber)
    plt.close()

    font_entries = []
    for i in label_1_indices_tSNE:
        # インデックスに対応するフォント情報を取得
        font_entries.append(f"Index: {i}, Font: {test_font_list_all[i]}")
    # フォントリストをテキストファイルに出力
    with open(f'/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォント画像の研究/DeepSVDD/result/images/A-Zstack/Features/test/{option}/tfepoch{tfepoch}/tSNE/test_Feature_tSNE_epoch{n_epochs}_plusNumber_font_info.txt', 'w') as f:
        for entry in font_entries:
            f.write(entry + '\n')

def create_fontlist_txt_MSE(test_scores, n_epochs):
    n_epochs = 130 #test時のみ
    
    # Load the CSV file and create a dictionary to map idx to font values
    csv_file_path = f'/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォント画像の研究/dataset/csv/test/for_text/{option}/test_fonts_{option}_all.csv'  # 実際のCSVファイルのパスを指定
    idx_to_font = {}

    # Read CSV and store font values for each idx
    with open(csv_file_path, mode='r') as file:
        reader = csv.DictReader(file)
        # next(reader)
        for row in reader:
            idx = row['id']
            fonts = []
            for key in row:
                if key not in ['id', 'length']:
                    # Check if the value is a list and flatten it
                    value = row[key]
                    if isinstance(value, list):  # If the value is a list, extend it
                        fonts.extend(value)
                    else:  # Otherwise, add the single value
                        fonts.append(value)
            idx_to_font[idx] = fonts  # Store the fonts for each idx

    # Sort self.test_scores by the scores (third element of each tuple)
    sorted_test_scores = sorted(test_scores, key=lambda x: x[2])  # Sort by score (x[2])

    # Prepare the output path for the txt file
    output_txt_path = f'/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォント画像の研究/DeepSVDD/result/images/A-Zstack/Features/test/{option}/tfepoch{tfepoch}/text/test_scores_epoch{n_epochs}.txt'
    # Write the sorted results to the txt file
    with open(output_txt_path, 'w') as f:
        f.write("Index\tLabel\tScore\tFont\n")  # Write header
        for idx, label, score in sorted_test_scores:
            # Get the font associated with the idx
            fonts = idx_to_font.get(idx, [])
            # Join fonts into a single string, separating by commas if there are multiple fonts
            font_str = ','.join(fonts) if fonts else 'N/A'  # Use 'N/A' if no font found for idx
            f.write(f"{idx}\t{label}\t{score:.6f}\t{font_str}\n")  # Write each entry with the format Index\tLabel\tScore\tFont

    # Prepare the output path for the txt file
    output_csv_path = f'/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォント画像の研究/dataset/csv/test/scores/{option}/tfepoch{tfepoch}/test_scores_and_fontlist_epoch{n_epochs}.csv'
    # Write the sorted results to the txt file
    with open(output_csv_path, 'w') as f:
        f.write("Index,Label,Score,Font\n")  # Write header
        for idx, label, score in sorted_test_scores:
            # Get the font associated with the idx
            fonts = idx_to_font.get(idx, [])
            # Join fonts into a single string, separating by commas if there are multiple fonts
            font_str = ','.join(fonts) if fonts else 'N/A'  # Use 'N/A' if no font found for idx
            f.write(f"{idx},{label},{score:.6f},{font_str}\n")  # Write each entry with the format Index\tLabel\tScore\tFont
    
    output_KDE_csv_path = f'/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォント画像の研究/dataset/csv/test/scores/{option}/tfepoch{tfepoch}/KDE/test_scores_and_label_epoch{n_epochs}_KDE.csv'
    # Write the sorted results to the txt file
    with open(output_KDE_csv_path, 'w') as f:
        f.write("Index,Label,Score\n")  # Write header
        for idx, label, score in sorted_test_scores:
            f.write(f"{idx},{label},{score:.6f}\n")  # Write each entry with the format Index\tLabel\tScore\tFont


def plot_before_train_cls_tokens(before_train_cls_token, c):
    before_train_cls_token_plot = before_train_cls_token
    # 特徴空間可視化
    before_train_cls_token_plot.append(c.tolist())
    data = np.array(before_train_cls_token_plot)  # numpy配列に変換

    # PCAで次元削減（512次元 -> 2次元）
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(data)
    colors = ['blue'] * (len(reduced_data) - 1) + ['orange']  # 最初の要素は赤、他は青
    plt.figure(figsize=(8, 6))
    # 2次元プロット
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=colors, s=2)
    plt.title("2D Visualization of 512D Vectors (PCA)  Train")
    tra_feature_PCA = f'/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォント画像の研究/DeepSVDD/result/images/A-Zstack/Features/train/{option}/tfepoch{tfepoch}/PCA/before_train_cls_tokens_PCA.png'
    plt.savefig(tra_feature_PCA)
    plt.close()

    # t-SNEによる次元削減（512次元 -> 2次元）
    tsne = TSNE(n_components=2, random_state=42)  # random_stateは再現性のため
    reduced_data = tsne.fit_transform(data)
    colors = ['blue'] * (len(reduced_data) - 1) + ['orange']  # 最初の要素はオレンジ、他は青
    plt.figure(figsize=(8, 6))
    # 2次元プロット
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=colors, s=2)
    plt.title("2D Visualization of 512D Vectors (t-SNE) Train")
    tra_feature_tSNE = f'/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォント画像の研究/DeepSVDD/result/images/A-Zstack/Features/train/{option}/tfepoch{tfepoch}/tSNE/before_train_cls_tokens_tSNE.png'
    plt.savefig(tra_feature_tSNE)
    plt.close()

def plot_before_test_cls_tokens(before_train_cls_token, label_list_all, c):
    before_train_cls_token_plot = before_train_cls_token
    # プロットの色を指定
    colors = []
    for lebel in label_list_all:
        if (lebel == 0):
            colors.append('blue')
        else:
            colors.append('green')
    # colors.append('orange') 

    sizes = [2] * len(colors)  # すべての点をサイズ2に設定
    # sizes[-1] = 50  # 最後の点（オレンジの点）のサイズを50に設定

    # 特徴空間可視化
    # before_train_cls_token_plot.append(c.tolist())
    data = np.array(before_train_cls_token_plot)  # numpy配列に変換
    # PCAで次元削減（512次元 -> 2次元）
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(data)

    plt.figure(figsize=(8, 6))
    # 2次元プロット
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=colors, s=sizes)
    plt.title("2D Visualization of 512D Vectors (PCA)  Test")
    tes_feature_PCA = f'/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォント画像の研究/DeepSVDD/result/images/A-Zstack/Features/test/{option}/tfepoch{tfepoch}/PCA/before_test_cls_tokens_PCA_noC.png'
    plt.savefig(tes_feature_PCA) 
    plt.close()

    # t-SNEによる次元削減（512次元 -> 2次元）
    tsne = TSNE(n_components=2, random_state=42)  # random_stateは再現性のため
    reduced_data = tsne.fit_transform(data)
    plt.figure(figsize=(8, 6))
    # 2次元プロット
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=colors, s=sizes)
    plt.title("2D Visualization of 512D Vectors (t-SNE) Test")
    test_feature_tSNE = f'/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォント画像の研究/DeepSVDD/result/images/A-Zstack/Features/test/{option}/tfepoch{tfepoch}/tSNE/before_test_cls_tokens_tSNE_noC.png'
    plt.savefig(test_feature_tSNE)
    plt.close()



def create_histogram_per_scores():
    mlpepoch = 130 #実行時に変更
    file_path = f'/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォント画像の研究/dataset/csv/test/scores/nofont1kind_arranged/tfepoch10000/test_scores_and_fontlist_epoch{mlpepoch}.csv'
    # ScoreごとのLabel 0とLabel 1の出現回数をカウントする辞書を初期化
    score_counts = defaultdict(lambda: {'0': 0, '1': 0})
    # CSVファイルを読み込む
    with open(file_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        # 各行を処理
        for row in reader:
            label = row['Label']  # 'Label'列の値
            score = float(row['Score'])  # 'Score'列の値（floatに変換）
            # ScoreごとのLabel 0とLabel 1のカウントを更新
            score_counts[score][label] += 1
    # プロット用にデータを準備
    scores = sorted(score_counts.keys())  # Scoreを昇順にソート
    label_0_counts = [score_counts[score]['0'] for score in scores]  # Label 0のカウント
    label_1_counts = [score_counts[score]['1'] for score in scores]  # Label 1のカウント
    # 最大値を取得
    max_count = max(max(label_0_counts), max(label_1_counts))
    # ヒストグラムをプロット
    plt.figure(figsize=(10, 6))
    # Label 0とLabel 1のカウントをバーで表示
    width = 0.001  # 幅を調整
    plt.bar(scores, label_0_counts, width=width, label="Label 0", align='center', alpha=0.7)
    plt.bar(scores, label_1_counts, width=width, label="Label 1", align='edge', alpha=0.7)
    # 軸ラベルとタイトルを追加
    plt.xlabel('Score')
    plt.ylabel('Count')
    plt.title('Histogram of Scores per Label')
    plt.legend()
    # 縦軸の最大値をlabelのカウントの最大値に設定
    plt.ylim(0, max_count * 1.1)  # 最大値に少し余裕を持たせる（例えば10%増）
    # レイアウトを自動調整
    plt.tight_layout()
    # プロットを表示
    # plt.show()
    test_histogram_per_scores = f'/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォント画像の研究/DeepSVDD/result/images/A-Zstack/Features/test/{option}/tfepoch{tfepoch}/histogram/test_histogram_per_scores_epoch{mlpepoch}.png'
    plt.savefig(test_histogram_per_scores)
    plt.close()



# def check():
#     test_features = np.random.rand(5, 512).tolist()
#     test_font_list_all = [[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7]]
#     label_list_all = [0, 0, 1, 1, 1]
#     n_epochs = 00
#     c = np.mean(test_features, axis=0)
#     plot_Features_test(test_features, test_font_list_all, label_list_all, n_epochs, c)


# create_histogram_per_scores()


def plot_Features_test_and_train(test_features, test_font_list_all, label_list_all, n_epochs, c):
    n_epochs = 130 #test時のみ

    # 保存したファイルパス
    train_features_load_path = f'/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォント画像の研究/DeepSVDD/result/features/train/{option}/tfepoch{tfepoch}/train_features_epoch{n_epochs}.pkl'
    # ピクルファイルを読み込む
    with open(train_features_load_path, 'rb') as f:
        train_features = pickle.load(f)
    # 読み込んだデータを確認
    # print(train_features)

    # プロットの色を指定
    colors = []
    for i in range(len(train_features)):
        colors.append('black')
    for lebel in label_list_all:
        if (lebel == 0):
            colors.append('blue')
        else:
            colors.append('green')
    colors.append('orange') 

    # 特徴空間可視化
    features = []
    features.extend(train_features)
    features.extend(test_features)
    features.append(c.tolist())
    data = np.array(features)  # numpy配列に変換
    # PCAで次元削減（512次元 -> 2次元）
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(data)
    plt.figure(figsize=(8, 6))
    # 2次元プロット
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=colors, s=2)
    plt.title("2D Visualization of 512D Vectors (PCA)  Train & Test")
    tes_feature_PCA = f'/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォント画像の研究/DeepSVDD/result/images/A-Zstack/Features/test/{option}/tfepoch{tfepoch}/PCA/test_and_train_Feature_PCA_epoch{n_epochs}.png'
    plt.savefig(tes_feature_PCA) 
    plt.close()

    # t-SNEによる次元削減（512次元 -> 2次元）
    tsne = TSNE(n_components=2, random_state=42)  # random_stateは再現性のため
    reduced_data = tsne.fit_transform(data)
    plt.figure(figsize=(8, 6))
    # 2次元プロット
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=colors, s=2)
    plt.title("2D Visualization of 512D Vectors (t-SNE) Train & Test")
    test_feature_tSNE = f'/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォント画像の研究/DeepSVDD/result/images/A-Zstack/Features/test/{option}/tfepoch{tfepoch}/tSNE/test_and_train_Feature_tSNE_epoch{n_epochs}.png'
    plt.savefig(test_feature_tSNE)
    plt.close()

# def plot_Features_test_and_train(test_features, test_font_list_all, label_list_all, n_epochs, c):
#     n_epochs = 50  # test時のみ

#     # 保存したファイルパス
#     train_features_load_path = f'/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォント画像の研究/DeepSVDD/result/features/train/{option}/tfepoch{tfepoch}/train_features_epoch{n_epochs}.pkl'
    
#     # ピクルファイルを読み込む
#     with open(train_features_load_path, 'rb') as f:
#         train_features = pickle.load(f)
    
#     # プロットの色を指定
#     colors = []
#     for i in range(len(train_features)):
#         colors.append('black')
#     for label in label_list_all:
#         if label == 0:
#             colors.append('blue')
#         else:
#             colors.append('green')
#     colors.append('orange')

#     # 特徴空間可視化
#     features = []
#     features.extend(train_features)
#     features.extend(test_features)
#     features.append(c.tolist())
#     data = np.array(features)  # numpy配列に変換

#     # PCAで次元削減（512次元 -> 2次元）
#     pca = PCA(n_components=2)
#     reduced_data = pca.fit_transform(data)

#     # PCAプロット
#     plt.figure(figsize=(112, 84))
#     scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=colors, s=2)
#     plt.title("2D Visualization of 512D Vectors (PCA)  Train & Test")

#     # 目盛りの間隔を0.025に設定
#     x_min, x_max = np.min(reduced_data[:, 0]), np.max(reduced_data[:, 0])
#     y_min, y_max = np.min(reduced_data[:, 1]), np.max(reduced_data[:, 1])

#     # X軸、Y軸の範囲に余裕を持たせる（例えば5%）
#     x_margin = (x_max - x_min) * 0.05
#     y_margin = (y_max - y_min) * 0.05

#     # 新しい範囲を設定
#     x_min -= x_margin
#     x_max += x_margin
#     y_min -= y_margin
#     y_max += y_margin

#     # X軸とY軸の目盛りを0.025の間隔で設定
#     x_ticks = np.arange(np.floor(x_min / 0.025) * 0.025, np.ceil(x_max / 0.025) * 0.025, 0.025)
#     y_ticks = np.arange(np.floor(y_min / 0.025) * 0.025, np.ceil(y_max / 0.025) * 0.025, 0.025)

#     plt.xticks(x_ticks)
#     plt.yticks(y_ticks)

#     # 保存するパス
#     tes_feature_PCA = f'/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォント画像の研究/DeepSVDD/result/images/A-Zstack/Features/test/{option}/tfepoch{tfepoch}/PCA/test_and_train_Feature_PCA_epoch{n_epochs}.png'
#     plt.savefig(tes_feature_PCA) 
#     plt.close()

#     # t-SNEによる次元削減（512次元 -> 2次元）
#     tsne = TSNE(n_components=2, random_state=42)  # random_stateは再現性のため
#     reduced_data = tsne.fit_transform(data)

#     # t-SNEプロット
#     plt.figure(figsize=(112, 84))
#     scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=colors, s=2)
#     plt.title("2D Visualization of 512D Vectors (t-SNE) Train & Test")

#     # 保存するパス
#     test_feature_tSNE = f'/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォント画像の研究/DeepSVDD/result/images/A-Zstack/Features/test/{option}/tfepoch{tfepoch}/tSNE/test_and_train_Feature_tSNE_epoch{n_epochs}.png'
#     plt.savefig(test_feature_tSNE)
#     plt.close()
