import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import pickle

def KDE_nomal():
    # データをCSVから読み込む
    tfepoch = 10000
    mlpepoch = 90
    data = pd.read_csv(f'/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォント画像の研究/dataset/csv/test/scores/nofont1kind_arranged/tfepoch{tfepoch}/KDE/test_scores_and_label_epoch{mlpepoch}_KDE.csv')
    # data = pd.read_csv(f'/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォント画像の研究/dataset/csv/test/scores/nofont1kind_arranged/miss/tfepoch{tfepoch}/KDE/test_scores_and_label_epoch{mlpepoch}_KDE.csv')

    # 'Label' 列をカテゴリカルデータとして扱う
    data['Label'] = data['Label'].astype(str)

    # グラフの設定
    plt.figure(figsize=(9, 3))

    # 'Label'ごとに色分けしてKDEをプロット
    sns.kdeplot(data=data[data['Label'] == '0']['Score'], fill=True, color='blue', label='Label 0 (crello)', bw_method=0.1)
    sns.kdeplot(data=data[data['Label'] == '1']['Score'], fill=True, color='red', label='Label 1 (random)', bw_method=0.1)

    # グラフのタイトルとラベルを設定
    # plt.title('Kernel Density Estimation (KDE) for Score by Label', fontsize=20)
    # plt.title('Kernel Density Estimation (KDE) for Score by Label', fontsize=14)
    # plt.xlabel('Score', fontsize=18)
    plt.xlabel('Score', fontsize=12)
    # plt.ylabel('Density', fontsize=18)
    plt.ylabel('Density', fontsize=12)

    # ラベルの表示
    # plt.legend(title='Label', fontsize=18)
    plt.legend(title='Label', fontsize=12)

    # 目盛の数字を大きくする
    # plt.tick_params(axis='both', labelsize=16)  # x軸とy軸の目盛りを16サイズに設定

    # 表示
    plt.tight_layout()

    # test_KDE_path = f'/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォント画像の研究/DeepSVDD/result/images/A-Zstack/Combinations_from_score/nofont1kind_arranged/tfepoch{tfepoch}/epoch{mlpepoch}/KDE/test_KDE_epoch{mlpepoch}.png'
    # # test_KDE_path = f'/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォント画像の研究/DeepSVDD/result/images/A-Zstack/Combinations_from_score/nofont1kind_arranged/miss/tfepoch{tfepoch}/epoch{mlpepoch}/KDE/test_KDE_epoch{mlpepoch}.png'
    # plt.savefig(test_KDE_path)
    test_KDE_path_svg= f'/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォント画像の研究/DeepSVDD/result/images/A-Zstack/Combinations_from_score/nofont1kind_arranged/tfepoch{tfepoch}/epoch{mlpepoch}/KDE/test_KDE_epoch{mlpepoch}.svg'
    plt.savefig(test_KDE_path_svg)

    # グラフを表示
    # plt.show()

    plt.close()



def KDE_1to4():
    # データをCSVから読み込む
    tfepoch = 10000
    mlpepoch = 40
    data = pd.read_csv(f'/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォント画像の研究/dataset/csv/test/scores/nofont1kind_arranged/tfepoch{tfepoch}/KDE/test_scores_and_label_epoch{mlpepoch}_KDE.csv')

    # data = pd.read_csv(f'/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォント画像の研究/dataset/csv/test/scores/nofont1kind_arranged/miss/tfepoch{tfepoch}/KDE/test_scores_and_label_epoch{mlpepoch}_KDE.csv')

    # 'Label' 列をカテゴリカルデータとして扱う
    data['Label'] = data['Label'].astype(str)

    # Scoreが1〜4の範囲にフィルタリング
    filtered_data = data[(data['Score'] >= 1) & (data['Score'] <= 4)]

    # グラフの設定
    plt.figure(figsize=(10, 6))

    # 'Label'ごとに色分けしてKDEをプロット
    sns.kdeplot(data=filtered_data[filtered_data['Label'] == '0']['Score'], fill=True, color='blue', label='Label 0 (crello)', bw_method=0.2)
    sns.kdeplot(data=filtered_data[filtered_data['Label'] == '1']['Score'], fill=True, color='red', label='Label 1 (random)', bw_method=0.2)

    # グラフのタイトルとラベルを設定
    plt.title('Kernel Density Estimation (KDE) for Score by Label', fontsize=14)
    plt.xlabel('Score', fontsize=12)
    plt.ylabel('Density', fontsize=12)

    # ラベルの表示
    plt.legend(title='Label', fontsize=12)

    # 表示
    plt.tight_layout()

    # 保存パス設定
    # test_KDE_path = f'/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォント画像の研究/DeepSVDD/result/images/A-Zstack/Combinations_from_score/nofont1kind_arranged/miss/tfepoch{tfepoch}/epoch{mlpepoch}/KDE/test_KDE_epoch{mlpepoch}_1to4.png'
    test_KDE_path = f'/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォント画像の研究/DeepSVDD/result/images/A-Zstack/Combinations_from_score/nofont1kind_arranged/tfepoch{tfepoch}/epoch{mlpepoch}/KDE/test_KDE_epoch{mlpepoch}_1to4.png'
    plt.savefig(test_KDE_path)

    # グラフを表示
    # plt.show()

    plt.close()



def KDE_train():
    model_path = ''
    model_dict = torch.load(model_path)
    c = model_dict['c']

    train_feature_path = ''
    with open(train_feature_path, 'rb') as f:
        features_epoch = pickle.load(f)
    outputs = torch.tensor(features_epoch, device='cuda')

    scores = torch.sum((outputs - c) ** 2, dim=1)




KDE_nomal()
# KDE_1to4()