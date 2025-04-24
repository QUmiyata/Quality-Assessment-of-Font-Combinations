import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# データをCSVから読み込む
# tfepoch = 50
mlpepoch = 130
data = pd.read_csv(f'/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォントIDの研究/dataset/csv/test/scores/nofont1kind_arranged/KDE/test_scores_and_label_epoch{mlpepoch}_KDE.csv')

# 'Label' 列をカテゴリカルデータとして扱う
data['Label'] = data['Label'].astype(str)

# グラフの設定
plt.figure(figsize=(9, 3))

# 'Label'ごとに色分けしてKDEをプロット
sns.kdeplot(data=data[data['Label'] == '0']['Score'], fill=True, color='blue', label='Label 0 (crello)', bw_method=0.2)
sns.kdeplot(data=data[data['Label'] == '1']['Score'], fill=True, color='red', label='Label 1 (random)', bw_method=0.2)

# グラフのタイトルとラベルを設定
# plt.title('Kernel Density Estimation (KDE) for Score by Label', fontsize=14)
plt.xlabel('Score', fontsize=12)
plt.ylabel('Density', fontsize=12)

# ラベルの表示
plt.legend(title='Label', fontsize=12)

# 表示
plt.tight_layout()

# test_KDE_path = f'/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォントIDの研究/DeepSVDD/result/images/Combinations_from_score/nofont1kind_arranged/epoch{mlpepoch}/KDE/test_KDE_epoch{mlpepoch}.png'
# plt.savefig(test_KDE_path)
test_KDE_path_svg = f'/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォントIDの研究/DeepSVDD/result/images/Combinations_from_score/nofont1kind_arranged/epoch{mlpepoch}/KDE/test_KDE_epoch{mlpepoch}_new.svg'
plt.savefig(test_KDE_path_svg, format='svg')

# グラフを表示
# plt.show()

plt.close()
