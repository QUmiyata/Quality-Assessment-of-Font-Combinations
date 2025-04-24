import torch
import numpy as np
import pickle

tfepoch = 10000
option = 'nofont1kind_arranged'
mlpepoch = 50

def calculate_trainfeature_border(c, device, percentile=0.9):
    train_features_load_path = f'/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォントIDの研究/DeepSVDD/result/features/train/{option}/train_features_epoch{mlpepoch}.pkl'
    with open(train_features_load_path, 'rb') as f:
        train_features = pickle.load(f)
    
    train_features = torch.tensor(train_features, device=device) 

    dist = torch.sum((train_features - c) ** 2, dim=1)

    # データを昇順に並べる
    sorted_data = sorted(dist)
    
    # 90%の位置を求める
    percentile_index = int(percentile * len(sorted_data))  # 90%の位置を計算（インデックスに丸める）
    
    # 90%の境界値を取得
    boundary_value = sorted_data[percentile_index]

    print(f"{percentile*100}%の境界値は: {boundary_value}")

    return boundary_value

