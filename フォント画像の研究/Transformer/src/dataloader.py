# 文字Aの周りを白で埋める
# 白で埋めたら縦横の長い方向にもう１つの辺の大きさを合わせるようにパディングする。（正方形にする）
# この正方形を全ての画像で同じ大きさ（縦横の長さ）に合わせる
# Csvファイルを読み込んで、font listの数字に対応する画像をその都度フォルダから探す。
# 全てのfont listの数字を１つに繋げて、写真も１つに繋げる（リストに画像を要素として追加していく。この時、配列に入れても普通の配列とは次元が異なるだけなので何も問題なし）
# Lenlistを見ながら全て繋げた画像のリストを区切って入力していく

import torch
import torch.nn as nn
import csv
from tqdm import tqdm
import random
import logging

logging.basicConfig(
    # filename='/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォント画像の研究/Transformer/result/A-Zstack/log/train_original_log.log',  # 単一フォント除外なし
    filename='/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォント画像の研究/Transformer/result/A-Zstack/log/train_nofont1kind_arranged_log.log',  # 単一フォント除外あり・フォントIDの並び替えあり
    level=logging.INFO,  # ログレベルをINFOに設定
    format='%(asctime)s - %(levelname)s - %(message)s'  # ログメッセージのフォーマット
)


class Dataloader_TF():
    def __init__(self, data_path, random_data_path, batch_size, is_training, true_label=0) -> None:
        self.data_path = data_path
        self.random_data_path = random_data_path
        self.batch_size = batch_size
        self.is_training = is_training
        self.true_label = true_label
        self.id_list = []
        self.len_list = []
        self.font_list = []
        self.label_list = []
        self.idx_list = []
        self.data_num = 0
        print(f"< {self.data_path} >")
        logging.info(f"< {self.data_path} >")
        font_list, len_list, id_list, label_list = self._load(self.data_path, self.true_label)
        print(f"< {self.random_data_path} >")
        logging.info(f"< {self.random_data_path} >")
        r_font_list, r_len_list, r_id_list, r_label_list = self._load(self.random_data_path, 1 - self.true_label)
        self.font_list = font_list + r_font_list
        self.len_list = len_list + r_len_list
        self.id_list = id_list + r_id_list
        self.label_list = label_list + r_label_list
        self._arrange(self.font_list, self.len_list, self.id_list, self.label_list)

    def _load(self, path, label):
        # Download data
        id_list = []
        len_list = []
        font_list = []
        label_list = []
        with open(path) as f:
            reader = csv.reader(f)
            for r in tqdm(reader, desc="[Download data]"):
                id_list.append(r[0])
                len_list.append(r[1])
                font_list.append(r[2:])
                label_list.append(label)

        # 'id', 'length', 'font'の文字、最初の意味のないラベルを削除
        id_list = id_list[1:]   
        len_list = len_list[1:]
        font_list = font_list[1:]
        label_list = label_list[1:]

        # Exchange Font data and length data into integer, because Font data is string
        f_list = []
        l_list = []
        for f, l in tqdm(zip(font_list, len_list), desc="[str to int]"):
            f_tmp = [int(i) for i in f]
            f_list.append(f_tmp)
            l_list.append(int(l))
        font_list = f_list
        len_list = l_list

        return font_list, len_list, id_list, label_list

    def _arrange(self, font_list, len_list, id_list, label_list):
        # Checking the number of data
        font_num = len(font_list)
        len_num = len(len_list)
        id_num = len(id_list)
        label_num = len(label_list)
        assert font_num == len_num == id_num == label_num, "Number of font data, length data and id data must be the same."
        self.data_num = label_num
        print(f"Number of total data: {self.data_num}")
        logging.info(f"Number of total data: {self.data_num}")
        self.idx_list = [i for i in range(self.data_num)] # 0〜(データ数-1)までの数字を格納
    
    def pad_data(self, font_list, max_len):     # font_listに含まれる各fontsリストの長さがmax_len未満の場合、不足する部分をゼロで埋める
        padded_fonts = []
        for fonts in font_list:
            padded_fonts.append(fonts + [0] * (max_len - len(fonts)))
        return padded_fonts

    def get_data(self, idx):
        # Get single data refer to idx
        return self.font_list[idx], self.len_list[idx], self.id_list[idx], self.label_list[idx]

    def shuffle_idx(self):
        # Shuffle idx
        random.shuffle(self.idx_list)
    
    def yield_batched_data(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        if self.is_training:
            self.shuffle_idx()
        
        # bn = self.data_num//batch_size  # バッチサイズを１つのまとまりとした時のまとまりの個数
        bn = (self.data_num + batch_size - 1) // batch_size
        max_len = max(self.len_list)

        for b in range(bn):  # まとまりごとに実行
            font_list = []
            len_list = []
            id_list = []
            label_list = []

            # 最後のバッチが不完全でもデータを取得
            start_idx = b * batch_size
            end_idx = min((b + 1) * batch_size, self.data_num)  # 最後のバッチで余りがあればそれを含める
        
            # for idx in self.idx_list[b*batch_size: b*batch_size + batch_size]:  # まとまり（バッチサイズ個）ずつリストからとる
            for idx in self.idx_list[start_idx:end_idx]:  # 余りも含めたバッチ処理
                font, length, id, label = self.get_data(idx)
                font_list.extend(font)
                len_list.append(length)
                id_list.append(id)
                label_list.append(label)
            # font_list = self.pad_data(font_list, max_len)     ##train
            # font_list = self.pad_data(font_list, 44)            ##test
            yield font_list, len_list, id_list, label_list

# def checker():
#     path = "/home/miyatamoe/ドキュメント/研究（支部大会2024）/SIBU_font_image/datasets/csv/train/train_fonts_no_0.csv"
#     random_path = "/home/miyatamoe/ドキュメント/研究（支部大会2024）/SIBU_font_image/datasets/csv/train/train_fonts_random_no_0.csv"

#     dataloader = Dataloader_TF(path, random_path, is_training=True, batch_size=512)

#     #使い方
#     for e in range(10):
#         for font_list, len_list, id_list, label_list in dataloader.yield_batched_data():
#             print(f"font: {font_list[0]}, len: {len_list[0]}, id_list: {id_list[0]}, label: {label_list[0]}")

#     print("check")

# if __name__ == "__main__":
#     checker()
