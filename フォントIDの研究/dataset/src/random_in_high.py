import csv

# Download data
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


train_random_path = '/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォントIDの研究/dataset/csv/train/train_no_font1kind_random_arranged_sorted.csv'
val_random_path = '/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォントIDの研究/dataset/csv/validation/validation_no_font1kind_random_arranged_sorted.csv'
test_random_path = '/home/miyatamoe/ドキュメント/研究/1class_TF_and_NN/フォントIDの研究/dataset/csv/test/test_no_font1kind_random_arranged_sorted.csv'

random_train_id_list = []
random_train_len_list = []
random_train_font_list = []
random_train_label_list = []
with open(train_random_path) as f:
    reader = csv.reader(f)
    next(reader)
    for r in reader:
        random_train_id_list.append(r[0])
        random_train_len_list.append(r[1])
        random_train_font_list.append(r[2:])
        random_train_label_list.append(1)

random_val_id_list = []
random_val_len_list = []
random_val_font_list = []
random_val_label_list = []
with open(val_random_path) as f:
    reader = csv.reader(f)
    next(reader)
    for r in reader:
        random_val_id_list.append(r[0])
        random_val_len_list.append(r[1])
        random_val_font_list.append(r[2:])
        random_val_label_list.append(1)

random_test_id_list = []
random_test_len_list = []
random_test_font_list = []
random_test_label_list = []
with open(test_random_path) as f:
    reader = csv.reader(f)
    next(reader)
    for r in reader:
        random_test_id_list.append(r[0])
        random_test_len_list.append(r[1])
        random_test_font_list.append(r[2:])
        random_test_label_list.append(1)

# 'id', 'length', 'font'の文字、最初の意味のないラベルを削除
# id_list_all = []  
# len_list_all = []
# font_list_all = []

# id_list_all.extend(random_train_id_list)
# id_list_all.extend(random_val_id_list)
# id_list_all.extend(random_test_id_list)

# len_list_all.extend(random_train_len_list)
# len_list_all.extend(random_val_len_list)
# len_list_all.extend(random_test_len_list)

# font_list_all.extend(random_train_font_list)
# font_list_all.extend(random_val_font_list)
# font_list_all.extend(random_test_font_list)



count = sum(1 for sublist in random_test_font_list if sublist in font_list_all)

print(count)