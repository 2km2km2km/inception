import os

import pandas as pd
def get_txt_CX(path):

    #get the train_txt and the test_txt
    #书写CX的train_dataset.txt和test_dataset.txt
    #path为CX数据集路径
    class_max_num=10

    test_path = path+"测试集/test.csv"
    train_path = path+"训练集/train.csv"
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    train_txt = open('../train.txt', mode='w')
    test_txt = open('../test.txt', mode='w')
    #imgs=train["image_path"]
    for i in train.index:
        img_path=train["image_path"][i]
        label=train["labels"][i]
        train_txt.write(path+ "训练集/" + img_path + str(label))
        train_txt.write("\n")
    for i in test.index:
        img_path=test["image_path"][i]
        label=test["labels"][i]
        test_txt.write(path+ "测试集/" + img_path + str(label))
        test_txt.write("\n")

    print("done")


get_txt_CX("/media/xzl/Newsmy/flyai/data/")
