import os

import pandas as pd
def get_txt_CX(path):

    #get the train_txt and the test_txt
    #书写CX的train_dataset.txt和test_dataset.txt
    #path为CX数据集路径
    class_max_num=10

    pre_path = path+"评估数据集/upload.csv"
    pre = pd.read_csv(pre_path)

    pre_txt = open('../pre.txt', mode='w')
    #imgs=train["image_path"]
    for i in pre.index:
        img_path=pre["image_path"][i]
        label=0
        pre_txt.write(path+ "评估数据集/" + img_path + str(label))
        pre_txt.write("\n")

    print("done")


get_txt_CX("/media/xzl/Newsmy/flyai/data/")
