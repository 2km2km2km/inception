from __future__ import division

from models import *
from utils.datasets import *

import tqdm
import pandas as pd

import torch
from torch.utils.data import DataLoader

from torch.autograd import Variable


def predict(model,path,img_size, batch_size):
    model.eval()
    pre_path = "/media/xzl/Newsmy/flyai/data/评估数据集/upload.csv"
    pre = pd.read_csv(pre_path)
    # Get dataloader
    dataset = CASIA_Dataset(path, img_size=img_size)
    #print(len(dataset))
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    total_acc=0

    for batch_i, (_, imgs, y_reals) in enumerate(tqdm.tqdm(dataloader)):
        #print(_[0].replace("/media/xzl/Newsmy/flyai/data/评估数据集/images/",""))

        y_reals = Variable(y_reals.type(Tensor), requires_grad=False)

        imgs = Variable(imgs.type(Tensor), requires_grad=False)

        with torch.no_grad():
            y_hats = model(imgs)
        _, prediction = torch.max(y_hats.data, 1)
        #print(prediction)
        pre["labels"][batch_i]=prediction
        batch_len=len(y_reals)
        correct = (prediction == y_reals).sum().item()
        acc = correct / batch_len
        total_acc += acc * batch_len
        #print("y_real",y_reals)
        #print("y_hat,",prediction)
        #sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)

    # Concatenate sample statistics
    #true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    #precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)
    precision=1
    #return precision, recall, AP, f1, ap_class
    pre.to_csv("predic.csv",index=None)
    return total_acc/len(dataset)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=InceptionV2().to(device)
model.load_state_dict(torch.load('./checkpoints/inception_ckpt_38.pth'))
predict(model,"pre.txt",224,1)