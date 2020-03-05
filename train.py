#各种库的导入
from __future__ import division
from models import *

from utils.datasets import *
from utils.loss import *
from utils.parse_config import *
from test import evaluate

import os
import time
import datetime
import argparse
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

def lr_decay(lr):
    return lr*0.95


if __name__ == "__main__":

    #载入部分超参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--learning_rate", default=0.001, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=6, help="size of each image batch")
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    parser.add_argument("--model_def", type=str, default="", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/casia.data", help="path to data config file")
    parser.add_argument("--img_size", type=int, default=224, help="size of each image dimension")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
    parser.add_argument("--pretrained_weights", type=str, default='',
                        help="if specified starts from checkpoint model")
    opt = parser.parse_args()
    #print("type(opt) --> ",type(opt))
    #print(opt)

    #创建文件夹
    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    #判断能否使用gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initiate model
    # 初始化网络模型
    #model = InceptionV2(opt.model_def, img_size=opt.img_size).to(device)
    model = InceptionV2().to(device)

    # 载入已经预训练的模型参数

    # If specified we start from checkpoint
    if opt.pretrained_weights:
        if opt.pretrained_weights.endswith(".pth"):
            model.load_state_dict(torch.load(opt.pretrained_weights))
        #else:
            #model.load_darknet_weights(opt.pretrained_weights)

    # Get data configuration
    # 获取数据配置
    data_config = parse_data_config(opt.data_config)

    train_path = data_config["train"]
    test_path = data_config["test"]


    # Get dataloader
    # 获取dataloader
    dataset = CASIA_Dataset(train_path,img_size=opt.img_size)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,  # 将数据整合成一个batch返回的方法
    )

    # 优化器
    lr=opt.learning_rate
    #optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate)
    precisions=[]
    for epoch in range(opt.epochs):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        model.train()
        start_time = time.time()
        lr=lr_decay(lr)
        print(lr)
        for batch_i, (_, imgs, y_reals) in enumerate(dataloader):
            batches_done = len(dataloader) * epoch + batch_i  #已训练的图片batch数
            imgs = Variable(imgs.to(device))
            y_reals = Variable(y_reals.to(device), requires_grad=False)
            #y_reals=real2more(y_reals,10)
            y_hats = model(imgs)
            loss=loss_CEL(y_hats,y_reals)
            loss.backward()

            if batches_done % opt.gradient_accumulations:
                # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()

            # ----------------
            #   Log progress
            # ----------------

            log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, opt.epochs, batch_i, len(dataloader))
            # log_str += AsciiTable(metric_table).table
            log_str += f"\nTotal loss {loss.item()}"

            # Determine approximate time left for epoch
            epoch_batches_left = len(dataloader) - (batch_i + 1)
            time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
            log_str += f"\n---- ETA {time_left}"

            print(log_str)
            
            #model.seen += imgs.size(0)

        if epoch % opt.evaluation_interval == 0:
            print("\n---- Evaluating Model ----")
            precision = evaluate(
                        model,
                        test_path,
                        opt.img_size,
                        6
                    )
            precisions.append(precision)
            print(precision)
            print(precisions)

        if epoch % opt.checkpoint_interval == 0:
            torch.save(model.state_dict(), f"checkpoints/inception_ckpt_%d.pth" % epoch)
    # writer.close()
