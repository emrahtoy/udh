import argparse
import contextlib
import os
import cv2
import torch
import numpy as np
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasetsss import MyDataset
from syncnet import SyncNet_color
from unet import Model
import random
import torchvision.models as models

mixed_precision = True

# Enable automatic mixed precision if available
try:
    if(mixed_precision):
        from torch.cuda.amp import autocast, GradScaler
    else:
        raise ImportError()
except ImportError:
    print("torch.cuda.amp not found. Mixed precision training will be disabled.")
    autocast = lambda: contextlib.nullcontext()
    GradScaler = None

def get_args():
    parser = argparse.ArgumentParser(description='Train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--use_syncnet', action='store_true', help="if use syncnet, you need to set 'syncnet_checkpoint'")
    parser.add_argument('--syncnet_checkpoint', type=str, default="")
    parser.add_argument('--dataset_dir', type=str)
    parser.add_argument('--save_dir', type=str, help="trained model save path.")
    parser.add_argument('--see_res', action='store_true')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batchsize', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--asr', type=str, default="hubert")
    parser.add_argument('--continue_checkpoint', type=int, default=0)

    return parser.parse_args()

args = get_args()
use_syncnet = args.use_syncnet

# Loss functions
class PerceptualLoss():
    def contentFunc(self):
        conv_3_3_layer = 14
        cnn = models.vgg19(pretrained=True).features.cuda()
        model = nn.Sequential(*list(cnn)[:conv_3_3_layer+1]).cuda()
        return model

    def __init__(self, loss):
        self.criterion = loss
        self.contentFunc = self.contentFunc()

    def get_loss(self, fakeIm, realIm):
        f_fake = self.contentFunc(fakeIm)
        f_real = self.contentFunc(realIm.detach())
        loss = self.criterion(f_fake, f_real)
        return loss

# Use BCEWithLogitsLoss instead of BCELoss
logloss = nn.BCELoss() if mixed_precision==False else nn.BCEWithLogitsLoss()
def cosine_loss(a, v, y):
    # print(f"a:{str(a)} v:{str(v)} y:{str(y)}")
    d = nn.functional.cosine_similarity(a, v)
    # loss = nn.functional.binary_cross_entropy_with_logits(d.unsqueeze(1), y)
    # loss = nn.functional.binary_cross_entropy(d.unsqueeze(1), y)
    loss = logloss(d.unsqueeze(1), y)
    return loss

def train(net, epoch, batch_size, lr):
    content_loss = PerceptualLoss(torch.nn.MSELoss())
    if use_syncnet:
        if args.syncnet_checkpoint == "":
            raise ValueError("Using syncnet, you need to set 'syncnet_checkpoint'. Please check README")
        syncnet = SyncNet_color(args.asr).eval().cuda()
        syncnet.load_state_dict(torch.load(args.syncnet_checkpoint))
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    dataloader_list = []
    dataset_list = []
    dataset_dir_list = [args.dataset_dir]
    for dataset_dir in dataset_dir_list:
        dataset = MyDataset(dataset_dir, args.asr)
        train_dataloader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=False, num_workers=4, pin_memory=True)
        dataloader_list.append(train_dataloader)
        dataset_list.append(dataset)

    optimizer = optim.Adam(net.parameters(), lr=lr)
    criterion = nn.L1Loss()
    
    # Initialize scaler if mixed precision is enabled
    scaler = GradScaler() if GradScaler else None

    for e in range(args.continue_checkpoint, epoch):
        net.train()
        random_i = random.randint(0, len(dataset_dir_list)-1)
        dataset = dataset_list[random_i]
        train_dataloader = dataloader_list[random_i]

        with tqdm(total=len(dataset), desc=f'Epoch {e + 1}/{epoch}', unit='img') as p:
            for batch in train_dataloader:
                imgs, labels, audio_feat = batch
                imgs = imgs.cuda()
                labels = labels.cuda()
                audio_feat = audio_feat.cuda()

                with autocast():
                    preds = net(imgs, audio_feat)
                    if use_syncnet:
                        y = torch.ones([preds.shape[0],1]).float().cuda()
                        a, v = syncnet(preds, audio_feat)
                        sync_loss = cosine_loss(a, v, y)
                    loss_PerceptualLoss = content_loss.get_loss(preds, labels)
                    loss_pixel = criterion(preds, labels)
                    if use_syncnet:
                        loss = loss_pixel + loss_PerceptualLoss*0.01 + 10*sync_loss
                    else:
                        loss = loss_pixel + loss_PerceptualLoss*0.01

                p.set_postfix(**{'loss (batch)': loss.item()})
                optimizer.zero_grad(set_to_none=True)
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                

                p.update(imgs.shape[0])

        if (e+1) % 5 == 0:
            torch.save(net.state_dict(), os.path.join(save_dir, str(e+1)+'.pth'))
        if args.see_res:
            net.eval()
            if not os.path.exists("./train_tmp_img"):
                 os.mkdir("./train_tmp_img")
            for c in range(5):
                random_index = random.randint(0, dataset.__len__()-1)
                img_concat_T, img_real_T, audio_feat = dataset.__getitem__(random_index)
                img_concat_T = img_concat_T[None].cuda()
                audio_feat = audio_feat[None].cuda()
                with torch.no_grad():
                    pred = net(img_concat_T, audio_feat)[0]
                pred = pred.cpu().numpy().transpose(1,2,0)*255
                pred = np.array(pred, dtype=np.uint8)
                img_real = img_real_T.numpy().transpose(1,2,0)*255
                img_real = np.array(img_real, dtype=np.uint8)
                cv2.imwrite("./train_tmp_img/epoch_"+str(e+1)+"_"+str(c)+"_"+str(random_index)+".jpg", pred)
                cv2.imwrite("./train_tmp_img/epoch_"+str(e+1)+"_"+str(c)+"_"+str(random_index)+"_real.jpg", img_real)

if __name__ == '__main__':
    net = Model(6, args.asr).cuda()
    previous_weights = os.path.join(args.save_dir, str(args.continue_checkpoint)+'.pth')
    if os.path.exists(previous_weights):
        net.load_state_dict(torch.load(previous_weights))
    train(net, args.epochs, args.batchsize, args.lr)
