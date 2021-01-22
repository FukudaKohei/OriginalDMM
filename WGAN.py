#!/usr/bin/env python

###################################################
#
# Wasserstein GANのサンプルプログラム by Kenta Hoshino
# Modification of GAN sample code by yunjey (https://github.com/yunjey/pytorch-tutorial)
#
###################################################

import os
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
latent_size = 64 # 変数zの次元
hidden_size = 256 # 隠れ層の次元
image_size = 784 # 画像のサイズ
num_epochs = 10 # 学習回数 (200くらいにするとよい)
batch_size = 100 # バッチ数
sample_dir = 'samples' # 結果を保存するディレクトリの名前

# Create a directory if not exists
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

# Image processing
transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5,),   # 3 for RGB channels
                                     std=(0.5,))])

# transform = transforms.Compose([
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean=(0.5, 0.5, 0.5),   # 3 for RGB channels
#                                      std=(0.5, 0.5, 0.5))])

# MNIST dataset
mnist = torchvision.datasets.MNIST(root='./data/',
                                   train=True,
                                   transform=transform,
                                   download=True)

# Data loader
data_loader = torch.utils.data.DataLoader(dataset=mnist,
                                          batch_size=batch_size, 
                                          shuffle=True)

data_loader2 = torch.utils.data.DataLoader(dataset=mnist,
                                          batch_size=batch_size, 
                                          shuffle=True)

# Discriminatorの定義
D = nn.Sequential(
    nn.Linear(image_size, hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size, 1))

# Generatorの定義
G = nn.Sequential(
    nn.Linear(latent_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, image_size),
    nn.Tanh())

# Device setting
D = D.to(device)
G = G.to(device)

# 学習に使うoptimizerを定義
d_optimizer = torch.optim.RMSprop(D.parameters(), lr=0.0001)
g_optimizer = torch.optim.RMSprop(G.parameters(), lr=0.0001)


def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

# DiscriminatorとGeneratorの勾配をリセットする関数
def reset_grad():
    d_optimizer.zero_grad()
    g_optimizer.zero_grad()

# Start training
total_step = len(data_loader)
for epoch in range(num_epochs):
    for i, (images, _) in enumerate(data_loader):
        # ================================================================== #
        #                      Train the discriminator                       #
        # ================================================================== #

        for _ in range(5): # Generatorの学習1回に対して，Discriminatorの学習を5回行う．

            # 実データの画像の取得と画像データの配列の整形
            images2, _ = next(iter(data_loader2))
            images2 = images2.reshape(batch_size, -1).to(device)

            # 実データをDiscriminatorにより評価 (論文のAlgorithm 1のStep 3)
            outputs_real = D(images2)

            # 偽の画像データのGeneratorによる生成(論文のAlgorithm 1のStep 4)
            z = torch.randn(batch_size, latent_size).to(device)
            G_sample = G(z)
            # Generatorによる画像をDiscriminatorによって判別
            outputs_fake = D(G_sample)

            # 評価関数を計算し，その勾配を計算してDiscriminatorの重みパラメータを更新 (Algorithm 1のStep 5)
            d_loss = -(torch.mean(outputs_real) - torch.mean(outputs_fake))
            reset_grad() # 勾配を計算する前に勾配を一度0にする
            d_loss.backward() # 勾配を計算
            d_optimizer.step() # 重みパラメータを更新 (Algorithm 1のStep 6)

            # 重みパラメータの値を-0.01から0.01の間にクリッピングする (Algorithm 1のStep 7)
            for p in D.parameters():
                p.data.clamp_(-0.01, 0.01)
            
                
        # ================================================================== #
        #                        Train the generator                         #
        # ================================================================== #

        # Generatorで偽の画像データを生成する (Algorithm 1のStep 9)
        z = torch.randn(batch_size, latent_size).to(device)
        fake_images = G(z)
        outputs_fake = D(fake_images)

        # 評価関数とその勾配を計算する (Algorithm 1のStep 10)
        g_loss = -torch.mean(outputs_fake)
        # Generatorの重みパラメータを更新する (Algorithm 1のStep 11)
        reset_grad()
        g_loss.backward()
        g_optimizer.step()

        # 途中経過を表示するための処理
        if (i+1) % 200 == 0:
            print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}' 
                  .format(epoch, num_epochs, i+1, total_step, d_loss.item(), g_loss.item()))
    
    # Save real images
    if (epoch+1) == 1:
        images = images.reshape(images.size(0), 1, 28, 28)
        save_image(denorm(images), os.path.join(sample_dir, 'real_images.png'))
    
    # Save sampled images
    fake_images = fake_images.reshape(fake_images.size(0), 1, 28, 28)
    save_image(denorm(fake_images), os.path.join(sample_dir, 'fake_images-{}.png'.format(epoch+1)))

# Save the model checkpoints 
torch.save(G.state_dict(), 'G.ckpt')
torch.save(D.state_dict(), 'D.ckpt')
