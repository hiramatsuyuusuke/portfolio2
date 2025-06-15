#CIFAR10を「転がる箱の衝突回避」用に書き換えて、ResNet18の学習を行います。
#ResNet18の学習部分のソースコードは、Qiitaの「CIFAR-10の分類を実質60行で実装in PyTorch」の記事を参考にさせていただきました。

from PIL import Image

import torch
import numpy as np
import torch.nn as nn
from torch import optim
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader
import torchvision.datasets as dsets

batch_size = 100
train_data = dsets.CIFAR10(root='data', train=True, download=True, transform=transforms.ToTensor())
test_data = dsets.CIFAR10(root='data', train=False, download=True, transform=transforms.ToTensor())

#クラス(10個)を末尾から削除して、0番と1番だけ書き換える
#['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'] → ['airplane', 'automobile']
i = 0
while i < 8:
    train_data.classes.pop(-1) #クラスを末尾から削除
    test_data.classes.pop(-1) #クラスを末尾から削除
    i += 1

#「転がる箱の衝突回避」用のクラスラベルを読み込む
np_class_label = np.loadtxt("box_collision_class_label.txt", delimiter=",")
print(len(np_class_label))
img_file_num = len(np_class_label)

#「転がる箱の衝突回避」用の画像を読み込んで画像データを変更し、ラベル番号を書き換える
j = 0   #while j < len( train_data):のカウント
i = 0   #読み込み画像番号のカウント
while j < len( train_data):
    #「転がる箱の衝突回避」用の画像の読み込み
    file_name1 = "img/test"+ str(i) + ".jpg"
    img_field = Image.open(file_name1)
   
    train_data.data[j] = img_field #画像データの変更
    train_data.targets[j] = int(np_class_label[i]) #ラベルデータの変更
    #読み込み画像番号のカウント
    i += 1
    if i == img_file_num:
      i = 0
      print("train" + str(j) + "," + str(i))
    j += 1

#「転がる箱の衝突回避」用の画像を読み込んで画像データを変更し、ラベル番号を変書き換える
j = 0   #while j < len( test_data):のカウント
i = 0   #読み込み画像番号のカウント
while j < len( test_data):
    #「転がる箱の衝突回避」用の画像の読み込み
    file_name1 = "img/test"+ str(i) + ".jpg"
    img_field = Image.open(file_name1)
    
    test_data.data[j] = img_field #画像データの変更
    test_data.targets[j] = int(np_class_label[i]) #ラベルデータの変更
    #読み込み画像番号のカウント
    i += 1
    if i == img_file_num:
      i = 0
      print("test" + str(j) + "," + str(i))
    j += 1

#書き換えたデータを使ってデータローダーを生成
train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True)
test_loader = DataLoader(test_data,batch_size=batch_size,shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model_ft = models.resnet18(weights = None)
model_ft.fc = nn.Linear(model_ft.fc.in_features, 2)
net = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(net.parameters(),lr=0.02,momentum=0.9,weight_decay=0.00005)

loss, epoch_loss, count = 0, 0, 0
acc_list = []
loss_list = []

# 訓練・推論
for i in range(1):
  
  #ここから学習
  net.train()
  
  for j,data in enumerate(train_loader,0):
    optimizer.zero_grad()

    #1:訓練データを読み込む
    inputs,labels = data
    inputs = inputs.to(device)
    labels = labels.to(device)

    #2:計算する
    outputs = net(inputs)

    #3:誤差を求める
    loss = criterion(outputs,labels)

    #4:誤差から学習する
    loss.backward()
    optimizer.step()

    epoch_loss += loss
    count += 1
    print('%d: %.3f'%(j+1,loss))

  print('%depoch:mean_loss=%.3f\n'%(i+1,epoch_loss/count))
  loss_list.append(epoch_loss/count)

  epoch_loss, count = 0, 0
  correct,total = 0, 0
  accuracy = 0.0

  #ここから推論
  net.eval()
 
  for j,data in enumerate(test_loader,0):

    #テストデータを用意する
    inputs,labels = data
    inputs = inputs.to(device)
    labels = labels.to(device)

    #計算する
    outputs = net(inputs)

    #予測値を求める
    _,predicted = torch.max(outputs.data,1)

    #精度を計算する
    correct += (predicted == labels).sum()
    total += batch_size

  accuracy = 100.*correct / total
  acc_list.append(accuracy)

  print('epoch:%d Accuracy(%d/%d):%f'%(i+1,correct,total,accuracy))
  torch.save(net.state_dict(),'Weight'+str(i+1)+'.pth')

#plt.plot(acc_list)
#plt.show(acc_list)
#plt.plot(loss_list)
#plt.show(loss_list)
