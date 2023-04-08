file_name='NN_202301141000.pth' # 副檔名通常以.pt或.pth儲存，建議使用.pth
import torch
device=torch.device('cpu') # 'cuda'/'cpu'，import torch
print(device)
num_classes=10
train_size=50000
valid_size=10000
batch_size=1024
learning_rate=0.1
step_size=1000 # Reriod of learning rate decay
epochs=100
from torchvision import transforms,datasets
transforms=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307),(0.3081))]) # 標準化每個channel為均值0.1307、標準差0.3081，import torchvision
dataset=datasets.FashionMNIST('',train=True,download=False,transform=transforms) # ''指資料存放在程式目前的資料夾，train=True時dataset為60000筆訓練資料，其中dataset.data：[60000,28,28]，dataset.targets：[60000]，train=False時dataset為10000筆測試資料，其中dataset.data：[10000,28,28]，dataset.targets：[10000]，import torchvision
train_data,valid_data=torch.utils.data.random_split(dataset,[train_size,valid_size]) # import torch
train_loader=torch.utils.data.DataLoader(train_data,batch_size=batch_size,shuffle=True) # import torch
valid_loader=torch.utils.data.DataLoader(valid_data,batch_size=batch_size,shuffle=True) # import torch

from torch import nn
from torch.nn import functional as F
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork,self).__init__()
        self.fc1=nn.Linear(28*28,256)
        self.fc2=nn.Linear(256,num_classes)

    def forward(self,x):
        x=torch.flatten(x,1)
        x=self.fc1(x)
        x=self.fc2(x)
        return x

NN=NeuralNetwork().to(device)
criterion=nn.CrossEntropyLoss() # 分類
optimizer=torch.optim.Adadelta(NN.parameters(),lr=learning_rate) # import torch
#optimizer=torch.optim.Adam(NN.parameters(),lr=learning_rate) # import torch
scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size,0.1) # import torch
import numpy 
train_acc_his,train_losses_his,valid_acc_his,valid_losses_his=[],[],[],[]
for i in range(1,epochs+1):
    print('Running Epoch:'+str(i))
    train_correct,train_loss,train_total,valid_correct,valid_loss,valid_total=0,0,0,0,0,0
    NN.train()
    for img,cls in train_loader: # 一個batch的img、cls。img：[batch_size,1,28,28]，cls：[batch_size]
        img,cls=img.to(device),cls.to(device)
        pred=NN(img) # pred：[batch_size,num_classes]
        loss=criterion(pred,cls) # loss.item()：一個batch的平均loss，[1]
        output_id=torch.max(pred,dim=1)[1] # output_id：網路輸出編號(0表示預測為第一個輸出)，[batch_size]
        train_correct+=numpy.sum(torch.eq(cls,output_id).cpu().numpy()) # 累加計算每一epoch正確預測總數，import numpy
        train_loss+=loss.item()*img.size(0) # 累加計算每一epoch的loss總和。loss.item()：一個batch的平均loss，[1]。img.size(0)：一個batch的訓練資料總數
        train_total+=img.size(0) # 累加計算訓練資料總數
        optimizer.zero_grad() # 權重梯度歸零
        loss.backward() # 計算每個權重的loss梯度
        optimizer.step() # 權重更新
    scheduler.step()

    NN.eval()
    for img,cls in valid_loader: # 一個batch的img、cls。img：[batch_size,1,28,28]，cls：[batch_size]
        img,cls=img.to(device),cls.to(device)
        pred=NN(img) # pred：[batch_size,num_classes]
        loss=criterion(pred,cls) # loss.item()：一個batch的平均loss，[1]
        output_id=torch.max(pred,dim=1)[1] # output_id：網路輸出編號(0表示預測為第一個輸出)，[batch_size]
        valid_correct+=numpy.sum(torch.eq(cls,output_id).cpu().numpy()) # 累加計算每一epoch正確預測總數
        valid_loss+=loss.item()*img.size(0) # 累加計算每一epoch的loss總和。loss.item()：一個batch的平均loss，[1]。img.size(0)：一個batch的驗證資料總數
        valid_total+=img.size(0) # 累加計算驗證資料總數

    train_acc=train_correct/train_total*100 # 計算每一個epoch的平均訓練正確率(%)
    train_loss=train_loss/train_total # 計算每一個epoch的平均訓練loss
    valid_acc=valid_correct/valid_total*100 # 計算每一個epoch的平均驗證正確率(%)
    valid_loss=valid_loss/valid_total # 計算每一個epoch的平均驗證loss
    train_acc_his.append(train_acc) # 累積紀錄每一個epoch的平均訓練正確率(%)，[epochs]
    train_losses_his.append(train_loss) # 累積記錄每一個epoch的平均訓練loss，[epochs]
    valid_acc_his.append(valid_acc) # 累積紀錄每一個epoch的平均驗證正確率(%)，[epochs]
    valid_losses_his.append(valid_loss) # 累積記錄每一個epoch的平均驗證loss，[epochs]
    print('Training Loss='+str(train_loss))
    print('Training Accuracy(%)='+str(train_acc))
    print('Validation Accuracy(%)='+str(valid_acc))

# 顯示第71筆訓練資料影像
import matplotlib.pyplot as plt
plt.imshow(dataset.data[70],cmap='gray') # 繪製點陣圖，cmap='gray'：灰階
plt.axis('off') # 隱藏刻度

# 對訓練過程繪圖
plt.figure(figsize=(15,10))
plt.subplot(211)
plt.plot(train_acc_his,'b',label='trainingaccuracy')
plt.plot(valid_acc_his,'r',label='validation accuracy')
plt.title('Accuracy(%)')
plt.legend(loc='best')
plt.subplot(212)
plt.plot(train_losses_his,'b',label='training loss')
plt.plot(valid_losses_his,'r',label='validation loss')
plt.title('Loss')
plt.legend(loc='best')
plt.show()

torch.save(NN.state_dict(),file_name)