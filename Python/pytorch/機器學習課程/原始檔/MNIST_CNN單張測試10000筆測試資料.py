file_name='Classifier_202211241900.pth' # 副檔名通常以.pt或.pth儲存，建議使用.pth
import torch
device=torch.device('cuda') # 'cuda'/'cpu'，import torch
num_classes=10 # 物件類別數+1(背景)

# 取得網路
from torch import nn
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.layer1=nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=16,kernel_size=5,stride=1,padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
        )
        self.layer2=nn.Sequential(
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=5,stride=1,padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
        )
        self.layer3=nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=5,stride=1,padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
        )        
        self.fc=nn.Linear(3*3*64,num_classes)

    def forward(self,x):
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=x.view(x.size(0),-1)
        x=self.fc(x)
        return x

classifier=CNN().to(device)
classifier.load_state_dict(torch.load(file_name)) # import torch
classifier.eval()

# 取得影像
from torchvision import transforms,datasets
transforms=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307),(0.3081))]) # 標準化每個channel為均值0.1307、標準差0.3081，import torchvision
dataset=datasets.MNIST('',train=False,download=False,transform=transforms) # ''指資料存放在程式目前的資料夾，train=True時dataset為60000筆訓練資料，其中dataset.data：[60000,28,28]，dataset.targets：[60000]，train=False時dataset為10000筆測試資料，其中dataset.data：[10000,28,28]，dataset.targets：[10000]，import torchvision
test_loader=torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=False) # import torch

import matplotlib.pyplot as plt
for index,(img,cls) in enumerate(test_loader): # img：[batch_size,1,28,28]，cls：[batch_size]
    img,cls=img.to(device),cls.to(device)
    pred=classifier(img) # pred：[batch_size,num_classes]
    output_id=torch.max(pred,dim=1)[1] # output_id：網路輸出編號(0表示預測為第一個輸出)，[batch_size]
    print('actual    :',cls.cpu().numpy())
    print('prediction:',output_id.cpu().numpy())
    print('softmax   :',torch.softmax(pred,dim=1).cpu().detach().numpy())
    plt.imshow(dataset.data[index],cmap='gray') # 繪製點陣圖，cmap='gray'：灰階
    plt.axis('off') # 隱藏刻度
    plt.show()
    index+=1
