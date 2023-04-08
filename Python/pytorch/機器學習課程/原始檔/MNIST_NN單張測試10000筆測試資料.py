file_name='NN_202301141000.pth' # 副檔名通常以.pt或.pth儲存，建議使用.pth
import torch
device=torch.device('cuda') # 'cuda'/'cpu'，import torch
num_classes=10 # 物件類別數+1(背景)

# 取得網路
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
NN.load_state_dict(torch.load(file_name)) # import torch
NN.eval()

# 取得影像
from torchvision import transforms,datasets
transforms=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307),(0.3081))]) # 標準化每個channel為均值0.1307、標準差0.3081，import torchvision
dataset=datasets.MNIST('',train=False,download=False,transform=transforms) # ''指資料存放在程式目前的資料夾，train=True時dataset為60000筆訓練資料，其中dataset.data：[60000,28,28]，dataset.targets：[60000]，train=False時dataset為10000筆測試資料，其中dataset.data：[10000,28,28]，dataset.targets：[10000]，import torchvision
test_loader=torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=False) # import torch

import matplotlib.pyplot as plt
for index,(img,cls) in enumerate(test_loader): # img：[batch_size,1,28,28]，cls：[batch_size]
    img,cls=img.to(device),cls.to(device)
    pred=NN(img) # pred：[batch_size,num_classes]
    output_id=torch.max(pred,dim=1)[1] # output_id：網路輸出編號(0表示預測為第一個輸出)，[batch_size]
    print('actual    :',cls.cpu().numpy())
    print('prediction:',output_id.cpu().numpy())
    print('softmax   :',torch.softmax(pred,dim=1).cpu().detach().numpy())
    plt.imshow(dataset.data[index],cmap='gray') # 繪製點陣圖，cmap='gray'：灰階
    plt.axis('off') # 隱藏刻度
    plt.show()
    index+=1
