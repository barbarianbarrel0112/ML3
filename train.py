import torch
print("1")
print(torch.__version__)
print(torch.cuda.is_available())

import sys
import numpy as np
import torch.nn as nn
import torch.utils.data
import torch
import modeL
from readData import myData

import torch
print(torch.cuda.is_available())

#读取数据
import numpy as np
import pandas as pd
import torch
from PIL import Image
import os
from torch.utils.data import Dataset
class myData(Dataset):
    def __init__(self,  rootDir, subDir, labelDir):
        self.rootDir = rootDir
        self.subDir = subDir
        self.path = os.path.join(self.rootDir,self.subDir)
        self.imgPath = os.listdir(self.path)
        df = pd.read_csv(labelDir)
        self.labelDict = dict(df.values)
        for k in list(self.labelDict):
            self.labelDict[str(k)] = self.labelDict.pop(k)



    def __getitem__(self, idx):
        imgName = self.imgPath[idx]
        imgIndex = imgName[:-4]
        imgAbsName = os.path.join(self.path,imgName)
        img = Image.open(imgAbsName)
        img = np.array(img,dtype=float).transpose(2, 0, 1) / 255
        img = torch.tensor(img, dtype=torch.float32)
        label = torch.tensor(self.labelDict[imgIndex],dtype=torch.long)
        return img, label
    def __len__(self):
        return len(self.imgPath)

class myData4Test(Dataset):
    def __init__(self,  rootDir, subDir):
        self.rootDir = rootDir
        self.subDir = subDir
        self.path = os.path.join(self.rootDir,self.subDir)
        self.imgPath = os.listdir(self.path)


    def __getitem__(self, idx):
        imgName = self.imgPath[idx]
        imgIndex = imgName[:-4]
        imgAbsName = os.path.join(self.path,imgName)
        img = Image.open(imgAbsName)
        img = np.array(img,dtype=float).transpose(2, 0, 1)
        img = torch.tensor(img / 255, dtype=torch.float32)
        return img, imgIndex
    def __len__(self):
        return len(self.imgPath)


rootDir = "dataset"
trainDir = "train"
dataSet = myData(rootDir, trainDir, "dataset/Mess1_annotation_train.csv")

trainSet, testSet = torch.utils.data.random_split(dataSet,[round(0.8 * len(dataSet)),round(0.2 * len(dataSet))],generator=torch.Generator().manual_seed(42))
trainLoader = torch.utils.data.DataLoader(dataset= trainSet, batch_size= 4, shuffle= True)
testLoader = torch.utils.data.DataLoader(dataset= testSet, batch_size= 4, shuffle= False)


testSize = len(testSet)
model = modeL.DesNet(2)
device = torch.device("cuda")
model = model.to(device)
lossF = nn.CrossEntropyLoss()
lossF = lossF.to(device)
learningRate = 1e-4
optim = torch.optim.Adam(model.parameters(),lr = learningRate)
epochNum = 30
bestLoss = np.inf
bestAcc = 0.0
bestWeight = None
bestEpoch = 0
patience = 5
batchNum = len(trainLoader)
withoutDev = 0
for epoch in range(epochNum):
    print("epoch {}".format(epoch+1))
    eLoss = 0;
    batchIndex = 0;
    for data in trainLoader:
        model.train()
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = model(imgs)
        result_loss = lossF(outputs,targets)
        eLoss += result_loss
        optim.zero_grad()
        result_loss.backward()
        optim.step()
        batchIndex += 1
    model.eval()
    testLoss = 0
    testAcc = 0.0
    with torch.no_grad():
        for data in testLoader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = model(imgs)
            valiLoss = lossF(outputs, targets)
            testLoss += valiLoss
            acc = (outputs.argmax(1) == targets).sum()
            testAcc += acc
    testAcc /= testSize
    print("    test loss: {}".format(testLoss))
    print("    test acc: {}".format(testAcc))

    if bestAcc < testAcc:
        withoutDev = 0
        bestLoss = testLoss
        bestAcc = testAcc
        bestEpoch = epoch + 1
        bestWeight = model.state_dict()
        torch.save(bestWeight, "modelWeight.pth")
    else:
        withoutDev += 1
    print("    best Acc: {} loss: {}".format(bestLoss, bestAcc))
    print("    patience: {}".format(patience-withoutDev))
    if withoutDev == patience:
        sys.exit()
    print("train loss: {}".format(eLoss))