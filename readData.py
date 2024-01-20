# 读取文件
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

