import pandas as pd
import torch.nn as nn
import torch.utils.data
import torch
import modeL
from readData import myData4Test

rootDir = "dataset"
testDir = "test"
testSet = myData4Test(rootDir, testDir)

testLoader = torch.utils.data.DataLoader(dataset= testSet, batch_size= 4, shuffle= False)

model = modeL.DesNet(2)

model.load_state_dict(torch.load("modelWeight.pth"))
device = torch.device("cuda")
model = model.to(device)
lossF = nn.CrossEntropyLoss()
lossF = lossF.to(device)
testSize = len(testSet)
model.eval()
testLoss = 0
testAcc = 0.0
outpuT = []
indeX = []
i = 0
with torch.no_grad():
    for data in testLoader:
        imgs,indexs = data
        imgs = imgs.to(device)
        outputs = model(imgs)
        outputs = outputs.argmax(1)
        indeX.extend(list(indexs))
        outpuT.extend(outputs.tolist())
        i += 1
        if i % 10 == 0:
            print("{} / {}".format(i,len(testLoader)))
ind = ['Image','Predict']
tmp = zip(indeX,outpuT)
finalData = [list(x) for x in tmp]
df = pd.DataFrame(data = finalData,columns= ind)
df.to_csv('results.csv',index=False)
