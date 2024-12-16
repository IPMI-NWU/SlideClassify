import glob
import os
import cv2
import numpy as np
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset, DataLoader
from models.resnet_custom import resnet50_baseline
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

patchRoot = "/home/ipmi2023/project/Data_18t/ipmi2023/data_ipmi2023_18t/patches/shaoyifu/patches"
desPath = "/home/ipmi2023/project/Data_18t/ipmi2023/data_ipmi2023_18t/BCL_log/ShaoYiFu/feature"
device = torch.device("cuda:1")
batch_size = 32

normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
trans = transforms.Compose([
    transforms.ToTensor(),
    normalize])

class Model(nn.Module):
    def __init__(self, input_dim=1024,n_classes=4):
        super(Model, self).__init__()
        self.backbone = resnet50_baseline(True)
        self.backbone.fc = nn.Linear(1024, n_classes)

        self.fc = nn.Linear(1024, n_classes)

    def forward(self, x):
        _, feat = self.backbone(x)
        prob = self.fc(feat)
        return prob, feat

class CustomDataset(Dataset):
    def __init__(self,folder_path, transform=None):
        self.folder_path = folder_path
        self.image_files = glob.glob(folder_path+"/*")
        self.image_files = sorted(self.image_files)
        self.transform = transform
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_path = self.image_files[index]
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8),-1)
        # image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if self.transform:
            image = self.transform(image)
        if image.shape != torch.Size([3, 256, 256]):
            # print(image.shape)
            # print(image_path)
            image = F.interpolate(image.unsqueeze(0), size = (256,256), mode='bilinear', align_corners=False)
            image = image.squeeze(0)
            # image = torch.zeros((3,256,256))
        return image


model = Model(input_dim=1024)
model.to(device)
model.eval()

slideAll = glob.glob(patchRoot+"/*")
for process_i,slide_item in enumerate(slideAll):
    slideName = os.path.basename(slide_item)
    print(f"extract {slide_item} {process_i} / {len(slideAll)}")

    if os.path.exists(os.path.join(desPath, f'{slideName}.pt')):
        print(f"exists!!!")
        continue

    dataset = CustomDataset(slide_item, transform=trans)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    features = torch.Tensor()
    with torch.no_grad():
        for input in tqdm(dataloader):
            input = input.to(device)
            _, feature = model(input)
            features = torch.cat((features, feature.cpu()), dim=0)
    torch.save(features.cpu(), os.path.join(desPath, f'{slideName}.pt'))

