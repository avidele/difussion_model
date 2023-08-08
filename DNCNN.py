import os
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DnCNN(nn.Module):
    def __init__(self, num_layers=17):
        super(DnCNN, self).__init__()
        self.num_layers = num_layers
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            *(nn.Conv2d(64, 64, kernel_size=3, padding=1),
              nn.BatchNorm2d(64),
              nn.ReLU(inplace=True)) * (num_layers - 2),
            nn.Conv2d(64, 3, kernel_size=3, padding=1)
        )

    def forward(self, x):
        out = self.features(x)
        return out

mse_loss = nn.MSELoss()
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

class CustomDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.file_list = os.listdir(folder_path)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        file_path = os.path.join(self.folder_path, file_name)
        image = Image.open(file_path).convert('RGB')
        label = self.get_label(file_name)

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def get_label(self, file_name):
        if 'norain' in file_name:
            return 0
        elif 'rain' in file_name:
            return 1
        elif 'rainregion' in file_name:
            return 2
        elif 'rainstreak' in file_name:
            return 3
        else:
            raise ValueError(f"Invalid file name: {file_name}")
dataset = CustomDataset('train/RainTrainH', transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# 训练 DnCNN 模型
dncnn = DnCNN().to(device)
optimizer = optim.Adam(dncnn.parameters(), lr=0.0002, betas=(0.5, 0.999))
num_epochs = 10

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(dataloader):
        real_images = images.to(device)
        labels = labels.to(device)
        noise = torch.randn(real_images.size()).to(device)
        noisy_images = real_images + noise
        optimizer.zero_grad()
        generated_images = dncnn(noisy_images)
        loss = mse_loss(generated_images, real_images)
        loss.backward()
        optimizer.step()

def denoise_image(image):
    with torch.no_grad():
        image = image.unsqueeze(0).to(device)
        output = dncnn(image)
    return output.squeeze(0).cpu()
test_image = Image.open('1.png').convert('RGB')
test_image = transform(test_image)
denoised_image = denoise_image(test_image)

# 显示原始图像和去噪后的图像
import matplotlib.pyplot as plt

plt.subplot(1, 2, 1)
plt.imshow(test_image.permute(1, 2, 0))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(denoised_image.permute(1, 2, 0))
plt.title('Denoised Image')