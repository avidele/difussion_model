import os
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torch.autograd import Variable
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(3 * 256 * 256, 1024)
        self.fc2 = nn.Linear(1024, 2048)
        self.fc3 = nn.Linear(2048, 3 * 256 * 256)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 将输入展平为一维向量
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x.view(x.size(0), 3, 256, 256)

mse_loss = nn.MSELoss()
generator = Generator()
optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
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

        if self.transform is not None:
            image = self.transform(image)

        return image

dataset = CustomDataset('images', transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
num_epochs = 10

for epoch in range(num_epochs):
    for i, images in enumerate(dataloader):
        real_images = Variable(images)
        noise = torch.randn(real_images.size())
        noisy_images = real_images + noise
        optimizer.zero_grad()
        generated_images = generator(noisy_images)
        loss = mse_loss(generated_images, real_images.transpose(3, 2))
        loss.backward()
        optimizer.step()

# 去雾预测
def denoise_image(image):
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(image.size())
        noisy_image = image + noise
        noisy_image = Variable(noisy_image.unsqueeze(0))
        output = generator(noisy_image)
    return output.squeeze(0)

test_image = Image.open('1.png').convert('RGB')
test_image = transform(test_image)

import matplotlib.pyplot as plt


original_image = test_image.permute(1, 2, 0)  # 将通道维度放到最后
plt.imshow(original_image)
plt.title('Original Image')
plt.axis('off')
plt.show()
denoised_image = denoise_image(test_image)
denoised_image = transforms.ToPILImage()(denoised_image)
plt.imshow(denoised_image)
plt.title('Final Image')
plt.axis('off')
plt.show()