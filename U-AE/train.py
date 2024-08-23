import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
from tqdm import tqdm  # 导入tqdm库

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义卷积自编码器
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # 编码器
        self.encoder1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # output: 16 x 1692 x 855
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # output: 32 x 846 x 428
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # output: 64 x 423 x 214
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        self.encoder4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # output: 128 x 212 x 107
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        self.encoder5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # output: 256 x 106 x 54
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )

        # 解码器
        self.decoder5 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # output: 128 x 212 x 107
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        self.decoder4 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # output: 64 x 423 x 214
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(128, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # output: 32 x 846 x 428
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(64, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # output: 16 x 1692 x 855
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # output: 1 x 3384 x 1710
            nn.Sigmoid()  # 使用Sigmoid以确保输出在[0, 1]范围内
        )

    def forward(self, x):
        # 编码器
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)
        enc5 = self.encoder5(enc4)

        # 解码器
        dec5 = self.decoder5(enc5)
        dec4 = self.decoder4(torch.cat([dec5, enc4], dim=1))
        dec3 = self.decoder3(torch.cat([dec4, enc3], dim=1))
        dec2 = self.decoder2(torch.cat([dec3, enc2], dim=1))
        dec1 = self.decoder1(torch.cat([dec2, enc1], dim=1))

        return dec1


# 自定义数据集加载器
class CustomImageDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_names = os.listdir(image_dir)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("L")

        label_name = img_name  # 假设图像名与标签名匹配
        label_path = os.path.join(self.label_dir, label_name)
        label_image = Image.open(label_path).convert("L")

        if self.transform:
            image = self.transform(image)
            label_image = self.transform(label_image)

        return image, label_image
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# IoU计算函数
def compute_iou(pred, target, threshold=0.5):
    pred = (pred > threshold).float()
    target = (target > threshold).float()

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    iou = intersection / union
    return iou.item()

# 图像预处理和数据加载
transform = transforms.Compose([
    transforms.Resize((1728, 3392)),
    transforms.ToTensor()
])


def train():
    for epoch in range(num_epochs):
        running_loss = 0.0
        total_iou = 0.0
        progress_bar = tqdm(data_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False)
        for data in progress_bar:
            imgs, label_imgs = data
            imgs, label_imgs = imgs.to(device), label_imgs.to(device)  # 将数据移动到GPU

            # 前向传播
            output = model(imgs)
            loss = criterion(output, label_imgs)

            # 计算IoU
            iou = compute_iou(output, label_imgs)
            total_iou += iou

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # 更新进度条描述
            progress_bar.set_postfix(loss=loss.item(), iou=iou)

        # 打印损失和IoU
        epoch_loss = running_loss / len(data_loader)
        epoch_iou = total_iou / len(data_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss}, IoU: {epoch_iou}')

        # 保存最佳IoU模型
        if epoch_iou > best_iou:
            best_iou = epoch_iou
            torch.save(model.state_dict(), best_model_path)
            print(f'Best model saved with IoU: {best_iou}')

        # 每5个epoch保存一次模型权重
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f'out_weights/conv_autoencoder_epoch_{epoch+1}.pth')
            print(f'Model weights saved at epoch {epoch+1}')


if __name__ == '__main__':
    image_dir = './img'  # 替换为你的图像文件夹路径
    label_dir = './label'  # 替换为你的标签文件夹路径

    dataset = CustomImageDataset(image_dir, label_dir, transform=transform)
    data_loader = DataLoader(dataset, batch_size=15, shuffle=True)

    # 实例化模型、定义损失函数和优化器
    model = ConvAutoencoder().to(device)  # 将模型移动到GPU
    print(f'The model has {count_parameters(model):,} trainable parameters')
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 保存最佳IoU模型
    best_iou = 0.0
    best_model_path = 'out_weights/best_conv_autoencoder.pth'
    # 训练卷积自编码器
    num_epochs = 100
    train()