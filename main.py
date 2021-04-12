import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from glob import glob
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torchvision import transforms
from torchsummary import summary


class ImageTransform():
    
    def __init__(self):
        self.data_transform = {
            'train': transforms.Compose([transforms.ToTensor()]),
            'validation': transforms.Compose([transforms.ToTensor()]),
            'test': transforms.Compose([transforms.ToTensor()])
        }

    def __call__(self, image, phase='train'):
        return self.data_transform[phase](image)


class MNISTDataset(data.Dataset):

    def __init__(self, phase='train', transform=None):
        
        target_path = os.path.join('dataset', phase, '**/*.png')
        path_list = glob(target_path)
        
        images = []
        labels = []
        
        for path in tqdm(path_list):
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            label = int(path.split(os.sep)[2]) # 画像のラベルをファイル名から取得
            images.append(image)
            labels.append(label)
        
        self.transform = transform
        self.phase = phase
        self.images = images
        self.labels = labels
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        # index番目の画像，ラベル
        image = self.images[index]  # H×W×C
        label = self.labels[index]

        image_transformed = self.transform(image, self.phase) # C×H×W

        return image_transformed, label


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
                                  nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2))
        
        self.fc = nn.Sequential(nn.Linear(64 * 7 * 7, 120), nn.ReLU(inplace=True),
                                nn.Linear(120, 84), nn.ReLU(inplace=True),
                                nn.Linear(84, 10))

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x.view(-1, 64 * 7 * 7))
        return x

def train():

    # parameters
    batch_size = 32
    epochs = 10
    learning_rate = 0.01
    save_dir = "result"

    #保存用ディレクトリを作成
    os.makedirs(save_dir, exist_ok=True)

    # GPUが使えるかを確認
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("使用デバイス：", device)

    # Tensorboard
    # log_dirでlogのディレクトリを指定
    # 新規端末で`tensorboard --logdir ./logs`により起動
    writer = SummaryWriter(log_dir="./logs")

    # Datasetを作成
    train_dataset = MNISTDataset(phase='train', transform=ImageTransform())
    val_dataset = MNISTDataset(phase='validation', transform=ImageTransform())

    # DataLoaderを作成
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    dataloaders_dict = {"train": train_dataloader, "validation": val_dataloader}

    # モデルの定義
    model = Net()

    # summaryを確認
    inputs, _ = list(dataloaders_dict["train"])[0]
    summary(model, inputs)

    # モデルをtensorboradへ
    writer.add_graph(model.to("cpu"), (inputs.to("cpu"), ))

    # モデルをGPUへ
    if torch.cuda.device_count() > 1: # マルチGPUが使える場合
        print("Let's use", torch.cuda.device_count(), "GPUs")
        model = nn.DataParallel(model)
    model.to(device) 

    # loss, optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # ネットワークがある程度固定であれば、高速化させる
    torch.backends.cudnn.benchmark = True

    # train
    history = {"train_loss":[], "val_loss":[], "train_acc":[], "val_acc":[]} # 学習曲線用
    for epoch in range(epochs):
        print('Epoch：{}/{}'.format(epoch+1, epochs), "-"*50)
        
        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                
            epoch_loss = 0.0
            epoch_corrects = 0
            
            with tqdm(dataloaders_dict[phase]) as pbar:
                for i, (inputs, labels) in enumerate(pbar):
                    
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # 順伝播
                    with torch.set_grad_enabled(phase == 'train'):
                        optimizer.zero_grad()
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        _, preds = torch.max(outputs, 1)  # ラベルを予測

                        # train時は学習
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                        epoch_loss += loss.item() * inputs.size(0)                  # lossの合計を更新
                        epoch_corrects += torch.sum(preds == labels.data).double()  # 正解数の合計を更新

                        # プログレスバーに表示
                        pbar.set_description('Epoch：{0}/{1} [{2}] '.format(epoch+1, epochs, phase[:5]))
                        pbar.set_postfix({"loss":"{:.5f}".format(epoch_loss/batch_size/(i+1)), "acc":"{:.5f}".format(epoch_corrects/batch_size/(i+1))})

                # epochごとのlossと正解率
                epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
                epoch_acc = epoch_corrects / len(dataloaders_dict[phase].dataset)
                
                # lossとaccuracyを保存
                if phase == "train":
                    history["train_loss"].append(epoch_loss)
                    history["train_acc"].append(epoch_acc)
                    writer.add_scalars("loss", {"train":epoch_loss}, epoch)
                    writer.add_scalars("accuracy", {"train":epoch_acc}, epoch)
                else:
                    history["val_loss"].append(epoch_loss)
                    history["val_acc"].append(epoch_acc)
                    writer.add_scalars("loss", {"validation":epoch_loss}, epoch)
                    writer.add_scalars("accuracy", {"validation":epoch_acc}, epoch)

    # tensorboardをclose
    writer.close()

    # モデルを保存する
    torch.save(model.to('cpu').state_dict(), os.path.join(save_dir, "model_MNIST.pth"))

    # 学習曲線を保存
    plt.figure()
    plt.title("loss")
    plt.plot(history["train_loss"], label="train")
    plt.plot(history["val_loss"], label="validation")
    plt.legend(loc="best")
    plt.savefig(os.path.join(save_dir, "loss.png"))

    plt.figure()
    plt.title("accuracy")
    plt.plot(history["train_acc"], label="train")
    plt.plot(history["val_acc"], label="validation")
    plt.legend(loc="best")
    plt.savefig(os.path.join(save_dir, "accuracy.png"))


def predict():

    print("predict", "-"*50)

    batch_size = 32
    save_dir = "result"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    test_dataset = MNISTDataset(phase='test', transform=ImageTransform())
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 保存したモデルを読み込む。
    model = Net()
    model.load_state_dict(torch.load(os.path.join(save_dir, "model_MNIST.pth")))

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs")
        model = nn.DataParallel(model)
    model.to(device)

    correct = 0

    with torch.no_grad():
        for inputs, labels in tqdm(test_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data)
    test_acc = correct.double() / len(test_dataloader.dataset)
    print("Test Accuracy :", test_acc.item())


if __name__ == "__main__":
    train()
    predict()