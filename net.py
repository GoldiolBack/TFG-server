from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from create_dataset_all_tiles import train_ds, val_ds, test_ds
import skimage.metrics as skm
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import os


logs_base_dir = "runs"
os.makedirs(logs_base_dir, exist_ok=True)

tb = SummaryWriter()


def crop(imageHR, imageLR, target, sizeHR, sizeLR, sizeTarget):
    imageHR_crop = torch.zeros([imageHR.shape[0], imageHR.shape[1], sizeHR, sizeHR], dtype=torch.float32)
    imageLR_crop = torch.zeros([imageLR.shape[0], imageLR.shape[1], sizeLR, sizeLR], dtype=torch.float32)
    target_crop = torch.zeros([target.shape[0], target.shape[1], sizeTarget, sizeTarget], dtype=torch.float32)

    for i in range(imageHR.shape[0]):
        j1 = np.random.randint(low=0, high=sizeHR / 2 - 1) * 2
        j2 = np.round((j1 / 2)).astype(dtype=np.int)
        k1 = np.random.randint(low=0, high=sizeLR / 2 - 1) * 2
        k2 = np.round((k1 / 2)).astype(dtype=np.int)
        imageHR_crop[i] = imageHR[i, :, j1:(j1+sizeHR), k1:(k1+sizeHR)]
        imageLR_crop[i] = imageLR[i, :, j2:(j2+sizeLR), k2:(k2+sizeLR)]
        target_crop[i] = target[i, :, j1:(j1+sizeTarget), k1:(k1+sizeTarget)]

    return imageHR_crop, imageLR_crop, target_crop


class Net(nn.Module):
    def __init__(self, input_size=10, feature_size=128, kernel_size=3):
        super(Net, self).__init__()
        self.ups = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True)
        self.conv1 = nn.Conv2d(input_size, feature_size, kernel_size, stride=(1, 1), padding=(1, 1))
        self.conv2 = nn.Conv2d(feature_size, 6, kernel_size, 1, 1)
        self.rBlock = ResBlock(feature_size, kernel_size)

    def forward(self, input10, input20, num_layers=6):
        upsamp20 = self.ups(input20)
        sentinel = torch.cat((input10, upsamp20), 1)
        x = sentinel
        x = self.conv1(x)
        x = F.relu(x)
        for i in range(num_layers):
            x = self.rBlock(x)
        x = self.conv2(x)
        x += upsamp20
        return x


class ResBlock(nn.Module):
    def __init__(self, feature_size=128, channels=3, kernel_size=3):
        super(ResBlock, self).__init__()
        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size, 1, 1)

    def forward(self, x, scale=0.1):
        tmp = self.conv3(x)
        tmp = F.relu(tmp)
        tmp = self.conv3(tmp)
        tmp = tmp * scale
        tmp += x
        return tmp


def train(args, train_loader, model, device, optimizer, epoch):
    model.train()
    for batch_idx, (hr, lr, target) in enumerate(train_loader):
#        print(f'batch {batch_idx+1}:')
        hr_crop, lr_crop, target_crop = crop(hr, lr, target, int(args.crop_size), int(args.crop_size/2), int(args.crop_size))
        lr_crop, hr_crop, target_crop = lr_crop.to(device), hr_crop.to(device), target_crop.to(device)
        optimizer.zero_grad()
        output = model(hr_crop, lr_crop)
#        gt = gt.long()
        loss_function = nn.L1Loss()
#        loss = F.nll_loss(output, gt)
        loss = loss_function(output, target_crop)
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(lr), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        tb.add_scalar("Loss_train", loss.item(), epoch)


def test(args, test_loader, model, device, epoch):
    model.eval()
    test_loss = 0
    rmse = 0
    psnr = 0
    ssim = 0
    with torch.no_grad():
        for hr, lr, target in test_loader:
            hr_crop, lr_crop, target_crop = crop(hr, lr, target, int(args.crop_size), int(args.crop_size/2), int(args.crop_size))
            lr_crop, hr_crop, target_crop = lr_crop.to(device), hr_crop.to(device), target_crop.to(device)
            output = model(hr_crop, lr_crop)
            test_loss_function = nn.L1Loss(reduction='mean')
            test_loss += test_loss_function(output, target_crop).item()
            real = np.moveaxis(target_crop.cpu().numpy(), 1, 3) * 2000
            predicted = np.moveaxis(output.cpu().numpy(), 1, 3) * 2000
            for i in range(real.shape[0]):
                rmse += np.sqrt(skm.mean_squared_error(real[i], predicted[i]))
                psnr += skm.peak_signal_noise_ratio(real[i], predicted[i], data_range=real.max() - real.min())
                ssim += skm.structural_similarity(real[i], predicted[i], multichannel=True,
                                                    data_range=real.max() - real.min())
            tb.add_scalar("Loss_test", test_loss, epoch)
            tb.add_scalar("RMSE_test", rmse/5, epoch)
            tb.add_scalar("PSNR_test", psnr/5, epoch)
            tb.add_scalar("SSIM_test", ssim/5, epoch)

#     test_loss /= len(test_loader.dataset)
    rmse /= len(test_loader.dataset)
    psnr /= len(test_loader.dataset)
    ssim /= len(test_loader.dataset)

    print('\nTest set: Average values --> Loss: {:.4f}, RMSE: ({:.2f}), PSNR: ({:.2f}dB),'
          ' SSIM: ({:.2f})\n'.format(test_loss, rmse, psnr, ssim))

    if epoch%10 == 0:
        np.save('test_input.npy', (np.moveaxis(lr_crop.cpu().numpy() * 2000, 1, 3)))
        np.save('test_real.npy', real)
        np.save('test_output.npy', predicted)


def validation(args, val_loader, model, device):
    model.eval()
    val_loss = 0
    rmse = 0
    psnr = 0
    ssim = 0
    with torch.no_grad():
        for hr, lr, target in val_loader:
            hr_crop, lr_crop, target_crop = crop(hr, lr, target, int(args.crop_size), int(args.crop_size/2), int(args.crop_size))
            lr_crop, hr_crop, target_crop = lr_crop.to(device), hr_crop.to(device), target_crop.to(device)
            output = model(hr_crop, lr_crop)
            val_loss_function = nn.L1Loss(reduction='mean')
            val_loss += val_loss_function(output, target_crop).item()
            real = np.moveaxis(target_crop.cpu().numpy(), 1, 3) * 2000
            predicted = np.moveaxis(output.cpu().numpy(), 1, 3) * 2000
            for i in range(real.shape[0]):
                rmse += np.sqrt(skm.mean_squared_error(real[i], predicted[i]))
                psnr += skm.peak_signal_noise_ratio(real[i], predicted[i], data_range=real.max() - real.min())
                ssim += skm.structural_similarity(real[i], predicted[i], multichannel=True,
                                                  data_range=real.max() - real.min())

#     val_loss /= len(val_loader.dataset)
    rmse /= len(val_loader.dataset)
    psnr /= len(val_loader.dataset)
    ssim /= len(val_loader.dataset)

    print('\nValidation set: Average values --> Loss: {:.4f}, RMSE: ({:.2f}), PSNR: ({:.2f}dB),'
          ' SSIM: ({:.2f})\n'.format(val_loss, rmse, psnr, ssim))

    np.save('val_input.npy', (np.moveaxis(lr_crop.cpu().numpy() * 2000, 1, 3)))
    np.save('val_real.npy', real)
    np.save('val_output.npy', predicted)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch TFG Net')
    parser.add_argument('--batch-size', type=int, default=44, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=210, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--crop-size', type=int, default=128, metavar='N',
                        help='crop size (default: 128')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=3, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_loader = DataLoader(train_ds.dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    val_loader = DataLoader(val_ds.dataset, batch_size=args.test_batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(test_ds, batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.004)

    scheduler = StepLR(optimizer, step_size=5, gamma=args.gamma)
    model = model.type(dst_type=torch.float32)

    # get some random training images
    dataiter = iter(train_loader)
    hr, lr, target = dataiter.next()


    # visualize the model
    tb.add_graph(model, (hr, lr))

    for epoch in range(1, args.epochs + 1):
        train(args, train_loader, model, device, optimizer, epoch)
        if epoch % 5 == 0:
            validation(args, val_loader, model, device)
        test(args, test_loader, model, device, epoch)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "net.pt")

    tb.close()


if __name__ == '__main__':
    main()

