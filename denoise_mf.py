import argparse
import logging
from signal import valid_signals
from PIL import Image 

import numpy as np
from skimage import io

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor
import tifffile
import random

import data, utils, models
import warnings
warnings.filterwarnings("ignore")

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def load_model(lr = 1e-4):
    args = argparse.Namespace(model='blind-video-net-4', 
                              channels=1, 
                              out_channels=1, 
                              bias=False, 
                              normal=False, 
                              blind_noise=False)
    
    model = models.build_model(args).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
    return model, optimizer

def read_image(path):
    image = Image.open(path)
    image = ToTensor()(image)
    return image

class DataSet(torch.utils.data.Dataset):
    def __init__(self, filename,image_size = None, transforms = False):
        super().__init__()
        self.x = image_size
        self.img = io.imread(filename)
        self.transforms = transforms

    def __len__(self):
        return self.img.shape[0]

    def __getitem__(self, index):
        if index < 2:
            out = np.concatenate((np.repeat(np.array([self.img[0]]), 2, axis=0), self.img[index:index+3]), axis=0)
        elif index > self.img.shape[0]-3:
            out = np.concatenate((self.img[index-2:index+1], np.repeat(np.array([self.img[-1]]), 2, axis=0)), axis=0)
        else:
            out = self.img[index-2:index+3]

        H, W = out.shape[-2:]
        
        if self.x is not None:
            h = np.random.randint(0, H-self.x)
            w = np.random.randint(0, W-self.x)
            out = out[:, h:h+self.x, w:w+self.x]
        
        if self.transforms:
            invert = random.choice([0, 1, 2])
            if invert == 1:
                out = out[:, :, ::-1]
            elif invert == 2:
                out = out[:, ::-1, :]

            rotate = random.choice([0, 1, 2, 3])
            if rotate != 0:
                out = np.rot90(out, rotate, (1, 2))
                
        out = np.copy(out)
        return torch.Tensor(np.float32(out)).to(device)


def main(args):
    data = args.data
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    print(device, args)
    

    model, optimizer = load_model()
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 30, 40], gamma=0.5)
    
    best_model = model.state_dict()
    best_loss = 1000000

    ds = DataSet(data, args.image_size, args.transforms)

    p = int(0.7*len(ds))
    valid, train = torch.utils.data.random_split(ds, [len(ds)-p, p], generator=torch.Generator().manual_seed(314))
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=True, num_workers=0)

    train_meters = {name: utils.RunningAverageMeter(0.98) for name in (["train_loss", "train_psnr", "train_ssim"])}
    valid_meters = {name: utils.AverageMeter() for name in (["valid_psnr", "valid_ssim"])}

    for epoch in range(num_epochs):
    
        for meter in train_meters.values():
            meter.reset()
        
        train_bar = utils.ProgressBar(train_loader, epoch)
        loss_avg, psnr_avg, ssim_avg, count = 0, 0, 0, 0
        for batch_id, inputs in enumerate(train_bar):
            model.train()

            frame = inputs[:,2].reshape((-1, 1, inputs.shape[-2], inputs.shape[-1])).to(device)
        
            inputs = inputs.to(device)

            outputs = model(inputs)
            loss = F.mse_loss(outputs, frame) / batch_size

            model.zero_grad()
            loss.backward()
            optimizer.step()
            train_psnr = utils.psnr(frame, outputs, False)
            train_ssim = utils.ssim(frame, outputs, False)
            train_meters["train_loss"].update(loss.item())
            train_meters["train_psnr"].update(train_psnr.item())
            train_meters["train_ssim"].update(train_ssim.item())
        
            loss_avg += loss.item()
            psnr_avg += train_psnr.item()
            ssim_avg += train_ssim.item()
            count += 1
        
            train_bar.log(dict(**train_meters, lr=optimizer.param_groups[0]["lr"]), verbose=True)
        scheduler.step()

        logging.info(train_bar.print(dict(**train_meters, lr=optimizer.param_groups[0]["lr"])))

        model.eval()
        for meter in valid_meters.values():
            meter.reset()

        valid_bar = utils.ProgressBar(valid_loader)
    
        loss_avg, psnr_avg, ssim_avg, count = 0, 0, 0, 0
        for sample_id, sample in enumerate(valid_bar):
            with torch.no_grad():
                sample = sample.to(device)
                frame = sample[:,2].reshape((-1, 1, inputs.shape[-2], inputs.shape[-1])).to(device)
                outputs = model(sample)


                valid_psnr = utils.psnr(frame, outputs, False)
                valid_ssim = utils.ssim(frame, outputs, False)
                valid_meters["valid_psnr"].update(valid_psnr.item())
                valid_meters["valid_ssim"].update(valid_ssim.item())
            
                loss_avg += F.mse_loss(outputs, frame) / batch_size
                psnr_avg += valid_psnr.item()
                ssim_avg += valid_ssim.item()
                count += 1
                
        if loss_avg/count < best_loss:
            best_loss = loss_avg/count
            best_model = model.state_dict()
    
    # Denoise the video
        
    ds = DataSet(data)
    denoised = np.zeros(ds.img.shape,dtype=np.float32)
    model.load_state_dict(best_model)
    model.eval()
    
    for k in range(len(ds)):
        with torch.no_grad():
            o  = model(ds[k].unsqueeze(0))
            o = o.cpu().numpy()
            denoised[k] = o
                    
    np.save(data[:-4]+'_udvd_mf'+'.npy', denoised)
                

        
    print('Denoised Prediction Saved at ', data[:-4]+'_udvd_mf' +'.npy')
    
    tensor_noisy = torch.Tensor(np.float32(ds.img)).unsqueeze(1)
    tensor_denoised = torch.Tensor(np.float32(denoised)).unsqueeze(1)
    uMSE, uPSNR = utils.uMSE_uPSNR(ds, model)

    print('MSE: ', utils.mse(tensor_noisy, tensor_denoised))    
    print('uMSE:', uMSE)
    print('uPSNR:', uPSNR)
    print('PSNR: ', utils.psnr(tensor_noisy, tensor_denoised))
    print('SSIM: ', utils.ssim(tensor_noisy, tensor_denoised))
        


def get_args():
    parser = argparse.ArgumentParser(allow_abbrev=False) 

    # Add data arguments
    parser.add_argument(
        "--data",
        default="data",
        help="path to .tif file to be denoised")
    parser.add_argument(
        "--num-epochs",
        default=50,
        type=int,
        help="epochs for the training")
    parser.add_argument(
        "--batch-size",
        default=1,
        type=int,
        help="train batch size")
    parser.add_argument(
        "--image-size",
        default=256,
        type=int,
        help="size of the patch")
    parser.add_argument(
        "--transforms",
        dest='feature',
        action='store_true')
    parser.add_argument(
        "--no-transforms",
        dest='feature', 
        action='store_false')
    parser.set_defaults(transforms=True)

    args, _ = parser.parse_known_args()
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    main(args)
