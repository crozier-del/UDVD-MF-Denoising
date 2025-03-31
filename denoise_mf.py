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

def load_model(lr = 1e-4, is_fourdim=False, is_include_neighbor=False):
    if is_fourdim:
        if is_include_neighbor:
        # Use the specialized 4D model
            args = argparse.Namespace(model='blind-video-net-4-4d', 
                                    channels=1, 
                                    out_channels=1, 
                                    bias=False, 
                                    normal=False, 
                                    blind_noise=False)
        else:
            args = argparse.Namespace(model='blind-video-net-5-4d', 
                        channels=1, 
                        out_channels=1, 
                        bias=False, 
                        normal=False, 
                        blind_noise=False)
    else:
        if is_include_neighbor:
            args = argparse.Namespace(model='blind-video-net-4', 
                                    channels=1, 
                                    out_channels=1, 
                                    bias=False, 
                                    normal=False, 
                                    blind_noise=False)
        else:
            args = argparse.Namespace(model='blind-video-net-5', 
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
    def __init__(self, filename,image_size = None, transforms = False,multiply=1):
        super().__init__()
        self.x = image_size
        if filename.lower().endswith('.npy'):
            # Read the .npy file using numpy
            self.img = np.load(filename)*multiply
        else:
            # Read the file using skimage.io.imread
            self.img = io.imread(filename)*multiply
        self.transforms = transforms
        self.multiply = multiply

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
    
class DataSet4D(torch.utils.data.Dataset):
    def __init__(self, filename, image_size=None, transforms=False, multiply=1):
        super().__init__()
        self.x = image_size
        if filename.lower().endswith('.npy'):
            # Read the .npy file using numpy
            self.img = np.load(filename)*multiply
        else:
            # Read the file using skimage.io.imread
            self.img = io.imread(filename)*multiply
        self.transforms = transforms

    def __len__(self):
        return self.img.shape[0]*self.img.shape[1]

    def __getitem__(self, index):
        a_idx = index % self.img.shape[0]
        b_idx = index // self.img.shape[0]
        
        positions = [
            (a_idx-1, b_idx-1),  
            (a_idx-1, b_idx),    
            (a_idx-1, b_idx+1),  
            (a_idx, b_idx-1),    
            (a_idx, b_idx),      
            (a_idx, b_idx+1),    
            (a_idx+1, b_idx-1),  
            (a_idx+1, b_idx),    
            (a_idx+1, b_idx+1), 
        ]
        
        out = []
        
        center_frame = self.img[a_idx, b_idx]
        
        for i, j in positions:
            if 0 <= i < self.img.shape[0] and 0 <= j < self.img.shape[1]:
                out.append(self.img[i, j])
            else:
                out.append(center_frame)
        
        out = np.array(out)

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
    output_file = args.output_file
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    four_dim = args.fourdim
    include_neighbor = args.include_neighbor
    save_model = args.save_model
    
    if four_dim:
        center_f=4
    else:
        center_f=2
        
    if output_file:
        file_name = output_file
    else:                         
        file_name = data[:-4]+'_udvd_mf'
    
    print(device, args)
    
    model, optimizer = load_model(is_fourdim=four_dim, is_include_neighbor=include_neighbor)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 30, 40], gamma=0.5)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=0)
    
    best_model = model.state_dict()
    best_loss = 1000000
    
    if four_dim:
        ds = DataSet4D(data, args.image_size, args.transforms,args.multiply)
    else:
        ds = DataSet(data, args.image_size, args.transforms,args.multiply)

    p = int(0.7*len(ds))
    valid, train = torch.utils.data.random_split(ds, [len(ds)-p, p], generator=torch.Generator().manual_seed(314))
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=True, num_workers=0)

    train_meters = {name: utils.RunningAverageMeter(0.98) for name in (["train_loss", "train_psnr", "train_ssim"])}
    valid_meters = {name: utils.AverageMeter() for name in (["valid_psnr", "valid_ssim"])}
    
    best_epoch=0

    for epoch in range(num_epochs):
    
        for meter in train_meters.values():
            meter.reset()
        
        train_bar = utils.ProgressBar(train_loader, epoch)
        loss_avg, psnr_avg, ssim_avg, count = 0, 0, 0, 0
        for batch_id, inputs in enumerate(train_bar):
            model.train()

            frame = inputs[:,center_f].reshape((-1, 1, inputs.shape[-2], inputs.shape[-1])).to(device)
            
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
                frame = sample[:,center_f].reshape((-1, 1, inputs.shape[-2], inputs.shape[-1])).to(device)
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
            best_epoch = epoch
    
    # Denoise the video
        
    if four_dim:
        ds = DataSet4D(data, multiply=args.multiply)
    else:
        ds = DataSet(data, multiply=args.multiply)
        
    denoised = np.zeros(ds.img.shape,dtype=np.float32)
    model.load_state_dict(best_model)
    model.eval()
    
    if save_model:
        torch.save(model.state_dict(), file_name+'.pth')  
    
    if four_dim:
        length = ds.img.shape[0]
        for k in range(len(ds)):
            with torch.no_grad():
                a_idx = k % length
                b_idx = k // length
                o  = model(ds[k].unsqueeze(0))
                o = o.cpu().numpy()
                denoised[a_idx, b_idx] = o
    else:
        for k in range(len(ds)):
            with torch.no_grad():
                o  = model(ds[k].unsqueeze(0))
                o = o.cpu().numpy()
                denoised[k] = o

    np.save(file_name+'.npy', denoised)
    print('Denoised Prediction Saved at '+ file_name+'.npy')
    print(f'Best model is generated at epoch {best_epoch}.')
    
    # Denoised metrics

    raw_data = np.float32(ds.img)
    denoised_data = np.float32(denoised)
    
    if four_dim:
        raw_data = raw_data.reshape(-1, *raw_data.shape[2:])
        denoised_data = denoised_data.reshape(-1, *denoised_data.shape[2:])

    tensor_noisy = torch.Tensor(raw_data).unsqueeze(1)
    tensor_denoised = torch.Tensor(denoised_data).unsqueeze(1)
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
        help="path to dataset to be denoised")
    parser.add_argument(
        "--output-file",
        default="",
        help="set the name of output files w/o the extension")
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
    parser.add_argument(
        "--fourdim", 
        action="store_true",
        help="set if data is 4D")
    parser.add_argument(
        "--include-neighbor", 
        action="store_true",
        help="set if not blinding neighbors")
    parser.add_argument(
        "--save-model", 
        action="store_true",
        help="set if saving the model")
    parser.add_argument(
        "--multiply",
        default=1,
        type=int,
        help="multiply by an integer to manually normalize the data")
    parser.set_defaults(transforms=True)

    args, _ = parser.parse_known_args()
    args = parser.parse_args()
    return args
    


if __name__ == "__main__":
    args = get_args()
    main(args)
