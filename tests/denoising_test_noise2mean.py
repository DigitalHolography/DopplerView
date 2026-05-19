import argparse
import h5py
import cv2
import numpy as np
import torch
import torch.nn as nn
from scipy.fft import fft2, ifft2, fftshift, ifftshift

# Simple U-Net architecture
class UNetSmall(nn.Module):
    def __init__(self, base_channels=32):
        super().__init__()
        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1), nn.ReLU(inplace=True),
            )
        self.e1, self.e2, self.e3 = block(1, base_channels), block(base_channels, base_channels*2), block(base_channels*2, base_channels*4)
        self.pool = nn.MaxPool2d(2)
        self.u2 = nn.ConvTranspose2d(base_channels*4, base_channels*2, 2, stride=2)
        self.d2 = block(base_channels*4, base_channels*2)
        self.u1 = nn.ConvTranspose2d(base_channels*2, base_channels, 2, stride=2)
        self.d1 = block(base_channels*2, base_channels)
        self.out = nn.Conv2d(base_channels, 1, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        x = self.d2(torch.cat([self.u2(e3), e2], 1))
        x = self.d1(torch.cat([self.u1(x), e1], 1))
        return torch.clamp(x - self.out(x), 0.0, 1.0) # Residual learning: input - noise

# Loss functions
class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        # Simplified SSIM - just L1 + gradient
        self.l1 = nn.L1Loss()
    
    def forward(self, pred, target):
        return 0.85 * (1 - self.ssim(pred, target)) + 0.15 * self.l1(pred, target)
    
    def ssim(self, pred, target):
        # Simplified SSIM approximation
        c1, c2 = 0.01**2, 0.03**2
        mu_x = torch.nn.functional.avg_pool2d(pred, 3, 1, 1)
        mu_y = torch.nn.functional.avg_pool2d(target, 3, 1, 1)
        sigma_x = torch.nn.functional.avg_pool2d(pred**2, 3, 1, 1) - mu_x**2
        sigma_y = torch.nn.functional.avg_pool2d(target**2, 3, 1, 1) - mu_y**2
        sigma_xy = torch.nn.functional.avg_pool2d(pred*target, 3, 1, 1) - mu_x*mu_y
        ssim_map = ((2*mu_x*mu_y + c1)*(2*sigma_xy + c2)) / ((mu_x**2 + mu_y**2 + c1)*(sigma_x + sigma_y + c2))
        return ssim_map.mean()

def get_loss_fn(loss_type):
    if loss_type == "mse":
        return nn.MSELoss()
    elif loss_type == "l1":
        return nn.L1Loss()
    elif loss_type == "smoothl1":
        return nn.SmoothL1Loss()
    elif loss_type == "ssim":
        return SSIMLoss()
    elif loss_type == "combined":
        def combined_loss(pred, target):
            return 0.5 * nn.L1Loss()(pred, target) + 0.5 * nn.MSELoss()(pred, target)
        return combined_loss
    else:
        return nn.SmoothL1Loss()

# Preprocessing functions
def preprocess_frame(frame, preproc_type):
    """Apply preprocessing to remove specific noise patterns"""
    if preproc_type == "none" or preproc_type is None:
        return frame
    
    elif preproc_type == "gaussian":
        return cv2.GaussianBlur(frame, (3, 3), 0)
    
    elif preproc_type == "median":
        return cv2.medianBlur((frame * 255).astype(np.uint8), 3).astype(np.float32) / 255
    
    elif preproc_type == "bilateral":
        return cv2.bilateralFilter(frame, 9, 75, 75)
    
    elif preproc_type == "hotpixel":
        # Remove hot pixels (>3 std above median)
        median = np.median(frame)
        std = np.std(frame)
        frame[frame > median + 3*std] = median
        return frame
    
    elif preproc_type == "highpass":
        # Remove low-frequency background drift
        f = fft2(frame)
        rows, cols = frame.shape
        crow, ccol = rows//2, cols//2
        fshift = fftshift(f)
        fshift[crow-5:crow+5, ccol-5:ccol+5] = 0  # Remove low freqs
        f_ishift = ifftshift(fshift)
        return np.real(ifft2(f_ishift))
    
    elif preproc_type == "anscombe":
        # Poisson noise stabilization
        return np.sqrt(np.maximum(frame + 0.375, 0))
    
    elif preproc_type == "wavelet":
        # Simple wavelet denoising (using DCT)
        img_float = frame.astype(np.float32)
        dct = cv2.dct(img_float)
        dct[np.abs(dct) < 0.1 * np.max(dct)] = 0
        return cv2.idct(dct)
    
    else:
        return frame

def train_and_export(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    
    # Load Data
    with h5py.File(args.input, "r") as f:
        dset = f[args.dataset]
        T, H, W = dset.shape
        data = dset[:].astype(np.float32)
        
        # Compute global min and max for normalization
        global_min = data.min()
        global_max = data.max()
        
        # Normalize the entire dataset to [0, 1]
        data = (data - global_min) / (global_max - global_min + 1e-8)
        
        # Apply preprocessing to all frames if specified
        if args.preprocess != "none":
            print(f"Applying preprocessing: {args.preprocess}")
            for i in range(T):
                data[i] = preprocess_frame(data[i], args.preprocess)
        
        # Compute mean image from normalized/preprocessed data
        mean_image = data.mean(axis=0)

    # Create model with adjustable channel size
    model = UNetSmall(base_channels=args.channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = get_loss_fn(args.loss)

    print(f"Training with {args.loss} loss, {args.preprocess} preprocessing...")
    
    for step in range(1, args.steps + 1):
        # Pick random image
        t = np.random.randint(0, T)
        
        if args.no_patch:
            # Use full frame
            noisy = torch.from_numpy(data[t][None, None]).to(device)
            target = torch.from_numpy(mean_image[None, None]).to(device)
        else:
            # Use random patch
            y = np.random.randint(0, H - args.patch)
            x = np.random.randint(0, W - args.patch)
            noisy = torch.from_numpy(data[t, y:y+args.patch, x:x+args.patch][None, None]).to(device)
            target = torch.from_numpy(mean_image[y:y+args.patch, x:x+args.patch][None, None]).to(device)

        # Optimization step
        optimizer.zero_grad()
        output = model(noisy)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"Step {step}/{args.steps} Loss: {loss.item():.6f}")

    # Export Videos
    print("Exporting...")
    model.eval()
    out_denoised = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H), isColor=False)
    out_noisy = cv2.VideoWriter(args.outputnoisy, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H), isColor=False)

    for i in range(T):
        # Process frame by frame
        frame = data[i]
        
        img_t = torch.from_numpy(frame[None, None]).to(device)
        with torch.no_grad():
            denoised = model(img_t).cpu().numpy()[0, 0]
            # denoised = (denoised - denoised.min()) / (denoised.max() - denoised.min())
            denoised = np.clip(denoised, 0, 1)

        out_noisy.write((frame * 255).astype(np.uint8))
        out_denoised.write((denoised * 255).astype(np.uint8))

    out_denoised.release()
    out_noisy.release()
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--dataset", default="/moment0")
    parser.add_argument("--output", default="denoised.mp4")
    parser.add_argument("--outputnoisy", default="noisy.mp4")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--patch", type=int, default=512)
    parser.add_argument("--no-patch", action="store_true", help="Use full frames instead of patches")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--channels", type=int, default=1, help="Base channels for UNet (default:32)")
    parser.add_argument("--loss", choices=["mse", "l1", "smoothl1", "ssim", "combined"], default="ssim", help="Loss function")
    parser.add_argument("--preprocess", choices=["none", "gaussian", "median", "bilateral", "hotpixel", "highpass", "anscombe", "wavelet"], default="none", help="Preprocessing to remove specific noise")
    
    train_and_export(parser.parse_args())