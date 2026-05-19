# noise2noise_h5_video.py
import argparse
import h5py
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class N2NDataset(Dataset):
    def __init__(self, h5_path, dataset="/moment0", patch=128, length=20000):
        self.h5_path = h5_path
        self.dataset = dataset
        self.patch = patch
        self.length = length

        with h5py.File(h5_path, "r") as f:
            self.shape = f[dataset].shape

        self.T, self.H, self.W = self.shape

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with h5py.File(self.h5_path, "r") as f:
            dset = f[self.dataset]

            i = np.random.randint(0, self.T)

            delta = np.random.randint(-4, 5)
            j = np.clip(i + delta, 0, self.T - 1)
            if j == i:
                j = (j + 1) % self.T

            y = np.random.randint(0, self.H - self.patch + 1)
            x = np.random.randint(0, self.W - self.patch + 1)

            a = dset[i, y:y+self.patch, x:x+self.patch].astype(np.float32)
            b = dset[j, y:y+self.patch, x:x+self.patch].astype(np.float32)

        a = normalize_(a)
        b = normalize_(b)

        return torch.from_numpy(a[None]), torch.from_numpy(b[None])


def normalize_(a):
    vmin = a.min()
    vmax = a.max()
    
    # vectorized normalization
    return ((a - vmin) / (vmax - vmin))


class UNetSmall(nn.Module):
    def __init__(self):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1),
                nn.ReLU(inplace=True),
            )

        self.e1 = block(1, 32)
        self.e2 = block(32, 64)
        self.e3 = block(64, 128)

        self.pool = nn.MaxPool2d(2)

        self.u2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.d2 = block(128, 64)

        self.u1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.d1 = block(64, 32)

        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, inp):
        e1 = self.e1(inp)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))

        x = self.u2(e3)
        x = self.d2(torch.cat([x, e2], dim=1))

        x = self.u1(x)
        x = self.d1(torch.cat([x, e1], dim=1))

        noise = self.out(x)
        return torch.clamp(inp - noise, 0.0, 1.0)


@torch.no_grad()
def denoise_video(
    model,
    h5_path,
    output_path,
    output_noisy_path,
    dataset="/moment0",
    fps=30,
    batch_size=4,
):
    device = next(model.parameters()).device
    model.eval()

    with h5py.File(h5_path, "r") as f:
        dset = f[dataset]
        T, H, W = dset.shape

        sample = dset[:min(T, 64)].astype(np.float32)
        lo = np.percentile(sample, 1)
        hi = np.percentile(sample, 99)
        scale = max(hi - lo, 1e-6)

        writer_denoised = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (W, H),
            isColor=False,
        )

        writer_noisy = cv2.VideoWriter(
            output_noisy_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (W, H),
            isColor=False,
        )

        for start in range(0, T, batch_size):
            end = min(start + batch_size, T)

            frames = dset[start:end].astype(np.float32)
            frames_norm = np.clip((frames - lo) / scale, 0, 1)

            x = torch.from_numpy(frames_norm[:, None]).to(device)
            y = model(x).cpu().numpy()[:, 0]

            for noisy_frame, denoised_frame in zip(frames_norm, y):
                noisy_u8 = np.clip(noisy_frame * 255, 0, 255).astype(np.uint8)
                denoised_u8 = np.clip(denoised_frame * 255, 0, 255).astype(np.uint8)

                writer_noisy.write(noisy_u8)
                writer_denoised.write(denoised_u8)

        writer_noisy.release()
        writer_denoised.release()


def train(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"Using device: {device}")

    dataset = N2NDataset(
        args.input,
        dataset=args.dataset,
        patch=args.patch,
        length=args.steps * args.batch_size,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(device == "cuda"),
    )

    model = UNetSmall().to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.SmoothL1Loss(beta=0.01)

    model.train()

    for step, (noisy_a, noisy_b) in enumerate(loader, start=1):
        noisy_a = noisy_a.to(device, non_blocking=True)
        noisy_b = noisy_b.to(device, non_blocking=True)

        pred = model(noisy_a)
        loss = loss_fn(pred, noisy_b)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 100 == 0:
            print(f"step {step:05d} / {args.steps}, loss = {loss.item():.6f}")

        if step >= args.steps:
            break

    denoise_video(
        model,
        args.input,
        args.output,
        args.outputnoisy,
        dataset=args.dataset,
        fps=args.fps,
        batch_size=args.infer_batch,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("input", help="Input H5 file")
    parser.add_argument("-o", "--output", default="denoised.mp4")
    parser.add_argument("--outputnoisy", default="noisy.mp4")
    parser.add_argument("--dataset", default="/moment0")
    parser.add_argument("--fps", type=int, default=30)

    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--infer-batch", type=int, default=4)
    parser.add_argument("--patch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cpu", action="store_true")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()