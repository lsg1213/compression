import argparse
import numpy as np
import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import torchaudio
import tqdm

from models import *
from metrics import *


parser = argparse.ArgumentParser()

# dataset parameters
parser.add_argument('--audio', type=str, default='all_mono.wav',
                    help='the path for the audio')
parser.add_argument('--name', required=True)
parser.add_argument('--out_folder', default='compressed',
                    help='folder to output images and model checkpoints')
parser.add_argument('--resume', action='store_true',
                    help='whether to continue from the saved checkpoint')
parser.add_argument('--overwrite', action='store_true',
                    help='overwrite the output dir if already exists')
parser.add_argument('--save', action='store_true')

# architecture parameters
parser.add_argument('--upscale', type=int, default=8)
parser.add_argument('--n_hidden_layers', type=int, default=4)
parser.add_argument('--hidden_dim', type=int, default=48)
parser.add_argument('--activation', type=str, default='gelu')
parser.add_argument('--n_bits', type=int, default=8)

# General training setups
parser.add_argument('--batch_size', type=int, default=30000)
parser.add_argument('--epochs', type=int, default=1500)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--grad_clip', type=float, default=1.)

# evaluation parameters
parser.add_argument('--eval_freq', type=int, default=250)


def get_cos_warmup_scheduler(optimizer, total_epoch, warmup_epoch):
    def lr_lambda(epoch):
        if epoch < warmup_epoch:
            return (epoch + 1) / (warmup_epoch + 1)
        return (1 + np.cos(np.math.pi * (epoch - warmup_epoch)
                           / (total_epoch - warmup_epoch))) / 2
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train(args):
    # 1. audio and grids
    audio, rate = torchaudio.load(args.audio)
    audio = audio.T # to [n_samples, channels]

    # 1.1 audio trimimng
    if len(audio) % rate:
        audio = audio[:-(len(audio) % rate)]
    assert len(audio) % args.upscale == 0
    audio = audio[:rate*10]
    n_samples = audio.shape[0]

    # 1.2 audio preprocessing
    audio = audio.reshape(-1, args.upscale).cuda()
    audio = audio / audio.abs().amax()

    # 1.3 make inputs
    n_channels = int(np.ceil(np.log(len(audio)) / np.log(2)))
    grids = torch.linspace(0, len(audio)-1, len(audio))
    grids = torch.stack([(grids % (2**(i+1))) / (2**(i+1)-1)
                         for i in range(n_channels)], -1) * 2 - 1
    '''
    # original POS ENC
    grids = torch.linspace(-1, 1, len(audio))
    grids = torch.stack([np.pi * grids * (2**i) for i in range(n_channels)], -1)
    grids = torch.cat([torch.cos(grids), torch.sin(grids)], -1)
    '''
    grids = grids.cuda()
    print(grids.shape)

    # 2. Model
    metrics = [PSNR(), PESQ(rate)]
    start_epoch = 0

    model = VINR(grids.shape[-1], args.upscale,
                 n_hidden_layers=args.n_hidden_layers,
                 hidden_dim=args.hidden_dim, activation=args.activation,
                 n_bits=args.n_bits)
    # model = Siren(grids.shape[-1], args.upscale,
    #               args.n_hidden_layers, args.hidden_dim)
    model = nn.DataParallel(model).cuda()
    # print(model)

    n_params = sum([p.numel() for p in model.parameters()])
    n_bits = model.module.get_bit_size() # 16 * n_params
    kbps = n_bits / (n_samples / rate) / 1000
    print(f'Model Params: {2*n_params/1e6:.4f}MB (kbps: {kbps:.4f})')
    print(f'avg bits per param: {n_bits / n_params:.4f}')

    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    scheduler = get_cos_warmup_scheduler(optimizer, args.epochs,
                                         int(0.2*args.epochs))
    scaler = torch.cuda.amp.GradScaler()

    # 3. Train
    model.train()
    best_score = 0

    for epoch in tqdm.tqdm(range(start_epoch, args.epochs)):
        # iterate over dataloader
        for i in torch.randperm(len(grids)).cuda().split(args.batch_size):
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=True):
                inputs = torch.index_select(grids, 0, i) # [B, H, W, 3]
                targets = torch.index_select(audio, 0, i) # [B, C, H, W]
                outputs = model(inputs)

                loss = F.mse_loss(outputs, targets)

                assert not torch.isnan(loss)
                scaler.scale(loss).backward(retain_graph=True)

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

        scheduler.step()

        # evaluation
        if (epoch + 1) % args.eval_freq == 0:
            scores = evaluate(model, grids, audio, metrics)
            print(scores)

            if best_score < scores[0]:
                best_score = scores[0]
                checkpoint = {'epoch': epoch+1,
                              'state_dict': model.state_dict(),
                              'optimizer': optimizer.state_dict()}
                torch.save(checkpoint, f'{args.out_folder}/best_score.pth')
            assert best_score < scores[0] + 5 # abrupt drop in performance

    checkpoint = {'epoch': epoch+1, 'state_dict': model.state_dict(),
                  'optimizer': optimizer.state_dict()}

    if args.save:
        torch.save(checkpoint, f'{args.out_folder}/latest.pth')


@torch.no_grad()
def evaluate(model, grids, audio, metrics):
    n_samples = len(audio)
    results = [0] * len(metrics)

    model.eval()

    with torch.cuda.amp.autocast(enabled=True):
        outputs = model(grids)

    print(outputs.max().cpu().numpy(), outputs.min().cpu().numpy())

    for j, m in enumerate(metrics):
        results[j] += m(outputs.reshape(-1).float(), audio.reshape(-1))

    model.train()

    return results


if __name__ == '__main__':
    args = parser.parse_args()

    print(vars(args))
    torch.set_printoptions(precision=5)

    args.out_folder = os.path.join(args.out_folder, args.name)

    if args.overwrite and os.path.isdir(args.out_folder):
        shutil.rmtree(args.out_folder)

    if not os.path.isdir(args.out_folder):
        os.makedirs(args.out_folder)

    with open(os.path.join(args.out_folder, 'config.cfg'), 'w') as o:
        o.write(str(vars(args)))

    train(args)

