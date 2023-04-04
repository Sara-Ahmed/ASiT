import warnings
warnings.filterwarnings("ignore")

import argparse
import os
import sys
import datetime
import time
import math
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import torchvision

import dataloader
from data_transformations import DataAugmentation
from losses import CLSLoss, DATALoss

import utils
import vision_transformer as vits

from einops import rearrange

def get_args_parser():
    parser = argparse.ArgumentParser('ASiT', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='vit_base', type=str, choices=['vit_tiny', 'vit_small', 'vit_base'], help="architecture Name")
    parser.add_argument('--patch_size', default=16, type=int, help="Size of patch in pixels")
    
    # Local and Global Head parameters
    parser.add_argument('--out_dim_cls', default=8192, type=int, help="Dimensionality of the cls head output.")
    parser.add_argument('--out_dim_data', default=1024, type=int, help="Dimensionality of the data head output.")
    
    # Reconstruction parameters
    parser.add_argument('--drop_perc', type=float, default=0.7, help='Drop X percentage of the input spectrogram.')
    parser.add_argument('--drop_replace', type=float, default=0.3, help='Replace X percentage of the input spectrogram.')

    parser.add_argument('--drop_align', type=int, default=1, help='Align drop with patches.')
    parser.add_argument('--drop_type', type=str, default='zeros', help='Drop Type.')
    parser.add_argument('--drop_only', type=int, default=1, help='claculate the loss over only the corrupted patches.')
   
    # Temperature teacher parameters
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="Base EMA")
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float, help="""Initial value for the teacher temperature.""")
    parser.add_argument('--teacher_temp', default=0.07, type=float, help="""Final value (after linear warmup).""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int, help='Number of warmup epochs for the teacher temperature.')

    # Training/Optimization parameters
    parser.add_argument('--weight_decay', type=float, default=0.04, help="Initial value of the weight decay.")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="Final value of the weight decay. ")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="Maximal parameter gradient norm")
    parser.add_argument('--batch_size', default=32, type=int, help='Per-GPU batch-size')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. """)
        
        
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate.""")
    parser.add_argument("--warmup_epochs", default=10, type=int, help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the end of optimization. """)
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")

    # Dataset
    parser.add_argument("--data_path", type=str, default='/path/to/audio/files', help="dataset_poth")    
    parser.add_argument("--data-train", type=str, default='AUDIO_Files/AudioSet2M.json', help="training data json")
    parser.add_argument("--num_frames", default=592,type=int, help="the input length in frames")
    parser.add_argument("--num_mel_bins", type=int, default=128, help="number of input mel bins")
    parser.add_argument("--data_mean", type=float, default=-4.2677393, help="the dataset mean, used for input normalization")
    parser.add_argument("--data_std", type=float, default=4.5689974, help="the dataset std, used for input normalizations")
    
    parser.add_argument('--num_crops', type=int, default=2, help='number of seconds to crop during augmentation')
    parser.add_argument('--secs_per_crop', type=int, default=6, help='number of seconds to crop during augmentation')
    

    parser.add_argument('--output_dir', default="checkpoints/vit_base/AudioSet2M", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=5, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="url used to set up distributed training")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    return parser

def train_ASiT(args):
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ preparing data  ============
    transform = DataAugmentation(args)
    dataset = dataloader.AudioDataset(args.data_train, args.data_path, transform=transform)

    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, 
        sampler=sampler, num_workers=args.num_workers, pin_memory=True, drop_last=True,
        collate_fn=utils.collate_batch(args.drop_replace, args.drop_align))
    print(f"Data loaded: there are {len(dataset)} images.")

    # building student and teacher networks 
    student = vits.__dict__[args.arch](audio_size=[args.num_frames, args.num_mel_bins], in_chans=1, drop_path_rate=args.drop_path_rate)
    teacher = vits.__dict__[args.arch](audio_size=[args.num_frames, args.num_mel_bins], in_chans=1)
    embed_dim = student.embed_dim


    # Create full models
    student = FullPipeline(student, vits.CLSHead(embed_dim, args.out_dim_cls, args.out_dim_data), 
                                     vits.RECHead(embed_dim, [args.num_frames, args.num_mel_bins], patch_size=args.patch_size, in_chans=1) )

    teacher = FullPipeline(teacher, vits.CLSHead(embed_dim, args.out_dim_cls, args.out_dim_data), 
                                     vits.RECHead(embed_dim, [args.num_frames, args.num_mel_bins], patch_size=args.patch_size, in_chans=1))
    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()
    teacher_without_ddp = teacher

    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])
    teacher_without_ddp.load_state_dict(student.module.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.arch} network.")

    # preparing loss 
    loss_cls = CLSLoss(
        args.out_dim_cls, args.warmup_teacher_temp,
        args.teacher_temp, args.warmup_teacher_temp_epochs,
        args.epochs).cuda()
    
    loss_data = DATALoss(
        args.out_dim_data, args.warmup_teacher_temp,
        args.teacher_temp, args.warmup_teacher_temp_epochs,
        args.epochs).cuda()


    # preparing optimizer 
    params_groups = utils.get_params_groups(student)
    optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs

    # for mixed precision training
    fp16_scaler = torch.cuda.amp.GradScaler()

    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size * utils.get_world_size()) / 256.,  # linear scaling rule
        args.min_lr, args.epochs, len(data_loader), warmup_epochs=args.warmup_epochs)
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, len(data_loader))
    
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                               args.epochs, len(data_loader))
    print("Loss, optimizer and schedulers ready.")

    # resume training if checkpoint.pth exist
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
        loss_cls=loss_cls,
        loss_data=loss_data,
    )
    start_epoch = to_restore["epoch"]

    start_time = time.time()
    print("Starting training ...")
    for epoch in range(start_epoch, args.epochs):
        data_loader.sampler.set_epoch(epoch)

        # training one epoch 
        train_stats = train_one_epoch(student, teacher, teacher_without_ddp, loss_cls, loss_data,
            data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
            epoch, fp16_scaler, args)

        # writing logs 
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'loss_cls': loss_cls.state_dict(),
            'loss_data': loss_data.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(student, teacher, teacher_without_ddp, loss_cls, loss_data, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule,epoch,
                    fp16_scaler, args):
    
    save_recon = os.path.join(args.output_dir, 'reconstruction_samples')
    Path(save_recon).mkdir(parents=True, exist_ok=True)
    bz = args.batch_size
    plot_ = True

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    for it, ((clean_crops, corrupted_crops, masks_crops), _) in enumerate(metric_logger.log_every(data_loader, 100, header)):
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # move images to gpu
        clean_crops = [im.cuda(non_blocking=True) for im in clean_crops]
        corrupted_crops = [im.cuda(non_blocking=True) for im in corrupted_crops]
        masks_crops = [im.cuda(non_blocking=True) for im in masks_crops]

        with torch.cuda.amp.autocast(fp16_scaler is not None):
            
            t_cls, t_data, _ = teacher(clean_crops, recons=False)  
            s_cls, s_data, s_recons = student(corrupted_crops)
 
            clsloss = loss_cls(s_cls, t_cls, epoch) 
            dtaloss = loss_data(s_data, t_data, epoch) 
            
            rloss = F.l1_loss(s_recons, torch.cat(clean_crops[:2]), reduction='none')
            recloss = rloss[torch.cat(masks_crops[:2])==1].mean()
        
        
            #-------------------------------------------------
            if plot_==True and utils.is_main_process():
                plot_ = False
                #validating: check the reconstructed images
                print_out = save_recon + '/epoch_' + str(epoch).zfill(5)  + '.jpg'
                imagesToPrint = torch.cat([clean_crops[0][0: min(5, bz)].permute(0, 1, 3, 2).cpu(),  
                                           corrupted_crops[0][0: min(5, bz)].permute(0, 1, 3, 2).cpu(),
                                           s_recons[0: min(5, bz)].permute(0, 1, 3, 2).cpu(), 
                                           masks_crops[0][0: min(5, bz)].permute(0, 1, 3, 2).cpu()], dim=0)
                torchvision.utils.save_image(imagesToPrint, print_out, nrow=min(5, bz), normalize=True, range=(-1, 1))


        loss = clsloss + dtaloss + 2*recloss

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # student update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(clsloss=clsloss.item())
        metric_logger.update(dtaloss=dtaloss.item())
        metric_logger.update(recloss=recloss.item())
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class FullPipeline(nn.Module):
    def __init__(self, backbone, head_cls, head_recons):
        super(FullPipeline, self).__init__()
        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.backbone = backbone
        self.head_cls = head_cls
        self.head_recons = head_recons

    def forward(self, x, recons=True):
        
        _out = self.backbone(torch.cat(x[0:]))
            
        cls_out, data_out = self.head_cls(_out)
        data_out = rearrange(data_out, "b n c -> (b n) c", b=_out.size()[0])
        
        recons_ = self.head_recons(_out[:, 1:]) if recons==True else None

        return cls_out, data_out, recons_


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ASiT', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_ASiT(args)
