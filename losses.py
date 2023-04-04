import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F

class CLSLoss(nn.Module):
    def __init__(self, out_dim, warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs, 
                 nepochs, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp))

    def forward(self, s_out, t_out, epoch):
        s_output = s_out / self.student_temp
        s_output = s_output.chunk(2)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        t_output = F.softmax((t_out - self.center) / temp, dim=-1)
        t_output = t_output.detach().chunk(2)

        loss1 = torch.sum(-t_output[0] * F.log_softmax(s_output[1], dim=-1), dim=-1)
        loss2 = torch.sum(-t_output[1] * F.log_softmax(s_output[0], dim=-1), dim=-1)
        
        loss = (loss1.mean() + loss2.mean()) / 2.0
        self.update_center(t_out)
        return loss

    @torch.no_grad()
    def update_center(self, t_out):
        batch_center = torch.sum(t_out, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(t_out) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
        
class DATALoss(nn.Module):
    def __init__(self, out_dim, warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs, 
                 nepochs, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp))

    def forward(self, s_out, t_out, epoch):
        s_output = s_out / self.student_temp

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        t_output = F.softmax((t_out - self.center) / temp, dim=-1)
        t_output = t_output.detach()
        
        loss = torch.sum(-t_output * F.log_softmax(s_output, dim=-1), dim=-1).mean()

        self.update_center(t_out)
        return loss

    @torch.no_grad()
    def update_center(self, t_out):
        batch_center = torch.sum(t_out, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(t_out) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
        


