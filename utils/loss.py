import torch
import torch.nn as nn
import torch.nn.functional as F

def kd_loss_fn(outputs, labels, teacher_outputs, T, alpha):
    '''
    knowledge distillation loss with soft and hard targets [Distilling the Knowledge in a Neural Network]
    paper suggests scaling distillation loss by T^2 due to how soft targets are scaled 
    '''
    soft_loss = nn.KLDivLoss()( F.log_softmax(outputs/T, dim=1), F.softmax(teacher_outputs/T, dim=1))
    soft_loss = soft_loss * (alpha * T * T)
    hard_loss = F.cross_entropy(outputs, labels) * (1. - alpha)

    kd_loss = soft_loss + hard_loss
    return kd_loss


def unweight_kd_loss_fn(outputs, labels, teacher_outputs, T, alpha):
    '''
    knowledge distillation loss with soft and hard targets just adjusting distillation loss
    '''
    alpha = nn.Parameter(torch.ones(1))
    soft_loss = nn.KLDivLoss()( F.log_softmax(outputs/T, dim=1), F.softmax(teacher_outputs/T, dim=1))
    soft_loss = soft_loss * alpha
    hard_loss = F.cross_entropy(outputs, labels)

    kd_loss = soft_loss + hard_loss
    return kd_loss