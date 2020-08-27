import numpy as np
import os
import torch
from pathlib import Path


def ensure_dir(dirname):
    '''
    确定存在dirname文件夹，若不存在创建该文件夹
    :param dirname: 文件夹的绝对路径
    :return:
    '''
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)



def log_write(file, log):
    """[将log写入文件中]

    Args:
        file ([file]): [要存储的文件]
        log ([dict]): [要保存的log]
    """
    string = ''
    for key, value in log.items():
        string += key + '\t:'+str(value) +'\n'
    string += '\n\n\n'
    file.write(string)


def adjust_learning_rate(optimizer, epoch, lr_step, lr_decay, increase_bottom=0, increase_amp=1.1):
    """[自定义自适应学习率变化，退火，学习率先上升后下降]

    Args:
        optimizer ([type]): [pytoch官方优化器]
        epoch ([int]): [当前的epoch number]
        lr_step ([int]): [lr变化的步数]
        lr_decay ([float]): [lr变化的幅度]
        increase_bottom (int, optional): [学习率上升的上界]. Defaults to 0.
        increase_amp (float, optional): [学习率上升的幅度]. Defaults to 1.1
    Returns:
        optimizer ([type]): [pytoch官方优化器]
    """
    nowLR = optimizer.param_groups[0]['lr']
    LR = nowLR
    if epoch < increase_bottom:
        LR = LR * increase_amp
    elif(epoch - increase_bottom) % lr_step == 0:
        LR = LR * lr_decay
    
    for param_group in optimizer.param_groups:
        param_group['lr']=LR
    return optimizer


def trans_arr_to_one_hot(label, class_num):
    """[将一维列表转换为独热编码]

    Args:
        label [list[int]]: [类别列表]
        class_num [int]: [label可取的类别个数]
    Returns:
        onehot.numpy() [二维array] : [转换为onehot后的数据]
    """
    batch_size = label.shape[0]
    m_zeros = torch.zeros(batch_size, class_num)
    # 从 value 中取值，然后根据 dim 和 index 给相应位置赋值
    onehot = m_zeros.scatter_(1, label, 1)  # (dim,index,value)

    return onehot.numpy()  # Tensor -> Numpy


def trans_int_to_one_hot(label, class_num):
    """[将整数转换为独热编码]

    Returns:
        label [int]: [转换为onehot后的数据]
        class_num [int]: [label可取的类别个数]
    """
    onehot = torch.zeros(class_num)
    onehot[label] = 1

    return onehot.numpy()