import argparse
import os
import sys
import random
import timeit
import datetime
import copy 
import numpy as np
import pickle
import scipy.misc
from core.configs import cfg

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils import data, model_zoo
from torch.autograd import Variable
import torchvision.transforms as transform

from model.deeplabv2 import Res_Deeplab

from core.utils.prototype_dist_estimator import prototype_dist_estimator

from utils.loss import CrossEntropy2d
from utils.loss import CrossEntropyLoss2dPixelWiseWeighted
from utils.loss import MSELoss2d
from core.utils.loss import PrototypeContrastiveLoss

from utils import transformmasks
from utils import transformsgpu
from utils.helpers import colorize_mask
import utils.palette as palette

from utils.sync_batchnorm import convert_model
from utils.sync_batchnorm import DataParallelWithCallback

from data import get_loader, get_data_path
from data.augmentations import *
from tqdm import tqdm

import PIL
from torchvision import transforms
import json
from torch.utils import tensorboard
from evaluateUDA import evaluate

import time

start = timeit.default_timer()
start_writeable = datetime.datetime.now().strftime('%m-%d_%H-%M')

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--gpus", type=int, default=1,
                        help="choose number of gpu devices to use (default: 1)")
    parser.add_argument("-c", "--config", type=str, default='config.json',
                        help='Path to the config file (default: config.json)')
    parser.add_argument("-r", "--resume", type=str, default=None,
                        help='Path to the .pth file to resume from (default: None)')
    parser.add_argument("-n", "--name", type=str, default=None, required=True,
                        help='Name of the run (default: None)')
    parser.add_argument("--save-images", type=str, default=None,
                        help='Include to save images (default: None)')
    parser.add_argument("-cfg",
                        "--config-file",
                        default="",
                        metavar="FILE",
                        help="path to config file",
                        type=str)
    return parser.parse_args()


def loss_calc(pred, label):
    label = Variable(label.long()).cuda()
    if len(gpus) > 1:
        criterion = torch.nn.DataParallel(CrossEntropy2d(ignore_label=ignore_label), device_ids=gpus).cuda()  # Ignore label ??
    else:
        criterion = CrossEntropy2d(ignore_label=ignore_label).cuda()  # Ignore label ??

    return criterion(pred, label)

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))

def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(learning_rate, i_iter, num_iterations, lr_power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1 :
        optimizer.param_groups[1]['lr'] = lr * 10

def create_ema_model(model):
    #ema_model = getattr(models, config['arch']['type'])(self.train_loader.dataset.num_classes, **config['arch']['args']).to(self.device)
    ema_model = Res_Deeplab(num_classes=num_classes)

    for param in ema_model.parameters():
        param.detach_()
    mp = list(model.parameters())
    mcp = list(ema_model.parameters())
    n = len(mp)
    for i in range(0, n):
        mcp[i].data[:] = mp[i].data[:].clone()
    #_, availble_gpus = self._get_available_devices(self.config['n_gpu'])
    #ema_model = torch.nn.DataParallel(ema_model, device_ids=availble_gpus)
    if len(gpus)>1:
        #return torch.nn.DataParallel(ema_model, device_ids=gpus)
        if use_sync_batchnorm:
            ema_model = convert_model(ema_model)
            ema_model = DataParallelWithCallback(ema_model, device_ids=gpus)
        else:
            ema_model = torch.nn.DataParallel(ema_model, device_ids=gpus)
    return ema_model

def update_ema_variables(ema_model, model, alpha_teacher, iteration):
    # Use the "true" average until the exponential average is more correct
    alpha_teacher = min(1 - 1 / (iteration + 1), alpha_teacher)
    if len(gpus)>1:
        for ema_param, param in zip(ema_model.module.parameters(), model.module.parameters()):
            #ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
            ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    else:
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            #ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
            ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model

def strongTransform(parameters, data=None, target=None):
    assert ((data is not None) or (target is not None))
    data, target = transformsgpu.oneMix(mask = parameters["Mix"], data = data, target = target)
    data, target = transformsgpu.colorJitter(colorJitter = parameters["ColorJitter"], img_mean = torch.from_numpy(IMG_MEAN.copy()).cuda(), data = data, target = target)
    data, target = transformsgpu.gaussian_blur(blur = parameters["GaussianBlur"], data = data, target = target)
    data, target = transformsgpu.flip(flip = parameters["flip"], data = data, target = target)
    return data, target

def weakTransform(parameters, data=None, target=None):
    data, target = transformsgpu.flip(flip = parameters["flip"], data = data, target = target)
    return data, target

def getWeakInverseTransformParameters(parameters):
    return parameters

def getStrongInverseTransformParameters(parameters):
    return parameters

class DeNormalize(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, tensor):
        IMG_MEAN = torch.from_numpy(self.mean.copy())
        IMG_MEAN, _ = torch.broadcast_tensors(IMG_MEAN.unsqueeze(1).unsqueeze(2), tensor)
        tensor = tensor+IMG_MEAN
        tensor = (tensor/255).float()
        tensor = torch.flip(tensor,(0,))
        return tensor

class Learning_Rate_Object(object):
    def __init__(self,learning_rate):
        self.learning_rate = learning_rate

def save_image(image, epoch, id, palette):
    with torch.no_grad():
        if image.shape[0] == 3:
            restore_transform = transforms.Compose([
            DeNormalize(IMG_MEAN),
            transforms.ToPILImage()])


            image = restore_transform(image)
            #image = PIL.Image.fromarray(np.array(image)[:, :, ::-1])  # BGR->RGB
            image.save(os.path.join('dacs/', str(epoch)+ id + '.png'))
        else:
            mask = image.numpy()
            colorized_mask = colorize_mask(mask, palette)
            colorized_mask.save(os.path.join('dacs/', str(epoch)+ id + '.png'))

def _save_checkpoint(iteration, model, optimizer, config, ema_model, save_best=False, overwrite=True):
    checkpoint = {
        'iteration': iteration,
        'optimizer': optimizer.state_dict(),
        'config': config,
    }
    if len(gpus) > 1:
        checkpoint['model'] = model.module.state_dict()
        if train_unlabeled:
            checkpoint['ema_model'] = ema_model.module.state_dict()
    else:
        checkpoint['model'] = model.state_dict()
        if train_unlabeled:
            checkpoint['ema_model'] = ema_model.state_dict()

    if save_best:
        filename = os.path.join(checkpoint_dir, f'best_model.pth')
        torch.save(checkpoint, filename)
        print("Saving current best model: best_model.pth")
    else:
        filename = os.path.join(checkpoint_dir, f'checkpoint-iter{iteration}.pth')
        print(f'\nSaving a checkpoint: {filename} ...')
        torch.save(checkpoint, filename)
        if overwrite:
            try:
                os.remove(os.path.join(checkpoint_dir, f'checkpoint-iter{iteration - save_checkpoint_every}.pth'))
            except:
                pass

def _resume_checkpoint(resume_path, model, optimizer, ema_model):
    print(f'Loading checkpoint : {resume_path}')
    checkpoint = torch.load(resume_path)

    # Load last run info, the model params, the optimizer and the loggers
    iteration = checkpoint['iteration'] + 1
    print('Starting at iteration: ' + str(iteration))

    if len(gpus) > 1:
        model.module.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint['model'])

    optimizer.load_state_dict(checkpoint['optimizer'])

    if train_unlabeled:
        if len(gpus) > 1:
            ema_model.module.load_state_dict(checkpoint['ema_model'])
        else:
            ema_model.load_state_dict(checkpoint['ema_model'])

    return iteration, model, optimizer, ema_model

def prototype_dist_init(cfg, trainloader, model):
    feature_num = 2048
    feat_estimator = prototype_dist_estimator(feature_num=feature_num, cfg=cfg, res = 0)
    out_estimator = prototype_dist_estimator(feature_num=cfg.MODEL.NUM_CLASSES, cfg=cfg, res = 0)

    torch.cuda.empty_cache()

    iteration = 0
    model.eval()
    end = time.time()
    start_time = time.time()
    max_iters = len(trainloader)
    print("init")
    with torch.no_grad():
        for i, (src_input, src_label, _, _) in enumerate(trainloader):
            data_time = time.time() - end

            src_input = src_input.cuda(non_blocking=True)
            src_label = src_label.cuda(non_blocking=True).long()

            src_out, src_feat = model(src_input)
            B, N, Hs, Ws = src_feat.size()
            _, C, _, _ = src_out.size()

            # source mask: downsample the ground-truth label
            src_mask = F.interpolate(src_label.unsqueeze(0).float(), size=(Hs, Ws), mode='nearest').squeeze(0).long()
            src_mask = src_mask.contiguous().view(B * Hs * Ws, )

            # feature level
            src_feat = src_feat.permute(0, 2, 3, 1).contiguous().view(B * Hs * Ws, N)
            feat_estimator.update(features=src_feat.detach().clone(), labels=src_mask)

            # output level
            src_out = src_out.permute(0, 2, 3, 1).contiguous().view(B * Hs * Ws, C)
            out_estimator.update(features=src_out.detach().clone(), labels=src_mask)

            batch_time = time.time() - end
            end = time.time()
            #meters.update(time=batch_time, data=data_time)

            iteration = iteration + 1
            #eta_seconds = meters.time.global_avg * (max_iters - iteration)
            #eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))


            if iteration == max_iters:
                feat_estimator.save(name='prototype_feat_dist.pth')
                out_estimator.save(name='prototype_out_dist.pth')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    print("Total training time: {} ({:.4f} s / it)".format(total_time_str, total_time / max_iters))

def main():
    print(config)
    list_name = []
    best_mIoU = 0
    feature_num = 2048

    if consistency_loss == 'MSE':
        if len(gpus) > 1:
            unlabeled_loss =  torch.nn.DataParallel(MSELoss2d(), device_ids=gpus).cuda()
        else:
            unlabeled_loss =  MSELoss2d().cuda()
    elif consistency_loss == 'CE':
        if len(gpus) > 1:
            unlabeled_loss = torch.nn.DataParallel(CrossEntropyLoss2dPixelWiseWeighted(ignore_index=ignore_label), device_ids=gpus).cuda()
        else:
            unlabeled_loss = CrossEntropyLoss2dPixelWiseWeighted(ignore_index=ignore_label).cuda()

    cudnn.enabled = True

    # create network
    model = Res_Deeplab(num_classes=num_classes)
    
    # load pretrained parameters
    #saved_state_dict = torch.load(args.restore_from)
        # load pretrained parameters
    if restore_from[:4] == 'http' :
        saved_state_dict = model_zoo.load_url(restore_from)
    else:
        saved_state_dict = torch.load(restore_from)

    # Copy loaded parameters to model
    new_params = model.state_dict().copy()
    for name, param in new_params.items():
        if name in saved_state_dict and param.size() == saved_state_dict[name].size():
            new_params[name].copy_(saved_state_dict[name])
    model.load_state_dict(new_params)
    
    # init ema-model
    if train_unlabeled:
        ema_model = create_ema_model(model)
        ema_model.train()
        ema_model = ema_model.cuda()
    else:
        ema_model = None

    if len(gpus)>1:
        if use_sync_batchnorm:
            model = convert_model(model)
            model = DataParallelWithCallback(model, device_ids=gpus)
        else:
            model = torch.nn.DataParallel(model, device_ids=gpus)
    
    
    cudnn.benchmark = True
    pcl_criterion_src = PrototypeContrastiveLoss(cfg)
    pcl_criterion_tgt = PrototypeContrastiveLoss(cfg)
    
    if dataset == 'cityscapes':
        data_loader = get_loader('cityscapes')
        data_path = get_data_path('cityscapes')
        if random_crop:
            data_aug = Compose([RandomCrop_city(input_size)])
        else:
            data_aug = None

        #data_aug = Compose([RandomHorizontallyFlip()])
        train_dataset = data_loader(data_path, is_transform=True, augmentations=data_aug, img_size=input_size, img_mean = IMG_MEAN)

    train_dataset_size = len(train_dataset)
    print ('dataset size: ', train_dataset_size)

    if labeled_samples is None:
        trainloader = data.DataLoader(train_dataset,
                        batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

        trainloader_remain = data.DataLoader(train_dataset,
                        batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        trainloader_remain_iter = iter(trainloader_remain)

    else:
        partial_size = labeled_samples
        print('Training on number of samples:', partial_size)
        np.random.seed(random_seed)
        trainloader_remain = data.DataLoader(train_dataset,
                        batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)

        trainloader_remain_iter = iter(trainloader_remain)

    #New loader for Domain transfer
    a = ['21660.png', '06139.png', '06346.png', '13868.png', '06011.png', '19862.png', '14603.png', '02377.png', '20638.png', '15364.png', '03237.png', '21485.png', '16352.png', '17014.png', '19321.png', '00339.png', '15376.png', '04791.png', '16870.png', '12634.png', '23713.png', '20245.png', '19450.png', '23179.png', '17338.png', '16633.png', '22730.png', '23497.png', '02989.png', '14452.png', '02285.png', '03147.png', '11815.png', '21788.png', '15960.png', '17074.png', '13427.png', '13766.png', '15626.png', '04207.png', '21881.png', '06312.png', '21451.png', '05220.png', '06705.png', '18779.png', '17131.png', '18148.png', '08093.png', '13860.png', '24440.png', '07511.png', '10211.png', '01625.png', '23585.png', '24027.png', '12617.png', '07677.png', '24508.png', '05889.png', '06208.png', '16235.png', '05610.png', '02076.png', '13836.png', '08316.png', '07924.png', '08389.png', '07943.png', '23294.png', '24828.png', '10469.png', '19997.png', '23864.png', '05376.png', '18618.png', '19849.png', '21634.png', '03466.png', '24603.png', '02029.png', '15683.png', '12955.png', '23534.png', '02414.png', '13391.png', '10517.png', '14741.png', '24019.png', '21601.png', '00084.png', '20279.png', '00095.png', '24283.png', '14716.png', '01430.png', '03524.png', '02824.png', '13276.png', '18888.png', '24108.png', '04594.png', '15997.png', '04242.png', '01628.png', '07748.png', '24697.png', '06239.png', '24930.png', '05213.png', '03518.png', '22634.png', '09588.png', '08181.png', '11832.png', '11201.png', '08170.png', '22994.png', '20007.png', '24155.png', '23634.png', '16503.png', '07950.png', '11852.png', '23495.png', '18699.png', '07632.png', '15539.png', '18319.png', '00230.png', '17705.png', '21062.png', '02186.png', '15038.png', '00389.png', '22161.png', '21981.png', '06257.png', '16837.png', '24703.png', '14406.png', '14072.png', '13337.png', '20405.png', '08441.png', '22464.png', '00268.png', '15845.png', '12552.png', '22324.png', '18794.png', '16114.png', '03919.png', '12203.png', '05237.png', '10663.png', '00976.png', '07051.png', '14994.png', '11975.png', '16743.png', '20173.png', '24541.png', '21516.png', '09287.png', '20800.png', '11951.png', '04271.png', '00902.png', '19733.png', '09184.png', '14708.png', '15521.png', '09800.png', '12216.png', '20756.png', '02742.png', '10924.png', '07043.png', '02275.png', '06364.png', '07407.png', '03031.png', '09160.png', '06951.png', '21093.png', '17286.png', '24300.png', '11164.png', '12679.png', '18463.png', '22270.png', '05444.png', '11608.png', '17080.png', '21222.png', '23367.png', '19427.png', '00687.png', '24519.png', '14396.png', '23577.png', '08999.png', '00227.png', '07328.png', '08651.png', '14352.png', '09558.png', '07194.png', '18177.png', '02723.png', '08663.png', '10752.png', '01704.png', '07732.png', '10493.png', '02826.png', '24342.png', '07024.png', '22698.png', '12460.png', '24846.png', '12793.png', '14956.png', '03100.png', '21782.png', '16146.png', '09787.png', '18064.png', '22507.png', '23777.png', '08071.png', '14881.png', '02454.png', '09076.png', '15775.png', '17526.png', '09303.png', '05856.png', '07463.png', '05253.png', '22990.png', '22236.png', '04131.png', '03261.png', '24955.png', '06577.png', '13327.png', '02655.png', '13941.png', '12259.png', '04877.png', '07909.png', '16652.png', '04678.png', '21090.png', '16140.png', '05677.png', '21996.png', '16888.png', '20966.png', '20927.png', '05909.png', '09299.png', '00848.png', '09476.png', '22198.png', '09320.png', '15534.png', '09584.png', '16476.png', '11077.png', '13760.png', '01600.png', '11176.png', '21283.png', '10796.png', '00736.png', '04912.png', '01504.png', '20360.png', '07250.png', '23802.png', '09553.png', '14119.png', '13281.png', '15934.png', '20528.png', '17196.png', '07582.png', '08462.png', '15744.png', '18679.png', '19664.png', '06920.png', '17920.png', '24847.png', '11334.png', '00983.png', '07390.png', '01821.png', '09677.png', '19968.png', '13413.png', '18382.png', '04507.png', '01336.png', '14607.png', '24495.png', '18830.png', '06104.png', '05916.png', '06903.png', '09101.png', '07901.png', '11183.png', '03436.png', '11728.png', '16773.png', '12188.png', '01030.png', '15427.png', '13201.png', '10692.png', '16000.png', '21644.png', '07799.png', '03273.png', '01150.png', '00676.png', '10799.png', '16336.png', '09096.png', '11972.png', '13477.png', '20248.png', '17553.png', '18012.png', '13851.png', '04363.png', '23017.png', '00549.png', '10462.png', '00716.png', '20453.png', '20367.png', '14153.png', '11974.png', '24475.png', '18626.png', '13245.png', '09473.png', '12123.png', '11051.png', '18277.png', '12644.png', '13863.png', '15664.png', '19403.png', '02750.png', '07167.png', '00102.png', '17382.png', '19276.png', '06265.png', '17189.png', '09437.png', '12084.png', '03937.png', '03275.png', '04840.png', '08230.png', '14478.png', '07075.png', '17508.png', '06366.png', '14580.png', '04224.png', '17234.png', '21002.png', '05479.png', '11496.png', '00221.png', '24480.png', '15324.png', '06670.png', '17564.png', '08374.png', '19927.png', '03739.png', '19659.png', '10492.png', '24821.png', '18546.png', '21097.png', '20444.png', '06020.png', '18334.png', '00738.png', '07404.png', '11925.png', '18584.png', '02171.png', '08097.png', '10235.png', '09760.png', '06407.png', '23163.png', '08772.png', '18566.png', '18291.png', '00702.png', '14488.png', '13200.png', '10803.png', '12645.png', '11075.png', '22782.png', '10508.png', '11260.png', '10704.png', '08061.png', '14916.png', '16089.png', '02465.png', '06681.png', '16484.png', '17740.png', '17413.png', '17239.png', '19149.png', '15366.png', '11794.png', '20408.png', '03305.png', '03075.png', '14749.png', '14182.png', '12149.png', '16255.png', '20229.png', '02749.png', '09426.png', '10439.png', '08294.png', '18272.png', '21212.png', '06549.png', '18917.png', '21667.png', '12809.png', '15910.png', '06482.png', '06651.png', '15451.png', '04453.png', '18411.png', '17930.png', '20106.png', '07436.png', '10990.png', '04980.png', '06724.png', '04637.png', '09634.png', '06433.png', '04735.png', '03214.png', '00070.png', '01068.png', '23682.png', '07667.png', '11387.png', '21416.png', '20749.png', '18297.png', '06910.png', '16347.png', '00106.png', '11305.png', '21647.png', '07761.png', '18417.png', '08927.png', '15382.png', '04008.png', '04189.png', '03397.png', '01994.png', '03303.png', '01773.png', '04795.png', '01200.png', '10610.png', '02911.png', '16477.png', '15438.png', '03448.png', '12647.png', '06802.png', '00422.png', '04118.png', '08060.png', '06184.png', '22460.png', '02147.png', '16996.png', '00858.png', '14063.png', '21015.png', '13394.png', '04836.png', '19535.png', '20031.png', '09662.png', '02962.png', '21889.png', '03341.png', '08522.png', '23226.png', '13031.png', '09808.png', '16110.png', '16566.png', '15678.png', '04136.png', '20852.png', '15242.png', '16439.png', '11261.png', '10440.png', '02138.png', '12100.png', '03721.png', '01472.png', '07723.png', '07883.png', '04672.png', '05198.png', '08761.png', '20170.png', '17085.png', '09910.png', '18577.png', '02060.png', '24129.png', '20064.png', '05653.png', '19520.png', '03180.png', '15284.png', '10182.png', '04596.png', '11710.png', '11058.png', '02966.png', '13122.png', '19926.png', '23148.png', '19792.png', '15740.png', '01887.png', '19873.png', '04358.png', '23966.png', '05645.png', '01609.png', '22087.png', '18663.png', '12083.png', '14012.png', '13399.png', '17753.png', '16658.png', '14462.png', '23512.png', '11929.png', '10037.png', '10409.png', '18904.png', '04248.png', '23243.png', '23749.png', '16014.png', '22578.png', '03278.png', '10657.png', '14055.png', '00489.png', '01961.png', '21293.png', '02154.png', '00690.png', '20552.png', '12244.png', '09566.png', '15979.png', '07158.png', '01639.png', '20555.png', '09540.png', '21213.png', '15504.png', '14073.png', '07821.png', '08197.png', '08769.png', '06795.png', '02106.png', '07620.png', '00363.png', '04694.png', '18879.png', '14062.png', '18575.png', '09654.png', '22627.png', '13385.png', '09480.png', '17819.png', '04504.png', '08008.png', '07031.png', '08530.png', '04505.png', '01526.png', '18111.png', '21869.png', '14317.png', '10370.png', '04057.png', '16186.png', '11749.png', '03495.png', '12983.png', '09774.png', '07948.png', '06038.png', '09259.png', '16517.png', '06083.png', '07331.png', '17523.png', '04382.png', '23156.png', '02804.png', '21137.png', '03948.png', '02401.png', '14914.png', '13904.png', '08638.png', '13339.png', '16742.png', '22009.png', '16595.png', '02021.png', '04643.png', '24844.png', '08013.png', '05712.png', '24816.png', '22356.png', '12577.png', '16062.png', '17510.png', '05257.png', '18579.png', '07968.png', '14354.png', '14318.png', '01393.png', '18948.png', '12378.png', '05287.png', '21586.png', '24857.png', '14279.png', '15350.png', '16792.png', '09710.png', '20330.png', '15444.png', '04162.png', '07490.png', '11807.png', '22506.png', '05044.png', '02956.png', '24852.png', '10228.png', '12963.png', '02143.png', '23144.png', '07868.png', '17850.png', '07647.png', '10902.png', '24007.png', '15032.png', '22862.png', '13644.png', '24599.png', '02293.png', '14520.png', '12235.png', '15556.png', '09254.png', '15090.png', '10063.png', '19102.png', '24452.png', '24286.png', '00027.png', '23316.png', '00245.png', '14166.png', '16340.png', '22169.png', '04845.png', '05983.png', '10196.png', '06659.png', '11374.png', '03176.png', '03663.png', '11137.png', '01128.png', '18707.png', '00567.png', '20970.png', '15165.png', '13075.png', '16497.png', '19630.png', '04178.png', '08364.png', '17744.png', '22737.png', '21807.png', '01009.png', '14802.png', '21756.png', '03921.png', '15799.png', '11195.png', '15636.png', '02469.png', '06125.png', '03276.png', '19706.png', '08470.png', '05451.png', '23755.png', '04299.png', '19358.png', '23449.png', '02893.png', '22968.png', '13420.png', '00057.png', '17784.png', '15569.png', '03935.png', '20247.png', '18281.png', '08185.png', '01489.png', '15220.png', '24937.png', '20004.png', '09388.png', '18345.png', '15322.png', '22914.png', '04146.png', '09474.png', '02513.png', '09889.png', '13902.png', '07574.png', '15633.png', '14504.png', '21527.png', '24845.png', '03735.png', '24647.png', '05817.png', '23967.png', '24517.png', '17173.png', '06160.png', '15916.png', '09471.png', '23346.png', '21502.png', '06959.png', '10477.png', '21811.png', '02657.png', '14955.png', '03500.png', '17759.png', '12349.png', '08117.png', '07375.png', '14966.png', '01933.png', '03693.png', '21496.png', '01024.png', '13118.png', '01190.png', '16545.png', '22279.png', '21220.png', '00114.png', '02319.png', '02114.png', '18422.png', '03946.png', '20603.png', '23360.png', '14947.png', '03681.png', '06423.png', '02097.png', '01082.png', '03289.png', '03584.png', '01977.png', '22203.png', '23709.png', '17807.png', '09348.png', '10937.png', '23355.png', '02589.png', '17187.png', '04325.png', '12629.png', '06671.png', '17213.png', '01632.png', '16965.png', '10392.png', '06953.png', '08192.png', '15857.png', '00092.png', '05934.png', '06762.png', '12204.png', '23274.png', '13403.png', '04397.png', '00068.png', '07697.png', '12376.png', '11191.png', '13810.png', '03474.png', '04477.png', '10840.png', '08704.png', '20201.png', '23275.png', '19704.png', '16505.png', '08988.png', '06776.png', '22995.png', '12761.png', '10327.png', '04018.png', '01857.png', '03917.png', '21494.png', '03902.png', '16152.png', '02064.png', '13619.png', '04243.png', '24145.png', '23332.png', '23697.png', '22483.png', '18488.png', '02402.png', '17651.png', '03432.png', '22900.png', '02561.png', '07394.png', '23627.png', '18321.png', '11014.png', '10513.png', '19999.png', '19539.png', '04581.png', '19148.png', '13959.png', '14629.png', '21383.png', '03598.png', '07368.png', '11377.png', '12294.png', '02595.png', '01255.png', '05609.png', '12775.png', '07106.png', '16705.png', '20723.png', '16575.png', '23553.png', '05228.png', '16100.png', '22013.png', '15568.png', '14132.png', '13813.png', '17640.png', '19835.png', '12137.png', '16399.png', '15688.png', '16153.png', '22817.png', '18741.png', '03124.png', '15358.png', '18049.png', '09475.png', '01139.png', '10291.png', '08915.png', '03764.png', '13017.png', '19137.png', '14351.png', '17876.png', '12528.png', '09601.png', '02807.png', '10254.png', '01237.png', '21641.png', '04399.png', '00124.png', '11059.png', '08005.png', '11105.png', '24784.png', '18312.png', '21784.png', '05776.png', '22754.png', '05323.png', '12153.png', '22146.png', '12572.png', '06340.png', '15458.png', '01135.png', '18654.png', '17079.png', '15949.png', '18964.png', '17092.png', '12886.png', '04576.png', '13307.png', '17065.png', '01000.png', '00965.png', '14407.png', '18692.png', '21816.png', '22004.png', '21277.png', '13837.png', '09022.png', '06444.png', '23229.png', '21750.png', '24326.png', '07003.png', '00426.png', '18627.png', '21278.png', '03241.png', '02907.png', '08578.png', '06572.png', '05844.png', '18552.png', '16278.png', '22790.png', '12908.png', '07896.png', '22244.png', '24454.png', '17416.png', '20203.png', '19622.png', '05391.png', '20932.png', '10670.png', '03265.png', '11208.png', '11909.png', '23839.png', '04626.png', '06979.png', '16096.png', '01302.png', '07182.png', '20493.png', '10027.png', '00739.png', '19639.png', '19175.png', '11542.png', '04259.png', '18516.png', '21453.png', '07289.png', '07996.png', '17473.png', '21852.png', '23212.png', '19407.png', '11682.png', '20641.png', '10279.png', '17817.png', '22949.png', '12025.png', '18656.png', '11679.png', '17698.png', '05925.png', '07706.png', '10829.png', '15012.png', '11082.png', '23430.png', '02917.png', '18464.png', '03741.png', '01325.png', '22677.png', '09807.png', '20701.png', '02269.png', '08080.png', '22928.png', '05747.png', '04698.png', '05544.png', '10736.png', '14309.png', '06176.png', '22916.png', '12864.png', '00262.png', '17215.png', '10427.png', '13513.png', '23664.png', '06688.png', '22563.png', '12722.png', '10848.png', '06693.png', '21849.png', '18721.png', '10042.png', '07927.png', '12986.png', '15239.png', '04273.png', '20958.png', '06412.png', '17995.png', '14772.png', '00653.png', '11266.png', '18021.png', '02142.png', '17699.png', '04902.png', '03533.png', '22815.png', '05069.png', '08133.png', '23962.png', '06942.png', '09313.png', '19952.png', '00450.png', '13146.png', '20994.png', '12495.png', '00681.png', '21936.png', '24106.png', '18725.png', '01611.png', '17713.png', '09070.png', '11427.png', '05660.png', '03469.png', '16291.png', '10059.png', '17710.png', '07829.png', '13811.png', '12341.png', '21249.png', '03870.png', '18524.png', '01206.png', '20755.png', '09283.png', '09280.png', '09717.png', '23357.png', '04653.png', '19724.png', '19528.png', '18568.png', '19435.png', '03354.png', '16653.png', '11323.png', '10400.png', '03652.png', '21349.png', '11624.png', '09305.png', '21614.png', '11049.png', '18435.png', '18859.png', '19524.png', '00434.png', '13301.png', '05248.png', '13555.png', '11392.png', '03053.png', '08713.png', '18001.png', '02763.png', '14643.png', '02274.png', '13034.png', '00383.png', '22119.png', '04864.png', '04639.png', '16638.png', '15686.png', '07254.png', '22174.png', '02031.png', '24282.png', '01264.png', '19051.png', '19368.png', '16868.png', '16071.png', '10996.png', '02041.png', '10684.png', '04522.png', '19790.png', '11256.png', '17045.png', '08474.png', '05223.png', '08081.png', '15300.png', '06561.png', '04890.png', '10981.png', '10600.png', '12318.png', '08671.png', '06559.png', '23281.png', '10802.png', '07355.png', '00649.png', '14605.png', '16829.png', '13821.png', '08227.png', '02681.png', '11316.png', '22803.png', '02736.png', '12833.png', '23302.png', '15988.png', '01674.png', '17415.png', '21314.png', '19739.png', '00773.png', '05190.png', '17042.png', '02424.png', '21804.png', '04869.png', '15088.png', '02829.png', '21147.png', '19608.png', '04293.png', '24803.png', '20366.png', '02022.png', '11709.png', '10002.png', '06774.png', '03089.png', '12196.png', '14242.png', '06875.png', '22035.png', '24462.png', '07797.png', '10207.png', '00169.png', '24428.png', '00001.png', '03503.png', '16679.png', '04793.png', '02395.png', '18979.png', '20785.png', '16115.png', '10393.png', '24307.png', '22432.png', '02378.png', '23756.png', '19861.png', '07272.png', '01983.png', '12753.png', '12146.png', '10769.png', '16528.png', '22547.png', '21761.png', '09593.png', '14862.png', '02871.png', '00995.png', '18080.png', '13939.png', '13310.png', '24347.png', '12820.png', '22667.png', '00412.png', '04038.png', '00129.png', '08776.png', '14038.png', '04974.png', '17728.png', '11033.png', '11545.png', '20643.png', '24754.png', '17248.png', '20464.png', '16386.png', '13504.png', '07200.png', '02975.png', '23515.png', '15080.png', '16464.png', '10025.png', '20151.png', '12988.png', '20456.png', '01171.png', '23770.png', '12555.png', '13212.png', '17028.png', '17580.png', '06830.png', '23609.png', '07795.png', '20868.png', '13769.png', '15055.png', '13945.png', '20141.png', '02090.png', '10278.png', '10639.png', '21894.png', '02067.png', '09885.png', '11121.png', '01886.png', '13487.png', '21135.png', '05842.png', '20825.png', '21567.png', '24329.png', '07983.png', '04591.png', '00587.png', '22178.png', '19960.png', '23246.png', '19451.png', '17141.png', '09019.png', '21728.png', '24529.png', '15422.png', '09093.png', '20665.png', '19317.png', '08635.png', '11666.png', '18142.png', '03930.png', '13057.png', '06021.png', '01734.png', '04440.png', '05583.png', '11513.png', '02816.png', '06405.png', '05239.png', '07691.png', '10071.png', '02258.png', '05843.png', '23867.png', '09707.png', '17163.png', '03736.png', '09716.png', '18326.png', '04010.png', '03102.png', '11286.png', '05944.png', '06914.png', '22370.png', '22049.png', '07157.png', '12281.png', '16632.png', '05799.png', '00962.png', '18527.png', '14368.png', '14585.png', '06750.png', '10443.png', '01134.png', '19319.png', '21677.png', '08730.png', '14584.png', '20904.png', '19042.png', '16074.png', '17723.png', '10746.png', '02782.png', '19361.png', '06310.png', '21874.png', '02604.png', '13319.png', '22229.png', '22603.png', '04833.png', '07138.png', '22650.png', '12723.png', '05269.png', '00619.png', '06641.png', '04133.png', '24622.png', '23790.png', '24039.png', '03044.png', '17311.png', '15547.png', '16239.png', '00622.png', '12867.png', '13524.png', '06131.png', '04037.png', '17629.png', '10357.png', '09231.png', '02110.png', '00923.png', '02887.png', '07262.png', '07570.png', '11107.png', '19382.png', '04182.png', '10020.png', '09670.png', '23149.png', '23131.png', '07974.png', '19602.png', '22453.png', '18443.png', '08739.png', '18640.png', '15061.png', '22237.png', '23650.png', '21522.png', '00623.png', '21443.png', '08963.png', '22232.png', '08908.png', '21134.png', '10075.png', '21554.png', '12863.png', '20181.png', '16537.png', '00864.png', '16272.png', '20074.png', '13501.png', '11773.png', '16277.png', '01738.png', '07388.png', '22310.png', '01307.png', '03715.png', '14828.png', '01848.png', '06677.png', '23861.png', '03679.png', '06708.png', '16858.png', '17889.png', '24269.png', '03683.png', '12410.png', '24951.png', '05252.png', '12415.png', '07473.png', '15492.png', '20874.png', '11475.png', '21476.png', '23107.png', '09268.png', '12889.png', '21551.png', '02612.png', '05675.png', '10911.png', '07223.png', '06144.png', '00651.png', '03336.png', '03096.png', '05814.png', '23304.png', '08482.png', '05864.png', '02166.png', '17441.png', '09012.png', '12035.png', '10241.png', '24160.png', '05064.png', '09555.png', '10470.png', '20292.png', '14724.png', '05628.png', '18548.png', '15795.png', '22077.png', '18124.png', '15942.png', '20705.png', '09955.png', '00647.png', '11120.png', '13158.png', '22307.png', '04355.png', '16624.png', '08944.png', '04154.png', '08838.png', '12688.png', '17769.png', '18762.png', '21875.png', '10599.png', '05025.png', '09131.png', '05497.png', '16413.png', '20435.png', '13642.png', '16539.png', '19660.png', '11126.png', '16151.png', '05166.png', '21149.png', '14910.png', '21280.png', '23265.png', '14064.png', '12326.png', '22814.png', '08119.png', '19770.png', '01785.png', '19267.png', '21603.png', '20972.png', '18351.png', '00943.png', '19676.png', '18449.png', '03713.png', '11956.png', '12806.png', '08377.png', '09745.png', '02639.png', '10626.png', '07383.png', '10361.png', '19334.png', '01681.png', '08542.png', '11226.png', '05305.png', '23617.png', '09938.png', '15733.png', '19083.png', '07792.png', '04106.png', '17990.png', '13978.png', '03309.png', '14657.png', '15952.png', '17632.png', '06247.png', '17863.png', '03539.png', '12876.png', '12399.png', '03610.png', '13626.png', '22260.png', '03831.png', '22536.png', '19073.png', '23747.png', '19248.png', '14691.png', '05749.png', '16654.png', '13801.png', '05364.png', '20018.png', '21226.png', '23041.png', '02071.png', '06786.png', '10623.png', '00336.png', '04875.png', '07849.png', '15465.png', '08371.png', '19454.png', '11154.png', '16189.png', '03762.png', '05911.png', '11599.png', '17083.png', '19874.png', '14529.png', '04962.png', '06877.png', '24894.png', '05450.png', '23459.png', '14248.png', '05962.png', '18155.png', '23022.png', '16749.png', '11006.png', '20411.png', '19522.png', '12327.png', '09882.png', '07408.png', '19773.png', '04001.png', '03230.png', '23098.png', '14134.png', '13361.png', '07939.png', '03401.png', '04752.png', '10625.png', '14685.png', '22123.png', '15057.png', '13899.png', '24500.png', '15957.png', '08519.png', '14938.png', '05831.png', '10028.png', '20187.png', '08217.png', '09629.png', '22844.png', '24161.png', '22919.png', '05124.png', '13637.png', '18680.png', '07080.png', '14734.png', '24492.png', '05849.png', '12445.png', '21148.png', '09514.png', '24309.png', '20987.png', '22501.png', '23741.png', '00324.png', '19052.png', '06194.png', '03855.png', '14973.png', '01577.png', '16006.png', '21264.png', '11314.png', '15525.png', '05214.png', '00578.png', '13940.png', '06868.png', '05207.png', '16794.png', '19590.png', '19917.png', '08618.png', '23492.png', '13973.png', '20035.png', '15195.png', '04100.png', '13269.png', '10398.png', '01980.png', '17976.png', '01464.png', '11696.png', '09779.png', '03957.png', '17989.png', '19828.png', '14593.png', '21455.png', '20379.png', '14170.png', '23466.png', '13259.png', '16206.png', '04687.png', '16815.png', '01902.png', '03537.png', '16216.png', '12282.png', '06506.png', '03689.png', '17507.png', '00456.png', '08933.png', '00257.png', '16504.png', '01310.png', '16867.png', '19698.png', '20557.png', '06552.png', '21662.png', '08537.png', '23540.png', '23135.png', '11348.png', '05433.png', '16203.png', '10669.png', '13317.png', '00166.png', '13325.png', '16970.png', '14397.png', '13669.png', '15700.png', '10749.png', '24696.png', '16019.png', '24514.png', '13894.png', '20492.png', '00565.png', '10992.png', '02225.png', '09806.png', '08718.png', '15528.png', '19988.png', '04928.png', '04843.png', '10795.png', '16104.png', '17843.png', '18808.png', '12931.png', '13572.png', '20394.png', '22780.png', '23565.png', '09250.png', '03556.png', '09338.png', '21434.png', '02006.png', '24042.png', '06865.png', '05140.png', '19946.png', '16430.png', '03860.png', '04635.png', '18219.png', '11869.png', '11327.png', '02805.png', '18554.png', '15937.png', '03445.png', '01081.png', '05717.png', '10339.png', '20531.png', '08452.png', '00387.png', '09018.png', '07853.png', '06964.png', '13283.png', '21201.png', '09063.png', '15346.png', '16850.png', '12689.png', '17154.png', '19089.png', '10908.png', '12218.png', '23882.png', '04249.png', '12033.png', '05630.png', '06470.png', '19196.png', '24012.png', '07722.png', '07327.png', '15429.png', '08871.png', '01322.png', '13013.png', '11641.png', '16721.png', '12874.png', '08664.png', '24764.png', '07411.png', '02699.png', '07418.png', '23168.png', '12768.png', '10864.png', '21658.png', '18250.png', '11114.png', '02971.png', '16981.png', '13995.png', '20831.png', '15753.png', '22193.png', '04059.png', '17612.png', '02794.png', '11353.png', '08250.png', '13649.png', '13592.png', '15518.png', '18565.png', '00228.png', '17058.png', '01014.png', '18727.png', '17793.png', '06453.png', '14090.png', '18048.png', '16403.png', '20963.png', '08273.png', '20922.png', '05532.png', '01597.png', '22374.png', '14756.png', '00939.png', '19772.png', '22793.png', '21076.png', '08053.png', '13223.png', '16292.png', '20442.png', '12505.png', '22929.png', '24487.png', '16011.png', '00171.png', '08539.png', '24443.png', '06150.png', '24395.png', '15482.png', '10202.png', '13226.png', '01247.png', '08075.png', '17409.png', '05861.png', '02217.png', '19817.png', '22593.png', '03601.png', '10954.png', '20352.png', '00156.png', '12687.png', '09190.png', '06152.png', '18260.png', '21060.png', '04659.png', '22447.png', '13426.png', '08777.png', '19402.png', '17278.png', '06499.png', '00560.png', '18438.png', '03492.png', '05555.png', '03152.png', '16582.png', '15144.png', '14490.png', '02546.png', '03177.png', '14608.png', '15992.png', '13749.png', '09934.png', '23454.png', '06045.png', '18279.png', '15141.png', '10860.png', '07002.png', '02508.png', '17039.png', '12288.png', '14347.png', '02366.png', '18966.png', '13525.png', '13113.png', '07427.png', '01074.png', '20653.png', '20686.png', '24747.png', '03425.png', '07839.png', '22342.png', '20286.png', '09907.png', '00369.png', '03585.png', '18038.png', '10197.png', '20920.png', '19112.png', '22933.png', '21917.png', '19993.png', '15110.png', '17483.png', '09384.png', '11753.png', '22806.png', '21330.png', '14077.png', '14328.png', '21661.png', '01315.png', '06728.png', '02130.png', '18152.png', '11072.png', '17730.png', '20148.png', '01280.png', '06909.png', '09188.png', '24584.png', '21145.png', '02773.png', '05895.png', '06689.png', '10995.png', '22779.png', '09488.png', '23840.png', '01360.png', '23364.png', '01278.png', '21842.png', '11123.png', '08932.png', '03166.png', '21878.png', '20222.png', '10991.png', '13620.png', '13522.png', '01560.png', '19868.png', '05682.png', '02035.png', '01270.png', '03475.png', '14523.png', '20211.png', '03107.png', '10921.png', '13618.png', '01145.png', '06082.png', '15047.png', '19329.png', '14738.png', '04282.png', '18945.png', '18278.png', '10479.png', '23922.png', '14155.png', '02346.png', '00217.png', '16351.png', '09655.png', '05486.png', '13070.png', '09920.png', '12921.png', '21922.png', '17037.png', '15374.png', '18832.png', '23013.png', '07303.png', '02136.png', '11330.png', '22257.png', '10117.png', '18052.png', '22648.png', '00812.png', '14022.png', '13571.png', '11160.png', '01996.png', '06345.png', '19652.png', '14712.png', '17117.png', '21372.png', '10442.png', '24126.png', '18129.png', '23125.png', '10092.png', '13219.png', '19950.png', '13896.png', '06757.png', '02094.png', '13115.png', '20782.png', '20543.png', '11908.png', '20625.png', '02048.png', '13716.png', '24402.png', '11095.png', '08813.png', '03829.png', '22097.png', '09564.png', '15401.png', '13653.png', '06237.png', '06727.png', '13478.png', '12999.png', '04311.png', '12380.png', '19749.png', '21190.png', '21991.png', '21764.png', '24745.png', '18453.png', '17235.png', '12189.png', '24393.png', '23012.png', '11772.png', '14299.png', '24577.png', '15076.png', '05748.png', '03236.png', '08753.png', '09969.png', '07972.png', '13062.png', '20178.png', '17313.png', '08484.png', '09110.png', '19857.png', '12598.png', '19076.png', '15050.png', '20268.png', '22517.png', '06374.png', '15005.png', '14190.png', '12841.png', '07177.png', '08992.png', '04770.png', '04883.png', '23440.png', '24476.png', '00231.png', '21407.png', '06864.png', '22107.png', '20370.png', '15570.png', '22784.png', '00428.png', '10067.png', '19113.png', '16681.png', '08700.png', '22089.png', '07771.png', '01815.png', '23507.png', '24417.png', '11388.png', '16325.png', '07286.png', '15135.png', '00168.png', '20126.png', '13197.png', '05174.png', '18396.png', '23241.png', '02239.png', '03777.png', '24120.png', '04651.png', '14106.png', '05615.png', '23725.png', '03687.png', '12406.png', '03302.png', '21982.png', '16204.png', '09411.png', '15273.png', '22532.png', '19566.png', '04783.png', '11274.png', '10693.png', '06282.png', '09369.png', '04921.png', '12642.png', '10335.png', '20267.png', '22376.png', '17349.png', '07142.png', '10907.png', '23472.png', '18972.png', '17009.png', '08028.png', '20381.png', '02918.png', '19080.png', '09365.png', '00577.png', '13323.png', '04811.png', '05626.png', '16806.png', '23177.png', '14921.png', '12551.png', '01137.png', '24509.png', '06390.png', '18740.png', '22586.png', '03849.png', '13970.png', '17016.png', '12019.png', '12697.png', '09334.png', '09696.png', '05387.png', '10630.png', '19232.png', '02719.png', '03530.png', '22238.png', '08442.png', '23760.png', '01850.png', '00822.png', '01799.png', '15232.png', '03144.png', '03573.png', '23435.png', '02350.png', '22716.png', '14030.png', '16808.png', '00558.png', '11424.png', '20688.png', '12250.png', '00860.png', '01321.png', '23672.png', '22699.png', '21954.png', '17376.png', '20291.png', '13373.png', '07684.png', '05931.png', '02277.png', '01327.png', '00183.png', '04598.png', '10488.png', '06612.png', '04280.png', '02686.png', '09648.png', '04515.png', '04511.png', '09137.png', '16875.png', '15034.png', '18132.png', '23207.png', '01602.png', '16178.png', '06181.png', '06594.png', '11827.png', '24680.png', '10504.png', '07128.png', '18073.png', '03941.png', '24180.png', '20559.png', '19308.png', '11594.png', '22102.png', '16947.png', '12442.png', '01684.png', '14814.png', '19806.png', '11581.png', '17797.png', '08338.png', '03608.png', '10780.png', '05102.png', '05453.png', '18777.png', '20893.png', '21973.png', '16197.png', '23902.png', '15312.png', '23931.png', '05798.png', '09761.png', '23621.png', '00233.png', '20563.png', '18681.png', '01905.png', '03748.png', '00042.png', '04692.png', '07759.png', '03564.png', '23032.png', '23057.png', '18361.png', '14522.png', '14683.png', '07990.png', '15334.png', '11867.png', '02421.png', '18688.png', '14931.png', '20811.png', '15450.png', '17858.png', '15523.png', '16448.png', '12536.png', '11582.png', '12263.png', '07926.png', '10252.png', '00999.png', '15056.png', '24755.png', '08895.png', '16849.png', '11321.png', '16017.png', '05403.png', '22436.png', '02129.png', '14338.png', '19375.png', '10473.png', '13867.png', '18659.png', '21294.png', '10155.png', '13304.png', '08350.png', '05845.png', '08728.png', '09013.png', '02625.png', '21590.png', '00209.png', '01401.png', '09679.png', '21768.png', '20858.png', '04812.png', '24394.png', '21478.png', '05115.png', '18544.png', '23701.png', '23657.png', '06218.png', '02087.png', '24229.png', '09880.png', '02679.png', '11903.png', '14254.png', '16925.png', '24639.png', '08215.png', '17925.png', '14843.png', '05953.png', '09582.png', '01309.png', '07661.png', '16442.png', '14286.png', '07486.png', '15631.png', '16285.png', '04805.png', '13157.png', '00775.png', '04994.png', '24075.png', '12255.png', '23413.png', '07110.png', '16764.png', '00806.png', '07437.png', '18459.png', '16043.png', '17100.png', '13142.png', '15039.png', '11184.png', '09858.png', '20096.png', '20045.png', '08072.png', '04534.png', '16717.png', '06472.png', '09981.png', '09686.png', '15748.png', '24095.png', '12366.png', '20524.png', '04828.png', '16725.png', '17226.png', '12641.png', '13217.png', '00213.png', '08757.png', '06091.png', '18068.png', '05779.png', '08746.png', '23613.png', '18684.png', '13877.png', '19644.png', '07704.png', '24619.png', '24032.png', '22823.png', '06389.png', '05210.png', '14402.png', '14223.png', '00744.png', '19061.png', '04352.png', '23282.png', '04064.png', '00455.png', '09253.png', '19858.png', '06323.png', '05602.png', '22599.png', '24886.png', '16986.png', '22406.png', '06487.png', '00136.png', '12018.png', '00574.png', '15030.png', '15601.png', '21124.png', '23859.png', '23937.png', '18402.png', '10121.png', '08587.png', '16207.png', '17956.png', '18682.png', '10725.png', '01661.png', '12669.png', '23730.png', '12301.png', '15621.png', '15617.png', '23708.png', '08824.png', '06395.png', '16949.png', '13997.png', '11669.png', '13944.png', '00498.png', '05454.png', '16245.png', '16445.png', '10697.png', '08179.png', '23066.png', '06027.png', '06463.png', '15604.png', '20850.png', '07084.png', '04203.png', '09519.png', '10195.png', '10888.png', '18687.png', '23132.png', '21179.png', '04693.png', '10380.png', '15717.png', '00325.png', '13232.png', '05200.png', '03465.png', '20403.png', '06526.png', '16854.png', '09494.png', '23244.png', '15827.png', '03140.png', '21758.png', '05987.png', '17918.png', '08625.png', '03754.png', '20711.png', '17061.png', '20798.png', '20882.png', '00141.png', '23406.png', '01182.png', '02745.png', '24427.png', '18191.png', '09341.png', '23424.png', '22888.png', '15948.png', '10534.png', '23048.png', '08719.png', '08655.png', '03569.png', '18089.png', '08936.png', '17171.png', '20684.png', '00931.png', '13549.png', '01542.png', '07863.png', '03600.png', '17475.png', '09911.png', '14688.png', '02289.png', '13039.png', '22246.png', '15529.png', '15846.png', '22325.png', '14194.png', '14849.png', '01048.png', '12930.png', '17716.png', '18470.png', '19863.png', '23784.png', '14822.png', '16338.png', '01601.png', '00892.png', '02581.png', '20266.png', '07911.png', '17610.png', '17565.png', '19108.png', '23536.png', '16966.png', '18273.png', '15769.png', '07231.png', '20324.png', '13600.png', '20910.png', '12976.png', '05495.png', '02819.png', '15923.png', '18320.png', '10688.png', '11302.png', '12423.png', '13422.png', '09725.png', '06629.png', '14909.png', '14400.png', '12801.png', '07679.png', '08812.png', '21957.png', '09682.png', '02133.png', '23624.png', '14832.png', '05644.png', '14856.png', '04128.png', '02997.png', '21958.png', '18031.png', '07590.png', '07445.png', '15881.png', '04140.png', '21898.png', '09090.png', '03181.png', '13559.png', '09490.png', '24065.png', '13267.png', '23847.png', '05662.png', '21165.png', '19718.png', '07218.png', '06093.png', '24874.png', '07485.png', '14107.png', '02012.png', '23851.png', '21938.png', '12830.png', '23062.png', '14951.png', '05672.png', '20981.png', '03592.png', '08771.png', '14475.png', '17673.png', '04819.png', '05651.png', '03947.png', '14168.png', '10403.png', '19154.png', '08140.png', '17848.png', '08247.png', '17392.png', '11596.png', '07664.png', '24406.png', '20485.png', '11422.png', '08826.png', '03507.png', '02112.png', '08030.png', '13358.png', '13033.png', '13289.png', '11674.png', '15852.png', '12476.png', '20650.png', '16346.png', '17707.png', '03747.png', '04227.png', '09166.png', '04024.png', '22799.png', '20516.png', '22767.png', '05175.png', '06889.png', '20280.png', '23236.png', '16554.png', '22337.png', '22601.png', '20269.png', '08815.png', '20897.png', '18195.png', '23942.png', '24223.png', '15909.png', '04298.png', '14154.png', '16812.png', '16639.png', '22303.png', '22444.png', '02121.png', '02831.png', '13258.png', '04034.png', '14180.png', '14082.png', '11205.png', '13826.png', '06569.png', '22940.png', '10487.png', '20246.png', '12321.png', '07141.png', '01761.png', '22896.png', '01204.png', '06972.png', '08095.png', '00954.png', '16770.png', '13482.png', '20320.png', '05834.png', '10899.png', '08189.png', '00516.png', '18198.png', '17151.png', '08833.png', '17881.png', '17138.png', '21078.png', '24198.png', '03527.png', '04794.png', '16274.png', '05417.png', '18501.png', '07090.png', '20116.png', '17600.png', '20270.png', '05422.png', '11868.png', '02751.png', '11300.png', '04317.png', '20450.png', '12540.png', '19740.png', '24692.png', '21681.png', '05010.png', '00763.png', '17833.png', '02554.png', '10213.png', '13214.png', '05867.png', '23981.png', '15985.png', '09613.png', '10058.png', '20348.png', '17159.png', '01752.png', '24343.png', '23846.png', '09163.png', '01267.png', '23545.png', '00695.png', '12199.png', '16758.png', '06767.png', '09700.png', '14183.png', '02330.png', '17546.png', '18754.png', '04478.png', '13418.png', '05093.png', '21261.png', '17792.png', '18996.png', '08463.png', '07282.png', '22889.png', '19797.png', '24245.png', '20184.png', '18228.png', '03505.png', '15849.png', '17910.png', '15768.png', '13611.png', '02619.png', '19640.png', '23419.png', '02242.png', '13990.png', '12015.png', '05467.png', '16933.png', '09403.png', '15028.png', '13693.png', '16431.png', '10097.png', '24118.png', '23503.png', '10046.png', '09918.png', '22461.png', '02247.png', '18519.png', '08995.png', '05968.png', '00463.png', '07006.png', '19886.png', '10467.png', '10156.png', '09286.png', '20376.png', '20697.png', '24526.png', '19552.png', '13535.png', '22668.png', '12286.png', '18130.png', '10850.png', '07826.png', '21762.png', '10052.png', '07164.png', '11318.png', '24786.png', '02712.png', '06890.png', '10251.png', '11492.png', '23130.png', '24181.png', '07753.png', '18940.png', '19658.png', '06862.png', '07827.png', '23520.png', '09277.png', '04246.png', '20950.png', '11550.png', '18173.png', '05578.png', '17721.png', '13937.png', '06874.png', '04919.png', '05795.png', '11871.png', '23323.png', '23743.png', '14231.png', '04610.png', '24775.png', '04670.png', '18461.png', '04969.png', '09203.png', '08939.png', '09668.png', '16120.png', '16526.png', '16915.png', '02328.png', '09594.png', '23189.png', '07726.png', '13227.png', '14079.png', '09815.png', '12981.png', '21924.png', '21665.png', '13782.png', '15892.png', '13781.png', '21043.png', '11045.png', '14196.png', '17034.png', '21962.png', '00478.png', '19219.png', '19900.png', '02003.png', '02888.png', '06579.png', '11885.png', '20180.png', '11924.png', '17329.png', '17491.png', '04026.png', '06014.png', '21150.png', '22657.png', '06052.png', '07317.png', '24447.png', '10783.png', '15564.png', '03580.png', '15660.png', '07065.png', '15156.png', '24265.png', '21080.png', '08544.png', '00454.png', '15128.png', '03271.png', '18194.png', '07082.png', '22008.png', '01788.png', '20977.png', '19636.png', '21690.png', '24757.png', '16547.png', '16962.png', '21253.png', '16298.png', '04065.png', '17643.png', '19619.png', '15911.png', '04469.png', '01363.png', '24853.png', '08846.png', '07874.png', '18210.png', '13879.png', '02580.png', '06839.png', '00435.png', '23050.png', '15393.png', '19104.png', '04061.png', '07733.png', '23551.png', '06977.png', '20827.png', '07960.png', '05434.png', '18224.png', '21332.png', '00588.png', '01456.png', '18878.png', '02806.png', '18325.png', '20599.png', '13312.png', '09578.png', '22024.png', '11633.png', '02801.png', '21146.png', '10388.png', '21828.png', '18401.png', '18975.png', '04647.png', '10832.png', '03949.png', '18881.png', '13048.png', '12992.png', '24255.png', '20886.png', '05448.png', '10134.png', '00409.png', '11053.png', '06085.png', '01389.png', '11775.png', '05601.png', '14678.png', '18071.png', '03604.png', '22979.png', '06607.png', '02176.png', '15093.png', '06763.png', '12400.png', '13976.png', '09468.png', '08527.png', '18456.png', '02361.png', '15126.png', '14260.png', '06170.png', '06263.png', '11592.png', '22168.png', '03653.png', '06614.png', '20799.png', '20721.png', '21418.png', '10114.png', '11907.png', '24672.png', '11362.png', '03705.png', '15433.png', '19972.png', '15667.png', '17244.png', '09842.png', '13419.png', '18976.png', '15307.png', '13437.png', '22693.png', '12070.png', '14792.png', '13462.png', '19553.png', '08174.png', '06601.png', '07542.png', '11923.png', '23722.png', '23948.png', '07189.png', '10988.png', '03932.png', '05613.png', '18299.png', '01023.png', '18783.png', '19374.png', '15063.png', '10947.png', '11985.png', '07145.png', '14211.png', '12037.png', '12958.png', '08303.png', '18836.png', '06994.png', '21685.png', '24076.png', '03908.png', '21013.png', '13546.png', '14887.png', '18835.png', '17927.png', '10859.png', '06516.png', '14506.png', '03366.png', '10708.png', '12945.png', '04393.png', '17357.png', '05941.png', '11588.png', '16608.png', '10629.png', '15474.png', '20703.png', '09024.png', '03345.png', '02120.png', '18701.png', '22843.png', '14763.png', '16474.png', '03174.png', '05078.png', '19480.png', '23625.png', '03384.png', '17440.png', '15981.png', '17935.png', '20565.png', '14809.png', '02062.png', '05593.png', '01416.png', '11117.png', '00937.png', '01858.png', '15819.png', '21815.png', '07521.png', '24502.png', '09387.png', '11876.png', '19075.png', '15818.png', '01728.png', '05575.png', '19178.png', '04714.png', '22425.png', '18941.png', '20996.png', '05128.png', '10000.png', '12600.png', '04604.png', '23560.png', '04553.png', '24839.png', '07992.png', '24116.png', '22100.png', '09714.png', '19184.png', '16397.png', '23202.png', '00959.png', '13788.png', '08694.png', '17320.png', '12472.png', '10619.png', '12606.png', '18414.png', '19833.png', '08785.png', '18086.png', '15298.png', '19871.png', '11748.png', '08628.png', '02957.png', '22242.png', '17677.png', '23410.png', '09721.png', '03195.png', '14255.png', '09041.png', '19292.png', '01147.png', '17164.png', '18759.png', '07643.png', '19136.png', '03462.png', '21132.png', '20946.png', '20780.png', '19489.png', '05608.png', '22652.png', '24893.png', '24187.png', '04949.png', '15268.png', '06872.png', '15139.png', '13008.png', '08541.png', '05893.png', '17036.png', '12397.png', '10151.png', '05992.png', '01052.png', '14948.png', '15160.png', '10310.png', '21510.png', '18444.png', '15679.png', '23372.png', '03318.png', '21469.png', '05552.png', '20804.png', '21616.png', '07321.png', '03011.png', '16909.png', '06682.png', '15888.png', '09638.png', '10891.png', '13746.png', '04491.png', '10404.png', '01926.png', '00335.png', '00111.png', '21086.png', '15228.png', '24177.png', '09067.png', '07504.png', '10538.png', '05179.png', '09577.png', '18447.png', '24686.png', '20194.png', '23034.png', '17121.png', '11976.png', '17512.png', '09580.png', '07346.png', '10107.png', '01835.png', '06650.png', '02491.png', '18193.png', '12175.png', '12165.png', '00583.png', '13433.png', '24819.png', '06975.png', '00023.png', '15513.png', '24658.png', '05432.png', '13204.png', '24009.png', '20069.png', '20430.png', '03624.png', '22848.png', '14159.png', '09169.png', '08023.png', '15289.png', '22787.png', '20940.png', '22996.png', '10056.png', '23030.png', '02161.png', '12066.png', '18630.png', '23804.png', '07015.png', '06869.png', '24258.png', '19238.png', '18852.png', '24835.png', '04357.png', '10645.png', '24736.png', '14760.png', '12436.png', '19237.png', '02743.png', '13707.png', '16367.png', '14818.png', '18434.png', '12430.png', '02772.png', '10889.png', '12850.png', '20472.png', '13855.png', '12313.png', '24086.png', '10144.png', '09973.png', '23795.png', '22873.png', '14393.png', '13665.png', '23758.png', '16780.png', '09312.png', '22959.png', '19506.png', '09796.png', '22070.png', '20434.png', '04688.png', '02701.png', '00158.png', '17093.png', '19305.png', '22421.png', '14644.png', '11884.png', '06654.png', '00348.png', '12210.png', '13382.png', '00259.png', '18903.png', '09868.png', '07403.png', '24268.png', '03370.png', '18483.png', '24341.png', '13783.png', '22724.png', '03109.png', '24962.png', '02192.png', '18593.png', '23218.png', '18338.png', '19230.png', '14391.png', '05539.png', '07033.png', '01829.png', '18136.png', '24778.png', '12364.png', '15448.png', '20301.png', '23322.png', '12105.png', '15830.png', '11680.png', '08327.png', '19879.png', '12102.png', '22159.png', '14449.png', '00541.png', '06769.png', '23594.png', '13787.png', '03867.png', '01036.png', '09743.png', '10771.png', '03038.png', '05654.png', '04267.png', '00307.png', '00920.png', '08157.png', '20901.png', '01226.png', '07708.png', '12285.png', '09371.png', '08213.png', '07844.png', '11968.png', '15970.png', '04592.png', '24083.png', '14266.png', '13159.png', '15524.png', '01575.png', '10304.png', '24247.png', '00709.png', '18617.png', '19067.png', '07855.png', '22976.png', '22574.png', '00576.png', '16456.png', '06174.png', '10838.png', '13334.png', '22255.png', '04857.png', '10529.png', '13799.png', '08309.png', '24000.png', '21838.png', '08818.png', '18695.png', '18967.png', '12233.png', '08561.png', '23423.png', '21216.png', '13082.png', '00192.png', '17301.png', '06988.png', '02105.png', '17670.png', '03376.png', '03853.png', '12357.png', '24066.png', '22386.png', '13161.png', '09199.png', '17966.png', '23716.png', '02352.png', '09641.png', '22876.png', '19609.png', '02565.png', '04838.png', '24386.png', '01181.png', '04871.png', '17236.png', '15907.png', '04432.png', '23334.png', '04180.png', '02628.png', '02438.png', '17841.png', '20863.png', '00234.png', '19189.png', '03714.png', '13347.png', '07424.png', '04449.png', '06489.png', '14416.png', '09401.png', '18492.png', '03738.png', '17490.png', '04989.png', '11122.png', '06717.png', '19126.png', '10206.png', '06518.png', '05908.png', '01959.png', '06628.png', '24925.png', '15902.png', '10906.png', '03734.png', '10634.png', '24080.png', '01997.png', '21474.png', '09176.png', '07641.png', '03371.png', '17044.png', '13189.png', '07111.png', '02185.png', '04807.png', '09459.png', '06784.png', '10877.png', '21237.png', '23853.png', '08314.png', '13406.png', '22681.png', '00251.png', '10715.png', '20750.png', '21421.png', '14913.png', '19570.png', '17237.png', '18961.png', '11892.png', '12970.png', '03700.png', '10245.png', '11150.png', '19538.png', '00517.png', '17013.png', '00040.png', '01936.png', '15572.png', '01572.png', '10839.png', '18891.png', '17879.png', '10192.png', '11769.png', '19283.png', '18766.png', '23311.png', '19527.png', '20551.png', '23085.png', '13748.png', '18275.png', '15516.png', '15906.png', '16433.png', '17315.png', '03471.png', '01914.png', '00303.png', '24098.png', '18602.png', '12952.png', '23574.png', '08317.png', '11198.png', '09424.png', '15747.png', '10418.png', '06828.png', '23837.png', '17645.png', '19255.png', '19270.png', '24451.png', '20866.png', '16516.png', '02704.png', '02333.png', '02213.png', '02511.png', '16320.png', '03523.png', '18366.png', '10203.png', '12277.png', '09615.png', '21890.png', '05250.png', '24296.png', '21619.png', '14612.png', '14781.png', '09442.png', '15994.png', '22866.png', '03346.png', '07156.png', '17492.png', '03962.png', '22535.png', '24627.png', '05114.png', '00381.png', '01823.png', '04810.png', '12055.png', '05201.png', '10076.png', '08928.png', '16358.png', '14431.png', '09035.png', '06110.png', '07098.png', '02505.png', '00503.png', '23234.png', '01771.png', '09731.png', '15620.png', '21035.png', '16085.png', '15475.png', '14530.png', '21844.png', '22973.png', '15200.png', '03498.png', '10030.png', '17965.png', '11993.png', '04668.png', '11356.png', '13663.png', '06308.png', '09159.png', '00755.png', '10999.png', '11073.png', '24434.png', '21535.png', '22851.png', '11797.png', '14555.png', '18507.png', '22477.png', '08840.png', '24855.png', '17476.png', '01972.png', '16657.png', '00467.png', '17298.png', '20338.png', '14532.png', '07376.png', '20678.png', '01770.png', '24225.png', '00875.png', '16810.png', '09418.png', '18742.png', '24136.png', '11180.png', '02845.png', '22942.png', '04451.png', '23906.png', '24313.png', '07005.png', '11933.png', '14721.png', '12932.png', '16710.png', '20183.png', '15437.png', '02992.png', '09288.png', '17147.png', '10527.png', '20693.png', '03955.png', '21599.png', '15335.png', '19693.png', '22705.png', '07867.png', '16793.png', '10334.png', '16366.png', '01976.png', '17132.png', '09446.png', '14096.png', '23632.png', '10739.png', '23008.png', '06486.png', '10460.png', '22405.png', '01109.png', '06789.png', '18990.png', '16640.png', '11820.png', '23238.png', '19515.png', '17160.png', '09831.png', '18145.png', '10186.png', '07456.png', '21624.png', '23314.png', '01875.png', '15578.png', '01243.png', '09454.png', '17124.png', '17463.png', '22367.png', '06833.png', '08979.png', '22972.png', '18949.png', '22708.png', '06175.png', '18982.png', '04677.png', '00317.png', '06957.png', '05714.png', '07660.png', '06385.png', '18864.png', '09420.png', '14321.png', '01433.png', '01039.png', '03151.png', '03878.png', '10515.png', '19512.png', '01665.png', '23400.png', '19971.png', '09680.png', '11774.png', '06804.png', '06998.png', '06415.png', '08263.png', '01904.png', '10381.png', '16109.png', '13378.png', '12330.png', '21048.png', '24416.png', '11294.png', '07476.png', '20404.png', '21721.png', '09099.png', '20142.png', '07421.png', '22181.png', '14405.png', '05085.png', '24866.png', '14277.png', '23554.png', '14335.png', '23468.png', '08361.png', '24871.png', '07108.png', '07073.png', '16771.png', '21558.png', '17297.png', '12947.png', '08357.png', '14367.png', '16675.png', '01603.png', '00585.png', '23946.png', '03766.png', '21649.png', '18369.png', '14175.png', '08454.png', '23921.png', '08424.png', '09590.png', '11562.png', '19824.png', '13081.png', '05374.png', '02224.png', '15294.png', '17616.png', '15641.png', '04372.png', '08688.png', '11460.png', '02227.png', '15576.png', '18352.png', '09571.png', '14538.png', '20891.png', '23558.png', '15362.png', '18767.png', '03115.png', '12050.png', '14201.png', '11618.png', '08617.png', '18243.png', '14884.png', '19808.png', '06214.png', '05103.png', '11684.png', '22339.png', '00543.png', '07524.png', '04739.png', '20646.png', '23233.png', '00844.png', '21486.png', '07055.png', '10671.png', '23993.png', '11414.png', '04901.png', '12095.png', '20962.png', '12354.png', '17521.png', '16728.png', '06172.png', '10262.png', '01782.png', '24612.png', '16422.png', '07575.png', '20221.png', '12317.png', '07020.png', '05362.png', '12826.png', '19185.png', '04579.png', '04781.png', '10152.png', '20056.png', '13842.png', '15704.png', '21595.png', '24730.png', '18358.png', '13088.png', '09987.png', '21199.png', '19344.png', '08016.png', '09541.png', '04303.png', '24464.png', '20416.png', '05951.png', '22380.png', '08026.png', '13881.png', '01242.png', '20618.png', '24742.png', '04609.png', '17602.png', '14479.png', '11196.png', '20980.png', '06159.png', '06041.png', '23715.png', '03139.png', '14131.png', '07922.png', '14594.png', '20217.png', '20794.png', '19330.png', '04160.png', '00391.png', '00732.png', '17611.png', '24537.png', '18660.png', '11061.png', '12135.png', '12909.png', '07627.png', '10268.png', '24859.png', '12684.png', '22201.png', '22859.png', '03884.png', '04792.png', '02467.png', '12221.png', '10791.png', '10402.png', '04965.png', '19537.png', '17557.png', '21193.png', '19585.png', '21438.png', '00720.png', '19475.png', '03950.png', '16801.png', '16513.png', '23382.png', '18007.png', '19684.png', '04423.png', '04085.png', '23445.png', '13003.png', '06322.png', '13566.png', '08396.png', '09498.png', '18205.png', '06221.png', '19296.png', '05876.png', '01878.png', '09382.png', '13492.png', '23669.png', '22958.png', '05877.png', '21960.png', '10987.png', '10618.png', '11977.png', '14739.png', '20476.png', '09636.png', '01759.png', '18611.png', '16634.png', '13907.png', '11404.png', '05523.png', '05745.png', '12504.png', '09701.png', '18051.png', '23543.png', '00237.png', '14304.png', '22332.png', '07081.png', '01604.png', '23220.png', '16918.png', '11337.png', '07488.png', '01331.png', '13209.png', '03433.png', '14267.png', '11795.png', '04722.png', '22298.png', '07822.png', '21623.png', '09898.png', '06297.png', '01861.png', '19795.png', '10270.png', '24107.png', '02288.png', '13695.png', '01541.png', '24140.png', '07700.png', '21003.png', '11025.png', '10433.png', '06529.png', '01167.png', '11470.png', '18908.png', '15281.png', '01484.png', '02024.png', '23911.png', '13905.png', '05385.png', '13043.png', '21439.png', '11322.png', '20490.png', '21085.png', '23544.png', '03280.png', '03931.png', '00130.png', '23739.png', '12271.png', '19431.png', '24919.png', '08494.png', '22516.png', '19016.png', '03857.png', '13085.png', '08748.png', '10522.png', '05898.png', '12029.png', '19898.png', '13060.png', '19648.png', '16754.png', '17906.png', '01950.png', '13436.png', '03975.png', '00308.png', '18494.png', '24468.png', '02417.png', '16158.png', '04750.png', '16416.png', '08697.png', '09903.png', '23119.png', '03850.png', '19924.png', '23393.png', '20605.png', '05887.png', '14700.png', '20496.png', '14575.png', '05131.png', '04035.png', '13101.png', '04107.png', '01863.png', '16357.png', '02943.png', '06825.png', '12219.png', '16847.png', '16551.png', '17762.png', '11790.png', '13150.png', '12194.png', '23221.png', '04641.png', '12270.png', '02108.png', '02641.png', '08507.png', '02999.png', '10701.png', '02594.png', '23611.png', '09669.png', '16032.png', '09279.png', '12565.png', '04759.png', '14239.png', '04882.png', '19490.png', '07395.png', '13027.png', '09608.png', '09178.png', '24251.png', '12470.png', '08402.png', '15573.png', '08293.png', '14655.png', '19468.png', '05884.png', '15828.png', '13883.png', '08605.png', '00034.png', '09697.png', '21000.png', '06469.png', '05850.png', '03277.png', '22573.png', '23266.png', '14437.png', '22167.png', '16187.png', '04291.png', '03224.png', '02730.png', '20806.png', '21152.png', '10659.png', '16802.png', '13180.png', '22286.png', '16025.png', '04986.png', '06274.png', '20122.png', '00093.png', '01241.png', '24947.png', '11691.png', '24418.png', '08161.png', '14383.png', '02564.png', '11450.png', '02607.png', '15675.png', '13194.png', '19846.png', '23703.png', '05557.png', '21786.png', '16890.png', '13557.png', '24278.png', '18355.png', '17388.png', '08873.png', '04104.png', '18612.png', '13694.png', '14579.png', '01271.png', '23289.png', '01300.png', '14699.png', '21045.png', '10040.png', '24923.png', '08292.png', '11000.png', '03711.png', '12450.png', '15058.png', '02102.png', '07607.png', '24211.png', '13165.png', '18530.png', '04644.png', '11331.png', '11041.png', '23812.png', '22580.png', '12443.png', '18393.png', '06809.png', '20125.png', '15507.png', '01130.png', '11015.png', '20500.png', '22220.png', '09947.png', '03877.png', '03010.png', '05678.png', '01108.png', '00752.png', '09589.png', '16179.png', '10724.png', '14138.png', '10454.png', '20796.png', '08896.png', '05416.png', '15712.png', '05943.png', '20462.png', '00888.png', '12993.png', '15325.png', '04658.png', '04115.png', '13765.png', '12672.png', '21671.png', '04113.png', '23053.png', '08965.png', '14998.png', '20660.png', '15998.png', '17933.png', '13096.png', '07364.png', '14905.png', '15727.png', '04835.png', '19117.png', '09107.png', '06524.png', '13547.png', '19193.png', '11663.png', '06112.png', '06754.png', '09972.png', '11112.png', '08994.png', '07255.png', '24163.png', '14663.png', '00181.png', '22223.png', '06749.png', '03282.png', '19011.png', '15372.png', '00475.png', '17522.png', '23832.png', '02741.png', '00370.png', '05752.png', '03989.png', '22597.png', '02181.png', '19029.png', '03837.png', '15833.png', '12064.png', '09984.png', '08478.png', '05945.png', '11932.png', '16872.png', '24377.png', '07410.png', '12747.png', '11366.png', '23259.png', '12494.png', '04223.png', '05760.png', '23321.png', '03417.png', '00352.png', '03015.png', '07551.png', '24233.png', '18271.png', '04173.png', '15384.png', '03389.png', '20931.png', '24776.png', '03980.png', '11973.png', '16099.png', '14960.png', '08996.png', '16056.png', '23979.png', '03677.png', '17866.png', '01346.png', '15091.png', '02904.png', '12854.png', '23339.png', '11212.png', '15920.png', '16800.png', '07910.png', '06916.png', '07653.png', '06012.png', '12759.png', '09704.png', '07920.png', '23785.png', '15695.png', '12917.png', '23542.png', '10874.png', '02642.png', '09600.png', '23546.png', '04549.png', '05777.png', '17126.png', '21473.png', '02873.png', '11138.png', '14811.png', '21139.png', '09619.png', '01854.png', '19546.png', '21032.png', '18649.png', '17983.png', '08666.png', '14740.png', '20945.png', '01992.png', '08510.png', '08991.png', '08344.png', '23990.png', '07588.png', '08129.png', '03442.png', '21105.png', '22036.png', '09813.png', '15140.png', '15082.png', '10426.png', '17115.png', '12432.png', '05784.png', '05143.png', '13544.png', '15049.png', '03707.png', '19485.png', '15514.png', '05535.png', '00630.png', '12943.png', '07967.png', '24662.png', '02334.png', '20470.png', '15280.png', '07198.png', '05379.png', '21557.png', '16661.png', '00155.png', '21240.png', '09968.png', '06065.png', '15485.png', '12069.png', '17143.png', '24641.png', '22105.png', '06555.png', '15330.png', '22158.png', '10236.png', '14426.png', '12585.png', '09135.png', '09563.png', '03263.png', '16549.png', '23748.png', '10776.png', '11605.png', '02032.png', '05206.png', '10506.png', '13288.png', '00318.png', '01694.png', '18146.png', '02128.png', '05021.png', '08819.png', '16229.png', '09782.png', '03789.png', '10886.png', '18014.png', '00659.png', '20389.png', '24651.png', '04992.png', '01528.png', '00270.png', '10668.png', '17620.png', '07500.png', '05371.png', '19714.png', '19973.png', '09017.png', '18918.png', '15418.png', '17046.png', '01714.png', '12610.png', '19909.png', '00680.png', '15467.png', '24939.png', '03460.png', '15588.png', '18981.png', '23076.png', '24064.png', '13762.png', '11128.png', '15267.png', '02666.png', '05563.png', '19293.png', '12957.png', '10606.png', '19735.png', '16276.png', '01015.png', '09908.png', '21118.png', '19129.png', '02409.png', '05333.png', '08897.png', '01487.png', '11425.png', '19887.png', '08763.png', '07369.png', '18910.png', '24202.png', '08960.png', '24227.png', '24473.png', '08692.png', '10189.png', '03787.png', '19872.png', '01051.png', '05117.png', '18232.png', '04228.png', '20076.png', '24677.png', '24883.png', '02924.png', '22690.png', '07908.png', '09811.png', '18968.png', '00668.png', '15486.png', '24378.png', '24483.png', '22685.png', '06292.png', '18629.png', '17653.png', '21827.png', '24449.png', '00861.png', '20913.png', '19494.png', '24729.png', '15821.png', '01640.png', '21513.png', '24888.png', '13089.png', '02872.png', '12828.png', '22766.png', '17265.png', '03661.png', '23428.png', '02955.png', '17909.png', '14542.png', '09738.png', '12822.png', '24384.png', '24698.png', '01797.png', '13703.png', '00654.png', '20440.png', '03745.png', '12905.png', '09269.png', '10362.png', '09520.png', '14302.png', '20353.png', '08953.png', '21925.png', '16726.png', '04032.png', '06752.png', '13839.png', '19901.png', '10086.png', '09836.png', '07339.png', '17326.png', '12682.png', '24851.png', '04011.png', '00514.png', '08834.png', '12348.png', '08385.png', '07415.png', '13469.png', '12150.png', '10032.png', '07750.png', '22519.png', '14838.png', '12611.png', '16499.png', '00561.png', '22908.png', '23587.png', '21403.png', '16404.png', '10384.png', '13841.png', '23519.png', '09098.png', '10686.png', '00176.png', '12236.png', '22607.png', '17869.png', '11026.png', '21777.png', '14337.png', '10628.png', '24020.png', '00444.png', '19001.png', '02776.png', '15162.png', '20192.png', '20592.png', '11368.png', '00505.png', '20351.png', '01654.png', '11809.png', '16317.png', '09875.png', '07235.png', '10714.png', '03416.png', '01811.png', '15693.png', '08983.png', '06224.png', '12469.png', '05919.png', '24038.png', '05008.png', '05880.png', '24133.png', '11127.png', '11426.png', '13078.png', '04496.png', '20041.png', '06425.png', '15973.png', '13155.png', '07397.png', '13246.png', '10760.png', '04369.png', '23860.png', '15375.png', '03509.png', '13812.png', '22130.png', '00397.png', '11891.png', '22209.png', '03099.png', '18063.png', '02389.png', '16680.png', '22910.png', '21868.png', '24695.png', '24385.png', '04493.png', '16961.png', '17071.png', '24676.png', '13981.png', '15112.png', '01211.png', '18427.png', '24572.png', '01090.png', '16246.png', '16250.png', '10342.png', '22758.png', '09620.png', '09640.png', '03804.png', '23719.png', '00024.png', '15898.png', '05394.png', '07153.png', '17391.png', '03579.png', '07789.png', '08693.png', '11194.png', '15890.png', '22965.png', '07754.png', '04093.png', '18421.png', '14472.png', '19748.png', '07257.png', '16702.png', '21445.png', '05698.png', '20441.png', '09152.png', '22250.png', '09924.png', '01374.png', '20323.png', '09999.png', '15248.png', '15111.png', '18458.png', '09755.png', '18845.png', '03888.png', '15306.png', '01540.png', '04613.png', '03731.png', '24820.png', '17340.png', '10176.png', '03869.png', '12492.png', '03936.png', '00960.png', '13755.png', '02388.png', '12583.png', '05196.png', '05411.png', '03207.png', '14983.png', '06729.png', '14339.png', '20177.png', '15229.png', '10394.png', '22701.png', '07460.png', '22136.png', '18226.png', '18510.png', '18363.png', '20584.png', '19397.png', '22015.png', '04788.png', '08633.png', '11942.png', '07728.png', '12396.png', '16078.png', '16414.png', '07886.png', '13476.png', '11889.png', '22154.png', '13093.png', '11365.png', '08149.png', '08821.png', '22297.png', '05260.png', '12858.png', '23797.png', '09559.png', '17012.png', '24657.png', '08188.png', '00819.png', '16238.png', '18502.png', '11436.png', '03087.png', '20803.png', '15508.png', '11830.png', '18484.png', '12201.png', '10408.png', '14706.png', '15218.png', '06907.png', '17657.png', '19519.png', '01512.png', '12362.png', '07836.png', '14081.png', '23160.png', '13594.png', '12121.png', '24079.png', '19315.png', '22722.png', '08388.png', '02226.png', '07109.png', '24337.png', '13475.png', '03567.png', '20783.png', '06068.png', '13116.png', '00146.png', '01873.png', '13336.png', '06770.png', '07878.png', '04642.png', '12546.png', '23548.png', '06430.png', '24022.png', '12771.png', '19803.png', '04056.png', '13367.png', '08959.png', '18439.png', '06521.png', '15520.png', '18158.png', '02946.png', '14908.png', '14978.png', '00963.png', '00390.png', '11555.png', '03324.png', '15136.png', '11419.png', '09472.png', '19055.png', '23926.png', '04389.png', '07125.png', '15550.png', '03555.png', '02337.png', '23706.png', '18791.png', '24665.png', '14614.png', '23285.png', '19954.png', '10377.png', '17324.png', '24710.png', '05634.png', '15562.png', '09574.png', '08563.png', '08121.png', '00160.png', '17105.png', '06040.png', '14562.png', '21569.png', '01517.png', '22375.png', '16869.png', '02000.png', '14033.png', '21422.png', '16712.png', '11811.png', '21622.png', '19611.png', '18006.png', '08486.png', '10081.png', '04749.png', '04275.png', '00019.png', '21141.png', '20959.png', '05621.png', '17328.png', '13463.png', '16443.png', '20748.png', '16306.png', '02902.png', '04542.png', '08272.png', '13445.png', '15315.png', '00405.png', '14618.png', '08695.png', '24626.png', '10681.png', '11133.png', '20163.png', '15736.png', '03164.png', '10719.png', '06246.png', '03660.png', '23188.png', '08610.png', '14560.png', '07800.png', '24069.png', '19752.png', '16895.png', '18315.png', '02753.png', '22740.png', '22497.png', '19046.png', '00804.png', '10015.png', '18497.png', '18066.png', '22160.png', '18106.png', '06327.png', '05809.png', '06018.png', '12223.png', '06145.png', '05295.png', '14213.png', '05873.png', '06333.png', '09045.png', '16893.png', '11551.png', '16985.png', '16739.png', '16502.png', '13938.png', '12741.png', '13036.png', '15069.png', '02281.png', '21532.png', '11905.png', '20644.png', '11020.png', '24769.png', '00609.png', '04523.png', '17430.png', '04817.png', '09659.png', '17209.png', '08659.png', '22836.png', '14797.png', '04584.png', '14959.png', '01677.png', '08400.png', '04681.png', '22153.png', '19626.png', '12927.png', '13833.png', '06642.png', '22577.png', '15436.png', '04436.png', '02190.png', '23965.png', '22974.png', '21949.png', '23161.png', '07870.png', '16731.png', '01253.png', '08514.png', '15990.png', '19339.png', '07069.png', '07527.png', '06417.png', '15929.png', '23637.png', '02343.png', '22372.png', '04135.png', '17571.png', '14494.png', '18562.png', '01288.png', '22703.png', '06138.png', '01881.png', '10920.png', '03284.png', '22417.png', '16578.png', '09838.png', '00175.png', '06982.png', '00142.png', '22898.png', '01411.png', '14005.png', '12897.png', '10003.png', '24869.png', '13964.png', '09318.png', '23523.png', '15780.png', '14971.png', '23078.png', '19794.png', '19158.png', '23522.png', '15770.png', '12082.png', '04620.png', '21346.png', '04547.png', '12039.png', '05290.png', '12758.png', '14823.png', '19830.png', '04967.png', '16305.png', '04219.png', '10956.png', '01856.png', '14889.png', '02844.png', '09276.png', '04743.png', '16928.png', '07760.png', '17378.png', '23527.png', '24629.png', '11565.png', '09544.png', '09290.png', '09078.png', '19481.png', '13450.png', '06413.png', '06419.png', '08166.png', '24215.png', '24170.png', '12569.png', '09673.png', '16118.png', '14364.png', '04726.png', '21155.png', '24363.png', '13176.png', '02299.png', '10145.png', '15575.png', '19959.png', '01458.png', '00247.png', '15983.png', '15868.png', '21906.png', '23919.png', '04624.png', '00573.png', '00065.png', '17568.png', '19725.png', '15996.png', '04884.png', '06580.png', '15674.png', '11022.png', '23968.png', '05366.png', '09825.png', '11151.png', '08167.png', '06367.png', '19847.png', '02163.png', '16655.png', '05003.png', '01731.png', '00966.png', '03451.png', '17426.png', '23347.png', '11055.png', '16686.png', '04326.png', '10810.png', '13935.png', '15052.png', '08670.png', '21733.png', '08038.png', '23806.png', '23001.png', '03423.png', '24554.png', '05419.png', '02255.png', '06740.png', '20317.png', '10731.png', '03382.png', '11708.png', '06302.png', '18863.png', '03485.png', '15463.png', '16695.png', '10700.png', '08948.png', '13125.png', '22775.png', '14060.png', '14890.png', '01970.png', '22293.png', '14295.png', '09829.png', '03468.png', '19050.png', '04948.png', '07637.png', '00206.png', '10521.png', '10139.png', '20249.png', '00798.png', '00849.png', '13850.png', '24814.png', '13871.png', '23414.png', '03411.png', '19007.png', '12750.png', '00693.png', '08918.png', '00542.png', '17648.png', '21840.png', '09768.png', '24053.png', '12508.png', '24267.png', '21133.png', '14496.png', '24648.png', '05700.png', '01169.png', '19155.png', '05538.png', '08172.png', '07746.png', '07386.png', '11661.png', '18539.png', '11947.png', '00678.png', '07064.png', '22418.png', '01339.png', '22422.png', '18573.png', '15352.png', '06999.png', '01624.png', '17229.png', '17346.png', '07545.png', '10511.png', '15925.png', '22953.png', '23079.png', '19151.png', '08202.png', '22484.png', '14275.png', '06143.png', '00219.png', '11229.png', '01417.png', '06479.png', '18036.png', '14040.png', '02948.png', '02954.png', '12735.png', '03379.png', '13315.png', '05810.png', '12052.png', '03450.png', '04112.png', '15002.png', '20813.png', '02695.png', '01480.png', '00417.png', '00385.png', '23688.png', '15291.png', '06319.png', '07888.png', '13638.png', '21236.png', '20856.png', '02993.png', '08356.png', '10909.png', '19157.png', '11474.png', '09051.png', '03561.png', '14845.png', '22262.png', '24523.png', '02538.png', '03607.png', '21826.png', '21227.png', '12211.png', '08450.png', '08466.png', '24201.png', '00737.png', '12435.png', '14572.png', '03709.png', '01753.png', '04571.png', '02515.png', '04784.png', '21741.png', '07541.png', '19092.png', '08783.png', '12673.png', '13758.png', '06764.png', '04484.png', '08440.png', '21274.png', '08407.png', '17325.png', '15201.png', '10018.png', '24205.png', '16891.png', '16088.png', '20941.png', '01507.png', '14249.png', '15510.png', '10181.png', '03368.png', '20735.png', '10347.png', '16159.png', '00482.png', '02160.png', '13098.png', '10178.png', '20664.png', '20341.png', '10363.png', '05341.png', '17439.png', '12967.png', '00430.png', '05048.png', '04456.png', '23249.png', '10483.png', '24653.png', '07454.png', '08304.png', '07322.png', '17371.png', '08180.png', '16068.png', '24917.png', '15397.png', '21091.png', '19936.png', '23303.png', '08156.png', '24606.png', '14928.png', '13714.png', '22623.png', '03708.png', '06102.png', '05822.png', '15060.png', '00656.png', '06189.png', '09534.png', '14525.png', '01296.png', '03103.png', '20020.png', '08260.png', '10549.png', '17024.png', '08564.png', '04987.png', '09993.png', '17724.png', '16601.png', '15062.png', '17243.png', '05235.png', '20968.png', '09819.png', '23063.png', '05731.png', '18104.png', '02525.png', '22125.png', '08265.png', '19324.png', '08847.png', '00215.png', '02420.png', '14847.png', '05840.png', '15466.png', '04198.png', '05574.png', '18256.png', '02677.png', '11157.png', '11519.png', '21010.png', '07815.png', '06676.png', '02636.png', '09441.png', '16036.png', '11476.png', '20231.png', '15035.png', '02653.png', '19980.png', '05704.png', '15163.png', '01110.png', '14413.png', '09957.png', '19065.png', '22335.png', '00894.png', '02796.png', '04322.png', '11207.png', '17006.png', '06353.png', '00283.png', '23914.png', '16259.png', '14398.png', '17247.png', '02372.png', '06493.png', '17996.png', '10563.png', '01084.png', '06296.png', '22280.png', '07374.png', '24740.png', '24050.png', '23774.png', '02400.png', '21018.png', '08520.png', '00731.png', '02390.png', '04765.png', '00566.png', '10485.png', '03081.png', '21653.png', '21210.png', '07345.png', '20589.png', '20375.png', '17868.png', '20138.png', '15003.png', '16991.png', '08603.png', '15872.png', '13139.png', '00033.png', '04394.png', '19447.png', '00473.png', '02846.png', '11922.png', '10383.png', '17782.png', '01584.png', '11875.png', '21044.png', '19120.png', '01766.png', '24935.png', '00085.png', '23310.png', '10199.png', '06187.png', '21790.png', '09509.png', '16041.png', '00460.png', '14980.png', '09733.png', '18137.png', '10922.png', '07642.png', '01091.png', '05097.png', '05852.png', '02597.png', '20542.png', '10742.png', '24746.png', '10820.png', '02825.png', '08825.png', '22672.png', '01285.png', '01351.png', '01252.png', '08531.png', '16752.png', '01366.png', '03373.png', '14000.png', '13534.png', '04247.png', '06149.png', '14815.png', '09360.png', '05015.png', '22544.png', '12000.png', '17634.png', '02932.png', '17518.png', '10447.png', '20198.png', '23872.png', '19500.png', '02771.png', '04270.png', '12570.png', '21089.png', '21386.png', '05086.png', '16884.png', '19975.png', '19192.png', '19815.png', '15336.png', '24842.png', '11086.png', '01561.png', '14898.png', '00745.png', '05513.png', '23971.png', '20300.png', '03597.png', '12020.png', '24504.png', '05600.png', '13825.png', '03243.png', '05293.png', '04079.png', '12730.png', '03797.png', '21333.png', '22199.png', '11246.png', '11996.png', '09086.png', '12230.png', '19100.png', '14500.png', '20368.png', '00194.png', '13658.png', '04683.png', '16945.png', '22712.png', '15721.png', '13577.png', '04525.png', '02985.png', '12566.png', '17814.png', '22840.png', '05030.png', '19892.png', '09879.png', '03355.png', '08661.png', '24781.png', '07267.png', '15328.png', '00697.png', '18871.png', '17875.png', '15579.png', '03440.png', '05028.png', '21001.png', '07742.png', '12130.png', '17330.png', '17120.png', '05596.png', '08919.png', '00432.png', '24442.png', '06858.png', '10787.png', '03294.png', '00104.png', '09225.png', '14355.png', '12109.png', '09624.png', '03191.png', '03620.png', '10034.png', '01626.png', '03646.png', '01522.png', '18523.png', '10375.png', '00799.png', '14409.png', '13024.png', '13511.png', '21846.png', '12573.png', '01568.png', '07882.png', '19459.png', '16491.png', '01606.png', '20807.png', '16871.png', '09413.png', '14690.png', '07016.png', '21396.png', '24892.png', '17604.png', '24361.png', '18724.png', '14918.png', '18587.png', '23117.png', '20615.png', '05471.png', '20842.png', '23499.png', '15152.png', '09874.png', '23333.png', '09251.png', '23841.png', '13889.png', '07985.png', '17696.png', '12906.png', '23122.png', '09585.png', '14156.png', '11249.png', '15644.png', '08874.png', '11697.png', '07017.png', '03229.png', '00345.png', '11398.png', '05194.png', '04502.png', '08426.png', '13953.png', '24183.png', '15924.png', '19395.png', '03106.png', '15241.png', '13519.png', '22583.png', '19841.png', '03330.png', '11990.png', '08547.png', '15944.png', '03429.png', '19496.png', '00321.png', '00054.png', '10288.png', '12638.png', '00949.png', '07762.png', '09500.png', '15883.png', '22538.png', '06950.png', '16584.png', '04574.png', '07539.png', '04419.png', '20736.png', '07042.png', '04977.png', '03641.png', '24516.png', '17971.png', '15318.png', '11351.png', '04775.png', '19384.png', '03647.png', '11538.png', '05327.png', '20951.png', '16029.png', '01943.png', '16168.png', '19935.png', '20597.png', '18870.png', '04385.png', '10528.png', '01965.png', '13480.png', '02906.png', '02812.png', '04256.png', '20541.png', '07274.png', '16753.png', '03973.png', '18375.png', '13869.png', '13866.png', '13853.png', '12174.png', '05571.png', '02492.png', '23885.png', '04395.png', '14702.png', '06932.png', '15961.png', '01432.png', '07299.png', '11347.png', '02265.png', '02051.png', '04700.png', '19376.png', '08804.png', '09653.png', '09262.png', '01166.png', '19201.png', '12938.png', '23337.png', '17666.png', '08556.png', '12683.png', '09036.png', '00545.png', '13584.png', '09353.png', '13251.png', '07340.png', '06893.png', '00326.png', '09698.png', '07736.png', '14745.png', '15498.png', '04733.png', '13376.png', '06989.png', '20130.png', '06811.png', '03781.png', '00667.png', '17682.png', '00890.png', '22331.png', '17293.png', '01062.png', '15522.png', '00197.png', '01087.png', '01680.png', '02125.png', '01282.png', '18693.png', '11224.png', '16523.png', '16559.png', '18597.png', '20318.png', '03880.png', '08518.png', '01662.png', '17343.png', '02011.png', '18582.png', '15404.png', '01373.png', '17652.png', '15756.png', '15772.png', '19039.png', '10276.png', '13508.png', '13044.png', '07522.png', '01215.png', '09914.png', '20580.png', '13936.png', '07241.png', '20509.png', '05812.png', '18909.png', '17924.png', '21879.png', '19580.png', '22937.png', '07314.png', '23223.png', '12489.png', '19234.png', '14454.png', '13351.png', '05775.png', '08594.png', '01498.png', '10827.png', '04354.png', '10951.png', '00523.png', '01163.png', '09881.png', '18662.png', '23541.png', '03943.png', '11931.png', '16967.png', '05002.png', '22528.png', '20273.png', '10580.png', '17290.png', '18639.png', '01765.png', '03481.png', '05218.png', '23764.png', '14386.png', '21312.png', '21101.png', '00872.png', '00948.png', '17220.png', '11514.png', '19661.png', '15599.png', '21462.png', '20789.png', '18367.png', '05324.png', '01787.png', '07271.png', '07600.png', '18788.png', '20956.png', '02014.png', '22515.png', '16572.png', '10768.png', '00996.png', '21659.png', '14771.png', '12392.png', '22340.png', '18247.png', '15225.png', '08781.png', '19133.png', '07229.png', '22067.png', '06996.png', '06525.png', '19708.png', '02913.png', '01056.png', '12539.png', '00977.png', '15797.png', '13806.png', '04096.png', '06856.png', '09851.png', '00028.png', '05719.png', '19177.png', '23320.png', '16208.png', '24008.png', '17401.png', '15097.png', '05277.png', '01427.png', '22252.png', '17307.png', '14985.png', '00862.png', '00196.png', '13777.png', '07426.png', '20051.png', '11370.png', '01140.png', '01899.png', '22151.png', '15719.png', '08352.png', '09581.png', '00559.png', '16951.png', '08575.png', '15793.png', '02982.png', '08040.png', '24121.png', '18601.png', '11573.png', '08146.png', '16708.png', '22065.png', '14631.png', '22326.png', '22116.png', '18645.png', '02394.png', '10536.png', '05299.png', '10305.png', '18983.png', '01187.png', '19688.png', '12120.png', '11883.png', '19214.png', '04580.png', '05435.png', '15302.png', '23049.png', '01067.png', '12766.png', '09769.png', '01053.png', '16268.png', '01138.png', '02056.png', '20190.png', '14648.png', '19822.png', '21533.png', '10997.png', '04774.png', '18517.png', '16784.png', '16024.png', '20857.png', '07318.png', '19804.png', '16135.png', '13662.png', '21945.png', '21266.png', '23547.png', '19845.png', '01580.png', '23443.png', '15582.png', '09834.png', '01059.png', '00814.png', '19728.png', '00284.png', '04404.png', '19930.png', '14344.png', '12599.png', '01076.png', '15796.png', '05937.png', '09140.png', '02383.png', '01018.png', '24310.png', '01954.png', '01334.png', '16704.png', '05565.png', '09261.png', '11612.png', '00672.png', '08487.png', '11939.png', '19444.png', '12133.png', '24132.png', '08116.png', '03873.png', '19202.png', '16988.png', '17066.png', '10001.png', '21229.png', '10537.png', '19094.png', '18383.png', '01739.png', '20513.png', '16911.png', '10845.png', '17469.png', '16711.png', '24838.png', '05892.png', '08295.png', '12483.png', '20068.png', '21774.png', '02380.png', '11135.png', '18478.png', '09533.png', '15359.png', '14860.png', '23192.png', '12408.png', '24579.png', '01585.png', '10273.png', '16851.png', '00596.png', '11037.png', '02584.png', '03246.png', '19465.png', '12336.png', '00650.png', '24029.png', '05671.png', '02517.png', '02448.png', '19338.png', '05556.png', '12207.png', '20848.png', '02349.png', '06142.png', '01569.png', '21461.png', '22313.png', '12593.png', '09752.png', '12788.png', '17269.png', '24383.png', '09466.png', '17411.png', '19167.png', '23750.png', '03963.png', '17284.png', '01080.png', '10766.png', '05975.png', '18373.png', '18648.png', '11687.png', '09028.png', '17540.png', '17459.png', '02381.png', '21449.png', '19243.png', '00985.png', '17627.png', '22783.png', '22437.png', '15018.png', '21645.png', '18465.png', '16841.png', '11703.png', '06901.png', '18406.png', '15640.png', '01064.png', '24380.png', '14719.png', '05255.png', '20092.png', '19327.png', '00012.png', '22083.png', '24376.png', '23808.png', '01224.png', '22963.png', '01644.png', '10642.png', '04169.png', '07617.png', '08035.png', '06210.png', '17883.png', '09835.png', '09361.png', '23491.png', '08867.png', '03501.png', '07758.png', '05082.png', '15295.png', '16468.png', '18019.png', '24084.png', '12229.png', '02791.png', '21052.png', '07209.png', '02439.png', '00632.png', '19187.png', '24843.png', '24607.png', '07977.png', '16580.png', '12337.png', '01538.png', '17408.png', '24248.png', '24635.png', '00400.png', '06216.png', '17137.png', '06276.png', '08740.png', '15470.png', '22322.png', '11921.png', '00878.png', '22539.png', '16044.png', '23583.png', '11911.png', '20777.png', '23592.png', '18564.png', '06099.png', '09674.png', '19360.png', '15447.png', '16730.png', '08000.png', '12630.png', '22925.png', '22741.png', '11734.png', '15567.png', '05659.png', '07673.png', '23595.png', '00688.png', '10244.png', '09766.png', '14654.png', '11737.png', '01804.png', '06883.png', '22150.png', '05099.png', '18139.png', '09552.png', '06736.png', '21914.png', '01244.png', '02715.png', '03749.png', '02206.png', '23474.png', '00537.png', '03047.png', '07089.png', '11269.png', '05813.png', '08240.png', '10960.png', '15092.png', '02496.png', '05668.png', '21573.png', '02879.png', '04216.png', '10406.png', '17962.png', '08643.png', '24850.png', '08947.png', '05473.png', '06035.png', '14016.png', '03463.png', '04660.png', '08856.png', '05180.png', '07850.png', '02210.png', '00828.png', '21770.png', '13488.png', '10980.png', '16882.png', '11569.png', '22419.png', '11497.png', '10589.png', '14896.png', '18844.png', '02914.png', '01998.png', '03045.png', '10755.png', '15728.png', '02191.png', '15484.png', '11999.png', '12491.png', '10562.png', '10809.png', '22096.png', '10039.png', '17140.png', '03751.png', '24171.png', '24113.png', '08734.png', '21397.png', '05516.png', '03295.png', '14821.png', '12873.png', '10694.png', '22033.png', '07077.png', '17949.png', '24151.png', '09010.png', '02251.png', '12898.png', '23823.png', '04541.png', '17852.png', '16037.png', '10264.png', '01196.png', '05070.png', '23900.png', '16141.png', '24128.png', '22057.png', '19984.png', '23196.png', '17003.png', '06384.png', '04771.png', '03992.png', '22518.png', '23579.png', '01784.png', '20189.png', '10896.png', '11046.png', '07536.png', '07965.png', '08320.png', '02015.png', '19256.png', '07975.png', '12837.png', '18350.png', '24721.png', '00638.png', '20682.png', '11391.png', '19685.png', '16509.png', '22385.png', '19923.png', '23074.png', '24743.png', '01898.png', '14220.png', '24684.png', '01236.png', '10518.png', '19605.png', '06147.png', '15834.png', '08367.png', '00721.png', '06897.png', '12310.png', '16536.png', '21700.png', '08223.png', '07068.png', '16391.png', '20281.png', '16670.png', '16588.png', '24135.png', '21391.png', '10836.png', '07112.png', '06547.png', '23110.png', '20461.png', '23557.png', '14313.png', '00302.png', '07149.png', '02920.png', '13528.png', '18616.png', '06457.png', '24631.png', '05441.png', '01944.png', '16885.png', '13273.png', '22687.png', '08696.png', '23305.png', '11443.png', '19579.png', '10082.png', '22442.png', '07311.png', '18515.png', '02520.png', '18757.png', '16132.png', '20824.png', '09195.png', '01703.png', '13965.png', '18998.png', '12965.png', '16438.png', '04892.png', '03036.png', '02033.png', '08526.png', '06765.png', '23646.png', '17233.png', '11147.png', '18807.png', '04417.png', '02266.png', '07096.png', '19410.png', '12329.png', '14840.png', '03255.png', '22966.png', '05443.png', '01502.png', '15333.png', '01362.png', '15850.png', '01443.png', '17926.png', '02295.png', '02025.png', '10788.png', '09749.png', '02725.png', '15171.png', '06151.png', '00723.png', '06418.png', '24194.png', '02780.png', '14297.png', '20681.png', '11502.png', '16198.png', '01637.png', '23166.png', '21272.png', '10482.png', '00037.png', '20865.png', '09062.png', '05963.png', '16175.png', '09092.png', '01231.png', '10455.png', '10095.png', '15118.png', '22061.png', '08679.png', '06848.png', '10852.png', '02205.png', '13491.png', '11310.png', '20581.png', '17527.png', '13348.png', '17020.png', '09297.png', '19612.png', '04557.png', '21680.png', '07934.png', '05492.png', '24178.png', '05769.png', '01605.png', '18397.png', '19543.png', '15149.png', '04721.png', '03279.png', '12667.png', '06076.png', '15743.png', '12459.png', '10552.png', '08090.png', '07714.png', '15785.png', '07900.png', '11796.png', '03226.png', '14346.png', '01768.png', '02092.png', '18431.png', '11364.png', '04013.png', '11139.png', '14501.png', '03483.png', '12500.png', '13428.png', '23106.png', '22327.png', '17533.png', '16540.png', '01431.png', '06358.png', '20071.png', '24575.png', '04747.png', '08024.png', '05982.png', '24276.png', '24111.png', '21034.png', '11662.png', '08907.png', '04240.png', '10411.png', '22479.png', '23279.png', '10365.png', '03719.png', '17453.png', '06317.png', '14634.png', '05405.png', '04007.png', '15363.png', '24822.png', '23584.png', '15158.png', '19093.png', '13253.png', '15253.png', '10955.png', '12995.png', '06046.png', '11822.png', '24887.png', '19162.png', '03288.png', '16053.png', '06280.png', '13485.png', '19272.png', '19298.png', '01607.png', '15635.png', '22887.png', '06611.png', '23916.png', '05979.png', '01916.png', '08417.png', '03298.png', '03834.png', '16603.png', '15720.png', '09512.png', '19853.png', '22977.png', '20325.png', '00599.png', '06050.png', '14744.png', '13071.png', '09617.png', '14438.png', '24206.png', '19169.png', '02575.png', '02735.png', '12152.png', '19855.png', '21723.png', '23635.png', '05927.png', '16058.png', '00592.png', '02256.png', '18249.png', '17450.png', '20120.png', '07173.png', '15995.png', '12373.png', '21197.png', '17277.png', '13252.png', '19769.png', '04946.png', '07846.png', '21376.png', '23973.png', '15081.png', '19556.png', '24549.png', '21255.png', '14937.png', '20674.png', '18947.png', '17191.png', '13927.png', '15808.png', '14186.png', '06531.png', '12587.png', '07101.png', '04860.png', '07715.png', '15913.png', '10382.png', '00956.png', '20265.png', '22550.png', '06002.png', '10910.png', '09788.png', '09828.png', '06852.png', '21005.png', '11694.png', '13028.png', '00699.png', '12885.png', '11271.png', '05150.png', '22258.png', '00173.png', '15292.png', '21334.png', '12811.png', '19977.png', '03111.png', '23496.png', '23010.png', '03415.png', '06878.png', '23987.png', '03558.png', '07435.png', '10568.png', '23256.png', '10274.png', '17129.png', '02137.png', '04124.png', '24876.png', '24771.png', '01358.png', '15311.png', '00332.png', '00436.png', '17169.png', '16211.png', '12794.png', '23158.png', '09267.png', '16573.png', '08703.png', '21130.png', '09222.png', '23890.png', '03128.png', '07431.png', '03776.png', '04301.png', '20554.png', '15677.png', '23037.png', '03923.png', '23727.png', '16126.png', '20252.png', '17155.png', '16788.png', '22415.png', '22315.png', '17597.png', '19934.png', '15455.png', '06619.png', '04645.png', '17688.png', '21031.png', '07240.png', '01126.png', '10224.png', '13576.png', '17624.png', '00050.png', '14676.png', '12584.png', '05152.png', '11417.png', '16651.png', '12371.png', '18359.png', '17135.png', '04289.png', '24315.png', '19554.png', '16916.png', '13595.png', '04362.png', '02059.png', '07889.png', '20640.png', '19890.png', '08640.png', '20817.png', '07716.png', '20242.png', '07622.png', '03565.png', '13102.png', '04738.png', '01852.png', '17222.png', '14094.png', '09333.png', '11486.png', '15943.png', '03202.png', '02353.png', '10464.png', '22869.png', '00952.png', '17035.png', '22590.png', '08458.png', '04159.png', '05485.png', '24594.png', '12764.png', '02200.png', '13988.png', '07269.png', '01971.png', '15874.png', '14485.png', '03335.png', '19957.png', '05709.png', '16507.png', '15592.png', '19931.png', '05641.png', '22520.png', '19486.png', '00063.png', '19432.png', '03915.png', '05005.png', '13946.png', '24675.png', '20415.png', '24518.png', '12075.png', '18346.png', '18023.png', '01544.png']
    epochs_since_start = 0
    if True:
        data_loader = get_loader('gta')
        data_path = get_data_path('gta')
        if random_crop:
            data_aug = Compose([RandomCrop_gta(input_size)])
        else:
            data_aug = None
        #data_aug = Compose([RandomHorizontallyFlip()])
        train_dataset = data_loader(data_path, list_path = './data/gta5_list/train.txt', augmentations=data_aug, img_size=(1280,720), mean=IMG_MEAN, a = a)

    trainloader = data.DataLoader(train_dataset,
                    batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    trainloader_iter = iter(trainloader)
    print('gta size:',len(trainloader))
    #Load new data for domain_transfer

    # optimizer for segmentation network
    learning_rate_object = Learning_Rate_Object(config['training']['learning_rate'])

    if optimizer_type == 'SGD':
        if len(gpus) > 1:
            optimizer = optim.SGD(model.module.optim_parameters(learning_rate_object),
                        lr=learning_rate, momentum=momentum,weight_decay=weight_decay)
        else:
            optimizer = optim.SGD(model.optim_parameters(learning_rate_object),
                        lr=learning_rate, momentum=momentum,weight_decay=weight_decay)
    elif optimizer_type == 'Adam':
        if len(gpus) > 1:
            optimizer = optim.Adam(model.module.optim_parameters(learning_rate_object),
                        lr=learning_rate, momentum=momentum,weight_decay=weight_decay)
        else:
            optimizer = optim.Adam(model.optim_parameters(learning_rate_object),
                        lr=learning_rate, weight_decay=weight_decay)

    optimizer.zero_grad()
    model.cuda()
    model.train()
    #prototype_dist_init(cfg, trainloader, model)
    feat_estimator = prototype_dist_estimator(feature_num=feature_num, cfg=cfg)
    if cfg.SOLVER.MULTI_LEVEL:
        out_estimator = prototype_dist_estimator(feature_num=cfg.MODEL.NUM_CLASSES, cfg=cfg)
    interp = nn.Upsample(size=(input_size[0], input_size[1]), mode='bilinear', align_corners=True)
    start_iteration = 0
    if args.resume:
        start_iteration, model, optimizer, ema_model = _resume_checkpoint(args.resume, model, optimizer, ema_model)
    
    """
    if True:
        model.eval()
        if dataset == 'cityscapes':
            mIoU, eval_loss = evaluate(model, dataset, ignore_label=250, input_size=(512,1024))

        model.train()
        print("mIoU: ",mIoU, eval_loss)
    """
    
    accumulated_loss_l = []
    accumulated_loss_u = []
    accumulated_loss_feat = []
    accumulated_loss_out = []   
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    with open(checkpoint_dir + '/config.json', 'w') as handle:
        json.dump(config, handle, indent=4, sort_keys=True)

    
    print(epochs_since_start)
    for i_iter in range(start_iteration, num_iterations):
        model.train()

        loss_u_value = 0
        loss_l_value = 0
        loss_feat_value = 0
        loss_out_value = 0

        optimizer.zero_grad()

        if lr_schedule:
            adjust_learning_rate(optimizer, i_iter)

        # training loss for labeled data only
        try:
            batch = next(trainloader_iter)
            if batch[0].shape[0] != batch_size:
                batch = next(trainloader_iter)
        except:
            epochs_since_start = epochs_since_start + 1
            print('Epochs since start: ',epochs_since_start)
            if epochs_since_start >= 2:
                list_name = []
            if epochs_since_start == 1:
                data_loader = get_loader('gta')
                data_path = get_data_path('gta')
                if random_crop:
                    data_aug = Compose([RandomCrop_gta(input_size)])
                else:
                    data_aug = None        
                train_dataset = data_loader(data_path, list_path = './data/gta5_list/train.txt', augmentations=data_aug, img_size=(1280,720), mean=IMG_MEAN, a = None)
                trainloader = data.DataLoader(train_dataset,
                    batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
                print('gta size:',len(trainloader))
            trainloader_iter = iter(trainloader)
            batch = next(trainloader_iter)

        #if random_flip:
        #    weak_parameters={"flip":random.randint(0,1)}
        #else:
        
        weak_parameters={"flip": 0}


        images, labels, _, names = batch
        images = images.cuda()
        labels = labels.cuda().long()
        if epochs_since_start >= 2:
            for name in names:
                list_name.append(name)

        #images, labels = weakTransform(weak_parameters, data = images, target = labels)

        src_pred, src_feat= model(images)
        pred = interp(src_pred)
        L_l = loss_calc(pred, labels) # Cross entropy loss for labeled data
        #L_l = torch.Tensor([0.0]).cuda()

        if train_unlabeled:
            try:
                batch_remain = next(trainloader_remain_iter)
                if batch_remain[0].shape[0] != batch_size:
                    batch_remain = next(trainloader_remain_iter)
            except:
                trainloader_remain_iter = iter(trainloader_remain)
                batch_remain = next(trainloader_remain_iter)

            images_remain, _, _, _, _ = batch_remain
            images_remain = images_remain.cuda()
            inputs_u_w, _ = weakTransform(weak_parameters, data = images_remain)
            #inputs_u_w = inputs_u_w.clone()
            logits_u_w_65 = (ema_model(inputs_u_w)[0])
            logits_u_w = interp(logits_u_w_65)
            logits_u_w, _ = weakTransform(getWeakInverseTransformParameters(weak_parameters), data = logits_u_w.detach())
            logits_u_w_65, _ = weakTransform(getWeakInverseTransformParameters(weak_parameters), data = logits_u_w_65.detach()) 
            pseudo_label = torch.softmax(logits_u_w.detach(), dim=1)
            max_probs, targets_u_w = torch.max(pseudo_label, dim=1)
            pseudo_label_65 = torch.softmax(logits_u_w_65.detach(), dim=1)
            max_probs_65, targets_u_w_65 = torch.max(pseudo_label_65, dim=1)

            if mix_mask == "class":
                for image_i in range(batch_size):
                    classes = torch.unique(labels[image_i])
                    #classes=classes[classes!=ignore_label]
                    nclasses = classes.shape[0]
                    #if nclasses > 0:
                    classes = (classes[torch.Tensor(np.random.choice(nclasses, int((nclasses+nclasses%2)/2),replace=False)).long()]).cuda()

                    if image_i == 0:
                        MixMask0 = transformmasks.generate_class_mask(labels[image_i], classes).unsqueeze(0).cuda()
                    else:
                        MixMask1 = transformmasks.generate_class_mask(labels[image_i], classes).unsqueeze(0).cuda()

            elif mix_mask == None:
                MixMask = torch.ones((inputs_u_w.shape))

            strong_parameters = {"Mix": MixMask0}
            if random_flip:
                strong_parameters["flip"] = random.randint(0, 1)
            else:
                strong_parameters["flip"] = 0
            if color_jitter:
                strong_parameters["ColorJitter"] = random.uniform(0, 1)
            else:
                strong_parameters["ColorJitter"] = 0
            if gaussian_blur:
                strong_parameters["GaussianBlur"] = random.uniform(0, 1)
            else:
                strong_parameters["GaussianBlur"] = 0

            inputs_u_s0, _ = strongTransform(strong_parameters, data = torch.cat((images[0].unsqueeze(0),images_remain[0].unsqueeze(0))))
            strong_parameters["Mix"] = MixMask1
            inputs_u_s1, _ = strongTransform(strong_parameters, data = torch.cat((images[1].unsqueeze(0),images_remain[1].unsqueeze(0))))
            inputs_u_s = torch.cat((inputs_u_s0,inputs_u_s1))
            logits_u_s_tgt, tgt_feat = model(inputs_u_s)
            logits_u_s = interp(logits_u_s_tgt)

            strong_parameters["Mix"] = MixMask0
            _, targets_u0 = strongTransform(strong_parameters, target = torch.cat((labels[0].unsqueeze(0),targets_u_w[0].unsqueeze(0))))
            strong_parameters["Mix"] = MixMask1
            _, targets_u1 = strongTransform(strong_parameters, target = torch.cat((labels[1].unsqueeze(0),targets_u_w[1].unsqueeze(0))))
            targets_u = torch.cat((targets_u0,targets_u1)).long()
            
            if pixel_weight == "threshold_uniform":
                unlabeled_weight = torch.sum(max_probs.ge(0.968).long() == 1).item() / np.size(np.array(targets_u.cpu()))
                pixelWiseWeight = unlabeled_weight * torch.ones(max_probs.shape).cuda()
            elif pixel_weight == "threshold":
                pixelWiseWeight = max_probs.ge(0.968).float().cuda()
            elif pixel_weight == False:
                pixelWiseWeight = torch.ones(max_probs.shape).cuda()

            onesWeights = torch.ones((pixelWiseWeight.shape)).cuda()
            strong_parameters["Mix"] = MixMask0
            _, pixelWiseWeight0 = strongTransform(strong_parameters, target = torch.cat((onesWeights[0].unsqueeze(0),pixelWiseWeight[0].unsqueeze(0))))
            strong_parameters["Mix"] = MixMask1
            _, pixelWiseWeight1 = strongTransform(strong_parameters, target = torch.cat((onesWeights[1].unsqueeze(0),pixelWiseWeight[1].unsqueeze(0))))
            pixelWiseWeight = torch.cat((pixelWiseWeight0,pixelWiseWeight1)).cuda()

            if consistency_loss == 'MSE':
                unlabeled_weight = torch.sum(max_probs.ge(0.968).long() == 1).item() / np.size(np.array(targets_u.cpu()))
                #pseudo_label = torch.cat((pseudo_label[1].unsqueeze(0),pseudo_label[0].unsqueeze(0)))
                L_u = consistency_weight * unlabeled_weight * unlabeled_loss(logits_u_s, pseudo_label)
            elif consistency_loss == 'CE':
                L_u = consistency_weight * unlabeled_loss(logits_u_s, targets_u, pixelWiseWeight)

            loss = L_l + L_u

        else:
            loss = L_l
        if i_iter >= 8940:
            # source mask: downsample the ground-truth label
            src_out_ema, src_feat_ema = ema_model(images)
            tgt_out_ema, tgt_feat_ema = ema_model(inputs_u_s)
            B, A, Hs, Ws = src_feat.size()
            src_mask = F.interpolate(labels.unsqueeze(0).float(), size=(Hs, Ws), mode='nearest').squeeze(0).long()
            src_mask = src_mask.contiguous().view(B * Hs * Ws, )
            assert not src_mask.requires_grad
            pixelWiseWeight[pixelWiseWeight != 1.0] == 0.0
            pseudo_weight = F.interpolate(pixelWiseWeight.unsqueeze(1),
                                            size=(65,65), mode='nearest').squeeze(1)
            _, _, Ht, Wt = tgt_feat.size()
            tgt_mask = F.interpolate(targets_u.unsqueeze(1).float(), size=(65,65), mode='nearest').squeeze(1).long()
            tgt_mask_upt = copy.deepcopy(tgt_mask)
            
            for i in range(cfg.MODEL.NUM_CLASSES):
                tgt_mask_upt[(((max_probs_65 < cfg.SOLVER.DELTA) * (targets_u_w_65 == i)).int() + (pseudo_weight != 1.0).int()) == 2] = 255
            if i_iter > 89600:
                print("pw: ",(pseudo_weight != 1.0).sum())
                print("mask_255:", (tgt_mask_upt == 255).sum())
                for i in range(cfg.MODEL.NUM_CLASSES):
                    print(i, ((max_probs_65 >= cfg.SOLVER.DELTA) * (targets_u_w_65 == i)).sum(), ((max_probs_65 <= cfg.SOLVER.DELTA) * (targets_u_w_65 == i)).sum())
            if i_iter == 8960:
                print(max_probs_65[1,30,30], max_probs_65[0,30,30])
            tgt_mask_upt = tgt_mask_upt.contiguous().view(B * Hs * Ws, )
            src_feat = src_feat.permute(0, 2, 3, 1).contiguous().view(B * Hs * Ws, A)
            tgt_feat = tgt_feat.permute(0, 2, 3, 1).contiguous().view(B * Ht * Wt, A)
            src_feat_ema = src_feat_ema.permute(0, 2, 3, 1).contiguous().view(B * Hs * Ws, A)
            tgt_feat_ema = tgt_feat_ema.permute(0, 2, 3, 1).contiguous().view(B * Ht * Wt, A)

            # update feature-level statistics
            feat_estimator.update(features=tgt_feat_ema.detach(), labels=tgt_mask_upt)
            feat_estimator.update(features=src_feat_ema.detach(), labels=src_mask)

            # contrastive loss on both domains
            loss_feat = pcl_criterion_src(Proto=feat_estimator.Proto.detach(),
                                        feat=src_feat,
                                        labels=src_mask) \
                        + pcl_criterion_tgt(Proto=feat_estimator.Proto.detach(),
                                        feat=tgt_feat,
                                        labels=tgt_mask_upt)
            
            #meters.update(loss_feat=loss_feat.item())
            if i_iter == 8940 or i_iter == 8941:
                    pcl_criterion_tgt(Proto=feat_estimator.Proto.detach(),
                                        feat=tgt_feat,
                                        labels=tgt_mask_upt, aa = True)
            if cfg.SOLVER.MULTI_LEVEL:
                src_out = src_pred.permute(0, 2, 3, 1).contiguous().view(B * Hs * Ws, cfg.MODEL.NUM_CLASSES)
                tgt_out = logits_u_s_tgt.permute(0, 2, 3, 1).contiguous().view(B * Ht * Wt, cfg.MODEL.NUM_CLASSES)
                src_out_ema = src_out_ema.permute(0, 2, 3, 1).contiguous().view(B * Hs * Ws, cfg.MODEL.NUM_CLASSES)
                tgt_out_ema = tgt_out_ema.permute(0, 2, 3, 1).contiguous().view(B * Ht * Wt, cfg.MODEL.NUM_CLASSES)
                # update output-level statistics
                out_estimator.update(features=tgt_out_ema.detach(), labels=tgt_mask_upt)
                out_estimator.update(features=src_out_ema.detach(), labels=src_mask)

                # the proposed contrastive loss on prediction map
                loss_out = pcl_criterion_src(Proto=out_estimator.Proto.detach(),
                                        feat=src_out,
                                        labels=src_mask) \
                        + pcl_criterion_tgt(Proto=out_estimator.Proto.detach(),
                                        feat=tgt_out,
                                        labels=tgt_mask_upt)
                #meters.update(loss_out=loss_out.item())


                loss = loss + cfg.SOLVER.LAMBDA_FEAT * loss_feat + cfg.SOLVER.LAMBDA_OUT * loss_out
            else:
                loss = loss + cfg.SOLVER.LAMBDA_FEAT * loss_feat
            
            loss_feat_value += loss_feat.item()
            loss_out_value += loss_out.item()

        if len(gpus) > 1:
            #print('before mean = ',loss)
            loss = loss.mean()
            #print('after mean = ',loss)
            loss_l_value += L_l.mean().item()
            if train_unlabeled:
                loss_u_value += L_u.mean().item()
        else:
            loss_l_value += L_l.item()
            if train_unlabeled:
                loss_u_value += L_u.item()
        loss.backward()
        optimizer.step()

        # update Mean teacher network
        if ema_model is not None:
            alpha_teacher = 0.99
            ema_model = update_ema_variables(ema_model = ema_model, model = model, alpha_teacher=alpha_teacher, iteration=i_iter)

        #print('iter = {0:6d}/{1:6d}, loss_l = {2:.3f}, loss_u = {3:.3f}'.format(i_iter, num_iterations, loss_l_value, loss_u_value))

        if i_iter % save_checkpoint_every == 1486 and i_iter!=0:
            _save_checkpoint(i_iter, model, optimizer, config, ema_model, overwrite=False)
            feat_estimator.save(name='prototype_feat_dist.pth')
            out_estimator.save(name='prototype_out_dist.pth')
            print('save_prototype')

        if i_iter == 56505:
            print(list_name)
        if config['utils']['tensorboard']:
            if 'tensorboard_writer' not in locals():
                tensorboard_writer = tensorboard.SummaryWriter(log_dir, flush_secs=30)

        accumulated_loss_l.append(loss_l_value)
        if train_unlabeled:
            accumulated_loss_u.append(loss_u_value)
        if i_iter >= 9000:
            accumulated_loss_feat.append(loss_feat_value)
            accumulated_loss_out.append(loss_out_value)
            
        if i_iter % log_per_iter == 0 and i_iter != 0:
                #tensorboard_writer.add_scalar('Training/Supervised loss', np.mean(accumulated_loss_l), i_iter)
            if i_iter >= 9000:
                print('Training/contrastive_feat_loss', np.mean(accumulated_loss_feat), 'Training/contrastive_out_loss', np.mean(accumulated_loss_out), i_iter)
                accumulated_loss_feat = []
                accumulated_loss_out = []
            if train_unlabeled:
                #tensorboard_writer.add_scalar('Training/Unsupervised loss', np.mean(accumulated_loss_u), i_iter)
                print('Training/Supervised loss', np.mean(accumulated_loss_l), 'Training/Unsupervised loss', np.mean(accumulated_loss_u), i_iter)
                accumulated_loss_u = []
                accumulated_loss_l = []
            
        if save_unlabeled_images and train_unlabeled and (i_iter == 5650600):
            # Saves two mixed images and the corresponding prediction
            save_image(inputs_u_s[0].cpu(),i_iter,'input_s1',palette.CityScpates_palette)
            save_image(inputs_u_s[1].cpu(),i_iter,'input_s2',palette.CityScpates_palette)
            save_image(inputs_u_w[0].cpu(),i_iter,'input_w1',palette.CityScpates_palette)
            save_image(inputs_u_w[1].cpu(),i_iter,'input_w2',palette.CityScpates_palette)
            save_image(images[0].cpu(),i_iter,'input1',palette.CityScpates_palette)
            save_image(images[1].cpu(),i_iter,'input2',palette.CityScpates_palette)

            _, pred_u_s = torch.max(logits_u_w, dim=1)
            #save_image(pred_u_s[0].cpu(),i_iter,'pred1',palette.CityScpates_palette)
            #save_image(pred_u_s[1].cpu(),i_iter,'pred2',palette.CityScpates_palette)

    _save_checkpoint(num_iterations, model, optimizer, config, ema_model)

    model.eval()
    if dataset == 'cityscapes':
        mIoU, val_loss = evaluate(model, dataset, ignore_label=250, input_size=(512,1024), save_dir=checkpoint_dir)
    model.train()
    if mIoU > best_mIoU and save_best_model:
        best_mIoU = mIoU
        _save_checkpoint(i_iter, model, optimizer, config, ema_model, save_best=True)

    if config['utils']['tensorboard']:
        tensorboard_writer.add_scalar('Validation/mIoU', mIoU, i_iter)
        tensorboard_writer.add_scalar('Validation/Loss', val_loss, i_iter)


    end = timeit.default_timer()
    print('Total time: ' + str(end-start) + 'seconds')

if __name__ == '__main__':

    print('---------------------------------Starting---------------------------------')

    args = get_arguments()
    

    if False:#args.resume:
        config = torch.load(args.resume)['config']
    else:
        config = json.load(open(args.config))

    model = config['model']
    dataset = config['dataset']


    if config['pretrained'] == 'coco':
        restore_from = 'http://vllab1.ucmerced.edu/~whung/adv-semi-seg/resnet101COCO-41f33a49.pth'

    num_classes=19
    IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

    batch_size = config['training']['batch_size']
    num_iterations = config['training']['num_iterations']

    input_size_string = config['training']['data']['input_size']
    h, w = map(int, input_size_string.split(','))
    input_size = (h, w)

    ignore_label = config['ignore_label'] 

    learning_rate = config['training']['learning_rate']

    optimizer_type = config['training']['optimizer']
    lr_schedule = config['training']['lr_schedule']
    lr_power = config['training']['lr_schedule_power']
    weight_decay = config['training']['weight_decay']
    momentum = config['training']['momentum']
    num_workers = config['training']['num_workers']
    use_sync_batchnorm = config['training']['use_sync_batchnorm']
    random_seed = config['seed']

    labeled_samples = config['training']['data']['labeled_samples']

    #unlabeled CONFIGURATIONS
    train_unlabeled = config['training']['unlabeled']['train_unlabeled']
    mix_mask = config['training']['unlabeled']['mix_mask']
    pixel_weight = config['training']['unlabeled']['pixel_weight']
    consistency_loss = config['training']['unlabeled']['consistency_loss']
    consistency_weight = config['training']['unlabeled']['consistency_weight']
    random_flip = config['training']['unlabeled']['flip']
    color_jitter = config['training']['unlabeled']['color_jitter']
    gaussian_blur = config['training']['unlabeled']['blur']

    random_scale = config['training']['data']['scale']
    random_crop = config['training']['data']['crop']

    save_checkpoint_every = config['utils']['save_checkpoint_every']
    if args.resume:
        checkpoint_dir = os.path.join(*args.resume.split('/')[:-1]) + '_resume-' + start_writeable
    else:
        checkpoint_dir = os.path.join(config['utils']['checkpoint_dir'], start_writeable + '-' + args.name)
    log_dir = checkpoint_dir

    val_per_iter = config['utils']['val_per_iter']
    use_tensorboard = config['utils']['tensorboard']
    log_per_iter = config['utils']['log_per_iter']

    save_best_model = config['utils']['save_best_model']
    #if args.save_images:
    print('Saving unlabeled images')
    save_unlabeled_images = True
    #else:
    #save_unlabeled_images = False

    gpus = (0,1,2,3)[:args.gpus]    
    cfg.merge_from_file(args.config_file)
    #cfg.merge_from_list(args.opts)
    cfg.freeze()


    main()
