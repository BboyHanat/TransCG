import os
import yaml
import torch
import random
import logging
import argparse
import warnings

import numpy as np
import torch.nn as nn
from tqdm import tqdm
from prettytable import PrettyTable
import torch.backends.cudnn as cudnn
from utils.logger import ColoredLogger
from utils.builder import ConfigBuilder
from utils.constants import LOSS_INF
from utils.functions import display_results, to_device
from time import perf_counter
import torch.multiprocessing as mp
from torch.nn import SyncBatchNorm, parallel
from torch.utils.tensorboard import SummaryWriter
from utils.dist_util import reduce_mean, accuracy, AverageMeter


torch.multiprocessing.set_sharing_strategy('file_system')
logging.setLoggerClass(ColoredLogger)
logger = logging.getLogger(__name__)
warnings.simplefilter("ignore", UserWarning)

def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Speed-reproducibility tradeoff
    # https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True


def dist_trainer(local_rank, dist_num: int, config: dict):
    """

    :param local_rank:
    :param dist_num:
    :param config: distribute training parameters
    :return:
    """
    train_cfg = config['train_cfg']
    init_seeds(local_rank + 1, cuda_deterministic=False)
    init_method = 'tcp://' + config['dist_cfg']['ip'] + ':' + str(config['dist_cfg']['port'])
    dist.init_process_group(backend='nccl',  # noqa
                            init_method=init_method,
                            world_size=dist_num,
                            rank=local_rank)

    builder = ConfigBuilder(**config)
    logger.info('Building models ...')
    network_model = builder.get_model()

    logger.info('Checking checkpoints ...')
    start_epoch = 0
    checkpoint_loss = 10000000000.0
    max_epoch = builder.get_max_epoch()
    stats_dir = builder.get_stats_dir()
    checkpoint_file = os.path.join(stats_dir, 'checkpoint.tar')


    if os.path.isfile(checkpoint_file):
        checkpoint = torch.load(checkpoint_file)
        network_model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        checkpoint_metrics = checkpoint['metrics']
        checkpoint_loss = checkpoint['loss']
        logger.info("Checkpoint {} (epoch {}) loaded.".format(checkpoint_file, start_epoch))

    logger.info('Building optimizer and learning rate schedulers ...')
    resume = (start_epoch > 0)
    optimizer = builder.get_optimizer(network_model, resume=resume, resume_lr=builder.get_resume_lr())

    # convert model to ddp
    network_model = SyncBatchNorm.convert_sync_batchnorm(network_model).to(local_rank)
    network_model = parallel.DistributedDataParallel(network_model,
                                                     device_ids=[local_rank])
    logger.info('Building ddp dataloaders ...')
    train_dataloader, len_of_train = builder.get_dataloader(split='train')
    test_dataloader, len_of_val = builder.get_dataloader(split='test')

    logger.info('Building optimizer and learning rate schedulers ...')
    resume = (start_epoch > 0)

    lr_scheduler = builder.get_lr_scheduler(optimizer, resume=resume,
                                            resume_epoch=(start_epoch - 1 if resume else None))
    criterion = builder.get_criterion()
    metrics = builder.get_metrics()
    summary_writer = builder.get_summary_writer(**config['summary_cfg'])

    # start training
    if start_epoch != 0:
        min_loss = checkpoint_loss
        min_loss_epoch = start_epoch
    else:
        min_loss = LOSS_INF
        min_loss_epoch = None

    train_steps = int(len_of_train / train_cfg['train_batch'] / dist_num)

    for epoch in range(start_epoch, max_epoch):
        logger.info('--> Epoch {}/{}'.format(epoch + 1, max_epoch))
        train_one_epoch(network_model, optimizer,
                        criterion, train_dataloader,
                        lr_scheduler, summary_writer,
                        local_rank, train_steps,
                        epoch, dist_num)
        loss, metrics_result = test_one_epoch(network_model, criterion,
                                              train_dataloader, metrics,
                                              summary_writer,  local_rank,
                                              epoch, dist_num)
        if lr_scheduler is not None:
            lr_scheduler.step()
        criterion.step()
        save_dict = {
            'epoch': epoch + 1,
            'model_state_dict': network_model.module.state_dict(),
            'loss': loss,
            'metrics': metrics_result
        }
        torch.save(save_dict, os.path.join(stats_dir, 'checkpoint-epoch{}.tar'.format(epoch)))
        if loss < min_loss:
            min_loss = loss
            min_loss_epoch = epoch + 1
            torch.save(save_dict, os.path.join(stats_dir, 'checkpoint.tar'.format(epoch)))
    logger.info('Training Finished. Min testing loss: {:.6f}, in epoch {}'.format(min_loss, min_loss_epoch))



def train_one_epoch(network_model:nn.Module,
                    optimizer,
                    criterion,
                    train_dataloader,
                    lr_scheduler,
                    summary_writer,
                    local_rank: int,
                    train_steps: int,
                    epoch: int,
                    dist_num: int):
    logger.info('Start training process in epoch {}.'.format(epoch + 1))
    if lr_scheduler is not None:
        logger.info('Learning rate: {}.'.format(lr_scheduler.get_last_lr()))
    network_model.train()
    with tqdm(train_dataloader) as pbar:
        train_step = 0
        for data_dict in pbar:
            optimizer.zero_grad()
            gpu_device = torch.device('cuda:{}'.format(local_rank))
            data_dict = to_device(data_dict, gpu_device)
            res = network_model(data_dict['rgb'], data_dict['depth'])
            depth_scale = data_dict['depth_max'] - data_dict['depth_min']
            res = res * depth_scale.reshape(-1, 1, 1) + data_dict['depth_min'].reshape(-1, 1, 1)
            data_dict['pred'] = res
            loss_dict = criterion(data_dict)
            loss = loss_dict['loss']
            loss.backward()
            optimizer.step()
            torch.distributed.barrier()  # noqa
            reduced_loss = dict()
            for key in loss_dict.keys():
                reduced_loss[key] = reduce_mean(loss_dict[key], dist_num)
            if local_rank == 0:
                description = 'Epoch {} Step {}, '.format(epoch + 1, train_step)
                for key in reduced_loss.keys():
                    description = description + key + ": {:.8f}".format(reduced_loss[key].item()) + "  "
                pbar.set_description(description)
                for key in loss_dict.keys():
                    reduced_loss[key] = reduce_mean(loss_dict[key], dist_num)
                    summary_writer.add_scalar("train/"+key,
                                              reduced_loss[key].data.cpu().numpy(),
                                              global_step=epoch * train_steps + train_step)
            train_step += 1

def test_one_epoch(model,
                   test_dataloader,
                   criterion,
                   metrics,
                   summary_writer,
                   local_rank,
                   epoch,
                   dist_num):

    logger.info('Start testing process in epoch {}.'.format(epoch + 1))
    model.eval()
    metrics.clear()
    loss_avg = AverageMeter('loss', ':.4e')

    with tqdm(test_dataloader) as pbar:
        for data_dict in pbar:
            gpu_device = torch.device('cuda:{}'.format(local_rank))
            data_dict = to_device(data_dict, gpu_device)
            with torch.no_grad():
                res = model(data_dict['rgb'], data_dict['depth'])
                depth_scale = data_dict['depth_max'] - data_dict['depth_min']
                res = res * depth_scale.reshape(-1, 1, 1) + data_dict['depth_min'].reshape(-1, 1, 1)
                data_dict['pred'] = res
                loss_dict = criterion(data_dict)
                _ = metrics.evaluate_batch(data_dict, record = True)
                torch.distributed.barrier()  # noqa
                reduced_loss = dict()
                for key in loss_dict.keys():
                    reduced_loss[key] = reduce_mean(loss_dict[key], dist_num)
                loss_avg.update(reduced_loss["loss"])
                if local_rank == 0:
                    description = 'Epoch {}, '.format(epoch + 1)
                    for key in reduced_loss.keys():
                        description = description + key + ": {:.8f}".format(reduced_loss[key].item()) + "  "
                    pbar.set_description(description)

    metrics_result = metrics.get_results()
    metrics_result_reduce = dict()
    for key in metrics_result.keys():
        metrics_result_reduce[key] = reduce_mean(metrics_result[key], dist_num)
    if local_rank == 0:
        for key in metrics_result.keys():
            summary_writer.add_scalar(key,
                                      metrics_result_reduce[key].data.cpu().numpy(),
                                      global_step=epoch)

    return loss_avg.avg, metrics_result_reduce


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--cfg', '-c',
        default=os.path.join('configs', 'default.yaml'),
        help='path to the configuration file',
        type=str
    )

    args = parser.parse_args()
    cfg_filename = args.cfg
    with open(cfg_filename, 'r') as cfg_file:
        cfg_params = yaml.load(cfg_file, Loader = yaml.FullLoader)

    mp.spawn(dist_trainer, nprocs=cfg_params['dist_num'],
             args=(cfg_params['dist_num'], cfg_params))





