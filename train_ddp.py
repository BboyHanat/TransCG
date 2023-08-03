import os
import yaml
import torch
import random
import logging
import argparse
import warnings
import time
import numpy as np
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from utils.logger import ColoredLogger
from utils.builder import ConfigBuilder
from utils.constants import LOSS_INF
from utils.functions import to_device
from time import perf_counter
import torch.multiprocessing as mp
from torch.nn import SyncBatchNorm, parallel
from utils.dist_util import reduce_mean, AverageMeter


# torch.multiprocessing.set_sharing_strategy('file_system')
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
    init_seeds(local_rank + 1, cuda_deterministic=False)
    init_method = 'tcp://' + config['dist']['ip'] + ':' + str(config['dist']['port'])
    dist.init_process_group(backend='nccl',  # noqa
                            init_method=init_method,
                            world_size=dist_num,
                            rank=local_rank)
    torch.cuda.set_device(local_rank)

    builder = ConfigBuilder(**config)
    logger.info('Building models ...')
    network_model = builder.get_model()

    network_model.to(local_rank)
    # convert model to ddp
    network_model = parallel.DistributedDataParallel(network_model,
                                                     device_ids=[local_rank])
    network_model = SyncBatchNorm.convert_sync_batchnorm(network_model)

    if local_rank == 0:
        logger.info('Checking checkpoints ...')
    start_epoch = 0
    checkpoint_loss = 10000000000.0
    max_epoch = builder.get_max_epoch()
    stats_dir = builder.get_stats_dir()
    checkpoint_file = os.path.join(stats_dir, 'checkpoint.tar')

    if os.path.isfile(checkpoint_file):
        gpu_device = torch.device('cuda:{}'.format(local_rank))
        checkpoint = torch.load(checkpoint_file, map_location=gpu_device)
        network_model.module.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        checkpoint_metrics = checkpoint['metrics']
        checkpoint_loss = checkpoint['loss']
        logger.info("Checkpoint {} (epoch {}) loaded.".format(checkpoint_file, start_epoch))
    if local_rank == 0:
        logger.info('Building optimizer and learning rate schedulers ...')
    resume = (start_epoch > 0)
    optimizer = builder.get_optimizer(network_model, resume=resume, resume_lr=builder.get_resume_lr())

    if local_rank == 0:
        logger.info('Building ddp dataloaders ...')

    time.sleep(3)
    train_dataloader, len_of_train, train_batch_size = builder.get_dataloader_ddp(split='train')
    test_dataloader, len_of_val, test_batch_size = builder.get_dataloader_ddp(split='test')

    if local_rank == 0:
        logger.info('Building optimizer and learning rate schedulers ...')
    resume = (start_epoch > 0)

    lr_scheduler = builder.get_lr_scheduler(optimizer, resume=resume,
                                            resume_epoch=(start_epoch - 1 if resume else None))
    criterion = builder.get_criterion()
    metrics = builder.get_metrics()
    summary_writer = builder.get_summary_writer()

    # start training
    if start_epoch != 0:
        min_loss = checkpoint_loss
        min_loss_epoch = start_epoch
    else:
        min_loss = LOSS_INF
        min_loss_epoch = None

    train_steps = int(len_of_train / train_batch_size / dist_num)
    if local_rank == 0:
        logger.info('One epoch train steps: {}'.format(train_steps))
    for epoch in range(start_epoch, max_epoch):
        if local_rank == 0:
            logger.info('--> Epoch {}/{}'.format(epoch + 1, max_epoch))
        # train_one_epoch(network_model, optimizer,
        #                 criterion, train_dataloader,
        #                 lr_scheduler, summary_writer,
        #                 local_rank, train_steps,
        #                 epoch, dist_num)
        loss, metrics_result = test_one_epoch(network_model, test_dataloader,
                                              criterion, metrics,
                                              summary_writer,  local_rank,
                                              epoch, test_batch_size, dist_num)

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
    for ts, data_dict in enumerate(train_dataloader):
        optimizer.zero_grad()
        gpu_device = torch.device('cuda:{}'.format(local_rank))
        data_dict = to_device(data_dict, gpu_device)
        res = network_model(data_dict['rgb'], data_dict['depth'])
        depth_scale = data_dict['depth_max'] - data_dict['depth_min']
        res = res * depth_scale.reshape(-1, 1, 1) + data_dict['depth_min'].reshape(-1, 1, 1)
        data_dict['pred'] = res
        loss_dict = criterion(data_dict)
        torch.distributed.barrier()  # noqa
        loss = loss_dict['loss']
        loss.backward()
        optimizer.step()

        reduced_loss = dict()
        for key in loss_dict.keys():
            reduced_loss[key] = reduce_mean(loss_dict[key], dist_num)
        if local_rank == 0:
            description = 'Epoch {} Step {}/{}, '.format(epoch + 1, ts, train_steps)
            for key in reduced_loss.keys():
                description = description + key + ": {:.8f}".format(reduced_loss[key].item()) + "  "
            logger.info(description)
            for key in loss_dict.keys():
                summary_writer.add_scalar("train/"+key,
                                          reduced_loss[key].data.cpu().numpy(),
                                          global_step=epoch * train_steps + ts)

def test_one_epoch(model,
                   test_dataloader,
                   criterion,
                   metrics,
                   summary_writer,
                   local_rank,
                   epoch,
                   batch_size,
                   dist_num):

    logger.info('Start testing process in epoch {}.'.format(epoch + 1))
    model.eval()
    metrics.clear()
    loss_avg = AverageMeter('loss', ':.4e')
    gpu_device = torch.device('cuda:{}'.format(local_rank))
    with torch.no_grad():
        for vs, data_dict in enumerate(test_dataloader):
            if vs >10:
                break
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
                loss_avg.update(reduced_loss["loss"], batch_size)
                if local_rank == 0:
                    description = 'Epoch {}, Step {} '.format(epoch + 1, vs)
                    for key in reduced_loss.keys():
                        description = description + key + ": {:.8f}".format(reduced_loss[key].item()) + "  "
                    logger.info(description)

    metrics_result = metrics.get_results()
    metrics_result_reduce = dict()
    for key in metrics_result.keys():
        if key == "samples":
            continue
        metrics_result_reduce[key] = reduce_mean(metrics_result[key], dist_num)
    if local_rank == 0:
        for key in metrics_result.keys():
            if key == "samples":
                continue
            summary_writer.add_scalar(key,
                                      metrics_result_reduce[key].item(),
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

    mp.spawn(dist_trainer, nprocs=cfg_params['dist']['dist_num'],
             args=(cfg_params["dist"]['dist_num'], cfg_params))


if __name__ == "__main__":
    main()




