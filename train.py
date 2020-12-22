import fire
import os
from tqdm import tqdm
import torch
from torch import optim
import numpy as np
import torchvision
import torch.distributed as dist
from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils import parse_config_or_kwargs, store_yaml

from utils import check_dir, get_logger_2
from utils import set_seed, dist_init, getoneNode
from dataset import ImageFolder
from torch.multiprocessing import Process
from torch.nn.parallel import DistributedDataParallel as DDP
from network import U_Net, R2U_Net, AttU_Net, R2AttU_Net
from evaluation import *
from skimage.metrics import adapted_rand_error, variation_of_information


def save_checkpoint(state_dict, save_path):
    torch.save(state_dict, save_path, _use_new_zipfile_serialization=False)


def get_metric_val(label_tensor, predict_tensor, thres=0.5):
    '''
    label_tensor: batchsize x 1 x 512 x 512
    predict_tensor: batchsize x 1 x 512 x 512
    '''
    label_array = np.squeeze(label_tensor.data.cpu().numpy(), 1).astype(np.int)
    predict_array = np.squeeze(predict_tensor.data.cpu().numpy(), 1)
    predict_array = (predict_array > thres).astype(np.int)

    random_error_error = 0.0
    random_error_precision = 0.0
    random_error_recall = 0.0

    false_split = 0.0
    false_merge = 0.0

    for i in range(label_array.shape[0]):
        error, precision, recall = adapted_rand_error(label_array[i], predict_array[i])
        splits, merges = variation_of_information(label_array[i], predict_array[i])
        random_error_error += error
        random_error_precision += precision
        random_error_recall += recall
        false_split += splits
        false_merge += merges

    return random_error_error, random_error_precision, random_error_recall, false_split, false_merge


def test(model, test_loader, conf, logger, epoch, best_random_error):
    model.eval()
    acc = 0.  # Accuracy
    random_error_avg = 0.0
    random_precision_avg = 0.0
    random_recall_avg = 0.0
    false_split_avg = 0.0
    false_merge_avg = 0.0
    length = 0

    # here we store the 5 test images in the same big image
    result_store_dir = os.path.join(conf['exp_dir'], 'result')
    if conf['rank'] == 0:
        check_dir(result_store_dir)
    store_path_fmt = os.path.join(result_store_dir, 'epoch-{}-{}.png')

    # here we store each predicted image in a .png
    result_single_image_dir = os.path.join(conf['exp_dir'], 'result_single', 'epoch-{}'.format(epoch))
    dist.barrier()

    with torch.no_grad():
        for iter_idx, (images, labels, _) in enumerate(test_loader):
            images = images.to(conf['device'])
            labels = labels.to(conf['device'])
            seg_res = model(images)
            seg_prob = torch.sigmoid(seg_res)

            acc += get_accuracy(seg_prob, labels)
            random_error, random_precision, random_recall, false_split, false_merge = get_metric_val(labels, seg_prob)
            random_error_avg += random_error
            random_precision_avg += random_precision
            random_recall_avg += random_recall
            false_split_avg += false_split
            false_merge_avg += false_merge
            length += images.size(0)

            if epoch % conf['save_per_epoch'] == 0 and conf['rank'] == 0:

                torchvision.utils.save_image(images.data.cpu() + 0.5, store_path_fmt.format(epoch, 'image'))
                torchvision.utils.save_image(labels.data.cpu(), store_path_fmt.format(epoch, 'GT'))
                torchvision.utils.save_image(seg_prob.data.cpu(), store_path_fmt.format(epoch, 'SR'))
                torchvision.utils.save_image((seg_prob > 0.5).float().data.cpu(), store_path_fmt.format(epoch, 'PRE'))

                check_dir(result_single_image_dir)
                for i in range(seg_prob.shape[0]):
                    store_path = os.path.join(result_single_image_dir, '{}.png'.format(i))
                    torchvision.utils.save_image((seg_prob > 0.5).float()[i].data.cpu(), store_path)
                    store_path = os.path.join(result_single_image_dir, '{}-prob.png'.format(i))
                    torchvision.utils.save_image(seg_prob[i].data.cpu(), store_path)


    acc = acc / length
    random_error_avg /= length
    random_precision_avg /= length
    random_recall_avg /= length
    false_split_avg /= length
    false_merge_avg /= length

    if random_error_avg < best_random_error and conf['rank'] == 0:
        torchvision.utils.save_image(images.data.cpu() + 0.5, store_path_fmt.format('Best', 'image'))
        torchvision.utils.save_image(labels.data.cpu(), store_path_fmt.format('Best', 'GT'))
        torchvision.utils.save_image(seg_prob.data.cpu(), store_path_fmt.format('Best', 'SR'))
        torchvision.utils.save_image((seg_prob > 0.5).float().data.cpu(), store_path_fmt.format('Best', 'PRE'))
        result_single_image_dir = os.path.join(conf['exp_dir'], 'result_single', 'Best'.format(epoch))
        check_dir(result_single_image_dir)
        for i in range(seg_prob.shape[0]):
            store_path = os.path.join(result_single_image_dir, '{}.png'.format(i))
            torchvision.utils.save_image((seg_prob > 0.5).float()[i].data.cpu(), store_path)
            store_path = os.path.join(result_single_image_dir, '{}-prob.png'.format(i))
            torchvision.utils.save_image(seg_prob[i].data.cpu(), store_path)

    # if conf['rank'] == 0:
    #     logger.info("[Test] Rank: {} Epoch: [{}/{}] Acc: {:.3f}".format(conf['rank'],
    #                                                         epoch, conf['num_epochs'],
    #                                                         acc))
    if conf['rank'] == 0:
        logger.info("[Test] Rank: {} Epoch: [{}/{}] Acc: {:.3f} R_error: {:.3f} R_pre: {:.3f} R_recall: {:.3f}"
                    " F_split: {:.2f} F_merge: {:.2f}".format(conf['rank'], epoch, conf['num_epochs'],
                                                              acc, random_error_avg, random_precision_avg,
                                                              random_recall_avg, false_split_avg, false_merge_avg
                                                              ))


    return acc, random_error_avg


def train(model, train_loader, test_loader, optimizer, conf, logger):
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=conf['num_epochs'] * len(train_loader),
                                                eta_min=1e-6)

    model.train()
    best_random_error = 100.0

    for epoch in range(conf['num_epochs']):
        acc = 0.  # Accuracy
        random_error_avg = 0.0
        random_precision_avg = 0.0
        random_recall_avg = 0.0
        false_split_avg = 0.0
        false_merge_avg = 0.0
        length = 0
        epoch_loss = 0.0

        model.train()

        if conf['rank'] == 0:
            t_bar = tqdm(ncols=100, total=len(train_loader), desc='Epoch:{}'.format(epoch))
        for iter_idx, (images, labels, loss_weight) in enumerate(train_loader):
            if conf['rank'] == 0:
                t_bar.update()

            images = images.to(conf['device'])
            labels = labels.to(conf['device'])
            loss_weight = loss_weight.to(conf['device'])

            optimizer.zero_grad()
            seg_res = model(images)
            seg_prob = torch.sigmoid(seg_res)

            seg_res_flat = seg_res.view(seg_res.size(0), -1)
            labels_flat = labels.view(labels.size(0), -1)
            loss_weight_flat = loss_weight.view(loss_weight.size(0), -1)

            loss = F.binary_cross_entropy_with_logits(seg_res_flat, labels_flat, loss_weight_flat)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()

            acc += get_accuracy(seg_prob, labels)
            random_error, random_precision, random_recall, false_split, false_merge = get_metric_val(labels, seg_prob)
            random_error_avg += random_error
            random_precision_avg += random_precision
            random_recall_avg += random_recall
            false_split_avg += false_split
            false_merge_avg += false_merge
            length += images.size(0)

        if conf['rank'] == 0:
            t_bar.close()

        acc = acc / length
        random_error_avg /= length
        random_precision_avg /= length
        random_recall_avg /= length
        false_split_avg /= length
        false_merge_avg /= length
        epoch_loss /= len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']


        # logger.info("[Train] Rank: {} Epoch: [{}/{}] Acc: {:.3f} Loss: {:.3f} Lr:{:.3e}".format(conf['rank'],
        #                                                 epoch, conf['num_epochs'],
        #                                                 acc, epoch_loss, current_lr))

        logger.info("[Train] Rank: {} Epoch: [{}/{}] Acc: {:.3f} Loss: {:.3f} Lr:{:.3e} "
                    "R_error: {:.3f} R_pre: {:.3f} R_recall: {:.3f}"
                    " F_split: {:.2f} F_merge: {:.2f}".format(conf['rank'], epoch, conf['num_epochs'],
                                                              acc, epoch_loss, current_lr,
                                                              random_error_avg, random_precision_avg,
                                                              random_recall_avg, false_split_avg, false_merge_avg
                                                              ))

        test_acc, test_random_error = test(model, test_loader, conf, logger, epoch, best_random_error)

        if best_random_error > test_random_error and conf['rank'] == 0:
            best_random_error = test_random_error
            save_name = 'Best'
            state_dict = {'model': model.module.state_dict()}
            save_checkpoint(state_dict, conf['checkpoint_format'].format(save_name))

        if epoch % conf['save_per_epoch'] == 0 and conf['rank'] == 0:
            save_name = 'Epoch-{}'.format(epoch)
            state_dict = {'model': model.module.state_dict()}
            save_checkpoint(state_dict, conf['checkpoint_format'].format(save_name))


def main(config, rank, world_size, gpu_id, port, kwargs):
    torch.backends.cudnn.benchmark = True

    conf = parse_config_or_kwargs(config, **kwargs)

    # --------- multi machine train set up --------------
    if conf['train_local'] == 1:
        host_addr = 'localhost'
        conf['rank'] = rank
        conf['local_rank'] = gpu_id  # specify the local gpu id
        conf['world_size'] = world_size
        dist_init(host_addr, conf['rank'], conf['local_rank'], conf['world_size'], port)
    else:
        host_addr = getoneNode()
        conf['rank'] = int(os.environ['SLURM_PROCID'])
        conf['local_rank'] = int(os.environ['SLURM_LOCALID'])
        conf['world_size'] = int(os.environ['SLURM_NTASKS'])
        dist_init(host_addr, conf['rank'], conf['local_rank'],
                  conf['world_size'], '2' + os.environ['SLURM_JOBID'][-4:])
        gpu_id = conf['local_rank']
    # --------- multi machine train set up --------------

    # setup logger
    if conf['rank'] == 0:
        check_dir(conf['exp_dir'])
        logger = get_logger_2(os.path.join(conf['exp_dir'], 'train.log')
                              , "[ %(asctime)s ] %(message)s")
    dist.barrier()  # let the rank 0 mkdir first
    if conf['rank'] != 0:
        logger = get_logger_2(os.path.join(conf['exp_dir'], 'train.log')
                              , "[ %(asctime)s ] %(message)s")

    logger.info("Rank: {}/{}, local rank:{} is running".format(conf['rank'], conf['world_size'],
                                                               conf['rank']))

    # write the config file to the exp_dir
    if conf['rank'] == 0:
        store_path = os.path.join(conf['exp_dir'], 'config.yaml')
        store_yaml(config, store_path, **kwargs)

    cuda_id = 'cuda:' + str(gpu_id)
    conf['device'] = torch.device(cuda_id if torch.cuda.is_available() else 'cpu')

    model_dir = os.path.join(conf['exp_dir'], 'models')
    if conf['rank'] == 0:
        check_dir(model_dir)
    conf['checkpoint_format'] = os.path.join(model_dir, '{}.th')

    set_seed(666 + conf['rank'])

    model = eval(conf['model_type'])(base_ch_num=conf['base_ch_num'])
    model = model.to(conf['device'])
    model = DDP(model, device_ids=[conf['local_rank']], output_device=conf['local_rank'])
    optimizer = optim.Adam(model.parameters(), lr=conf['lr'], betas=(0.5, 0.99))

    if conf['rank'] == 0:
        num_params = sum(param.numel() for param in model.parameters())
        logger.info("Model type: {} Base channel num:{}".format(conf['model_type'], conf['base_ch_num']))
        logger.info("Number of parameters: {:.4f}M".format(1.0 * num_params / 1e6))
        logger.info(optimizer)

    train_set = ImageFolder(root=conf['root'], mode='train', augmentation_prob=conf['aug_prob'],
                            crop_size_min=conf['crop_size_min'], crop_size_max=conf['crop_size_max'],
                            data_num=conf['data_num'], gauss_size=conf['gauss_size'])
    train_loader = DataLoader(dataset=train_set, batch_size=conf['batch_size'],
                              shuffle=conf['shuffle'], num_workers=conf['num_workers'])

    test_set = ImageFolder(root=conf['root'], mode='test')
    test_loader = DataLoader(dataset=test_set, batch_size=5,
                             shuffle=False, num_workers=1)

    dist.barrier()  # synchronize here
    train(model, train_loader, test_loader, optimizer, conf, logger)


def spawn_process(config, gpu_id=None, port=23456, **kwargs):
    processes = []

    if gpu_id is None:
        gpu_id = [0]
    try:
        for rank, gpu_id_val in enumerate(gpu_id):
            p = Process(target=main, args=(config, rank, len(gpu_id), gpu_id_val, port, kwargs))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

    except KeyboardInterrupt:
        for p in processes:
            p.terminate()


if __name__ == '__main__':
    fire.Fire(spawn_process)
