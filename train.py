##!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import shutil
import argparse
import yaml
import time
import torch
import numpy as np
import torch.nn as nn
import torch.utils.data
from tt.model import Transducer
from tt.optim import Optimizer
from tt.dataset import AudioDataset
from warprnnt_pytorch import RNNTLoss
from tensorboardX import SummaryWriter
from tt.utils import AttrDict, init_logger, count_parameters, save_model, computer_cer, dict_map, write_result


def train(epoch, config, model, training_data, optimizer, criterion, logger, visualizer=None):
    model.train()
    start_epoch = time.process_time()
    total_loss = 0
    optimizer.epoch()
    batch_steps = len(training_data)

    for step, (inputs, inputs_length, targets, targets_length) in enumerate(training_data):

        start = time.process_time()

        max_inputs_length = inputs_length.max()
        max_targets_length = targets_length.max()
        inputs = inputs[:, :max_inputs_length, :]
        targets = targets[:, :max_targets_length]

        if config.training.num_gpu > 0:
            inputs, inputs_length = inputs.cuda(), inputs_length.cuda()
            targets, targets_length = targets.cuda(), targets_length.cuda()

        if config.optim.step_wise_update:
            optimizer.step_decay_lr()

        optimizer.zero_grad()

        logits = model(inputs, inputs_length, targets, targets_length)

        loss = criterion(logits, targets.int(), inputs_length.int(), targets_length.int())

        if config.training.num_gpu > 1:
            loss = torch.mean(loss)

        loss.backward()

        # total_loss += loss.item()
        total_loss += float(loss)

        grad_norm = nn.utils.clip_grad_norm_(
            model.parameters(), config.training.max_grad_norm)

        optimizer.step()

        if visualizer is not None:
            visualizer.add_scalar(
                'train_loss', loss.item(), optimizer.global_step)
            visualizer.add_scalar(
                'learn_rate', optimizer.lr, optimizer.global_step)
            visualizer.add_histogram('inputs', inputs, optimizer.global_step)
            visualizer.add_histogram('inputs_length', inputs_length, optimizer.global_step)
            visualizer.add_histogram('targets', targets, optimizer.global_step)
            visualizer.add_histogram('targets_length', targets_length, optimizer.global_step)

        avg_loss = total_loss / (step + 1)
        if optimizer.global_step % config.training.show_interval == 0:
            end = time.process_time()
            process = step / batch_steps * 100
            logger.info('-Training-Epoch:%d(%.5f%%), Global Step:%d, Learning Rate:%.6f, Grad Norm:%.5f, Loss:%.5f, '
                        'AverageLoss: %.5f, Run Time:%.3f' % (epoch, process, optimizer.global_step, optimizer.lr,
                                                              grad_norm, loss.item(), avg_loss, end - start))

        del loss

        # break
    end_epoch = time.process_time()
    logger.info('-Training-Epoch:%d, Average Loss: %.5f, Epoch Time: %.3f' %
                (epoch, total_loss / (step + 1), end_epoch - start_epoch))


def eval(epoch, config, model, validating_data, logger, visualizer=None, vocab=None):
    model.eval()
    total_loss = 0
    total_dist = 0
    total_word = 0
    batch_steps = len(validating_data)

    for step, (inputs, inputs_length, targets, targets_length) in enumerate(validating_data):

        max_inputs_length = inputs_length.max()
        max_targets_length = targets_length.max()
        inputs = inputs[:, :max_inputs_length, :]
        targets = targets[:, :max_targets_length]

        if config.training.num_gpu > 0:
            inputs, inputs_length = inputs.cuda(), inputs_length.cuda()
            targets, targets_length = targets.cuda(), targets_length.cuda()

        preds = model.recognize2(inputs, inputs_length)

        transcripts = [targets.cpu().numpy()[i][:targets_length[i].item()]
                       for i in range(targets.size(0))]

        if vocab is not None:
            preds = dict_map(preds, vocab)
            transcripts = dict_map(transcripts, vocab)

        dist, num_words = computer_cer(preds, transcripts)
        total_dist += dist
        total_word += num_words

        cer = total_dist / total_word * 100
        if step % config.training.show_interval == 0:
            process = step / batch_steps * 100
            logger.info('-Validation-Epoch:%d(%.5f%%), CER: %.5f %%' % (epoch, process, cer))
            write_result(preds, transcripts)

    val_loss = total_loss / (step + 1)
    logger.info('-Validation-Epoch:%4d, AverageLoss:%.5f , AverageCER: %.5f %%' %
                (epoch, val_loss, cer))

    if visualizer is not None:
        visualizer.add_scalar('cer', cer, epoch)

    return cer


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type=str, default='config/myjoint.yaml')
    parser.add_argument('-log', type=str, default='train.log')
    parser.add_argument('-mode', type=str, default='retrain')
    opt = parser.parse_args()

    configfile = open(opt.config)
    config = AttrDict(yaml.load(configfile, Loader=yaml.FullLoader))

    exp_name = os.path.join('egs', config.data.name, 'exp', config.training.save_model)
    if not os.path.isdir(exp_name):
        os.makedirs(exp_name)
    logger = init_logger(os.path.join(exp_name, opt.log))

    shutil.copyfile(opt.config, os.path.join(exp_name, 'config.yaml'))
    logger.info('Save config info.')

    # num_workers = config.training.num_gpu * config.data.batch_size
    # num_workers = config.data.batch_size
    train_dataset = AudioDataset(config.data, 'train')
    training_data = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.data.batch_size,
        # train_dataset, batch_size=config.data.batch_size * config.training.num_gpu,
        shuffle=config.data.shuffle, num_workers=0)
    logger.info('Load Train Set!')

    dev_dataset = AudioDataset(config.data, 'dev')
    validate_data = torch.utils.data.DataLoader(
        dev_dataset, batch_size=config.data.batch_size,
        # dev_dataset, batch_size=config.data.batch_size * config.training.num_gpu,
        shuffle=False, num_workers=0)
    logger.info('Load Dev Set!')

    vocab = {}
    with open(config.data.vocab, "r") as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            index = int(parts[1])
            vocab[index] = word
    logger.info('Load Vocabulary!')

    if config.training.num_gpu > 0:
        torch.cuda.manual_seed(config.training.seed)
        torch.backends.cudnn.deterministic = True
    else:
        torch.manual_seed(config.training.seed)
    logger.info('Set random seed: %d' % config.training.seed)

    model = Transducer(config.model)

    if config.training.load_model:
        checkpoint = torch.load(config.training.load_model)
        model.encoder.load_state_dict(checkpoint['encoder'])
        model.decoder.load_state_dict(checkpoint['decoder'])
        model.joint.load_state_dict(checkpoint['joint'])
        logger.info('Loaded model from %s' % config.training.load_model)
    elif config.training.load_encoder or config.training.load_decoder:
        if config.training.load_encoder:
            checkpoint = torch.load(config.training.load_encoder)
            model.encoder.load_state_dict(checkpoint['encoder'])
            logger.info('Loaded encoder from %s' %
                        config.training.load_encoder)
        if config.training.load_decoder:
            checkpoint = torch.load(config.training.load_decoder)
            model.decoder.load_state_dict(checkpoint['decoder'])
            logger.info('Loaded decoder from %s' %
                        config.training.load_decoder)

    if config.training.num_gpu > 0:
        model = model.cuda()
        if config.training.num_gpu > 1:
            device_ids = list(range(config.training.num_gpu))
            model = torch.nn.DataParallel(model, device_ids=device_ids)
        logger.info('Loaded the model to %d GPUs' % config.training.num_gpu)

    n_params, enc, dec = count_parameters(model)
    logger.info('# the number of parameters in the whole model: %d' % n_params)
    logger.info('# the number of parameters in the Encoder: %d' % enc)
    logger.info('# the number of parameters in the Decoder: %d' % dec)
    logger.info('# the number of parameters in the JointNet: %d' %
                (n_params - dec - enc))

    optimizer = Optimizer(model.parameters(), config.optim)
    logger.info('Created a %s optimizer.' % config.optim.type)

    criterion = RNNTLoss()
    logger.info('Created a RNNT loss.')

    if opt.mode == 'continue':
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        logger.info('Load Optimizer State!')
    else:
        start_epoch = 0

    # create a visualizer
    if config.training.visualization:
        visualizer = SummaryWriter(os.path.join(exp_name, 'log'))
        logger.info('Created a visualizer.')
    else:
        visualizer = None

    for epoch in range(start_epoch, config.training.epochs):

        train(epoch, config, model, training_data,
              optimizer, criterion, logger, visualizer)

        save_name = os.path.join(exp_name, '%s.epoch%d.chkpt' % (config.training.save_model, epoch))
        save_model(model, optimizer, config, save_name)
        logger.info('Epoch %d model has been saved.' % epoch)

        if config.training.eval_or_not:
            _ = eval(epoch, config, model, validate_data, logger, visualizer, vocab)

        if epoch >= config.optim.begin_to_adjust_lr:
            optimizer.decay_lr()
            # early stop
            if optimizer.lr < 1e-6:
                logger.info('The learning rate is too low to train.')
                break
            logger.info('Epoch %d update learning rate: %.6f' %
                        (epoch, optimizer.lr))

    logger.info('The training process is OVER!')


if __name__ == '__main__':
    main()
