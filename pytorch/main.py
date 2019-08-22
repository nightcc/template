import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchnet.meter as meter

import math
import time
from collections import OrderedDict
import parser
import msglogging
from models import create_model 
import utils

msglogger = None

def main():
    script_dir = os.path.dirname(__file__)
    module_path = os.path.abspath(script_dir)
    
    args = parser.get_parser().parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    global msglogger
    msglogger = msglogging.config_logger(os.path.join(script_dir, 'logging.conf'), args.name, args.output_dir)

    start_epoch = 0
    ending_epoch = args.epochs

    if args.cpu or not torch.cuda.is_available():
        args.device = 'cpu'
        args.gpus = -1
    else:
        args.device = 'cuda'
        if args.gpus is not None:
            try:
                args.gpus = [int(s) for s in args.gpus.split(',')]
            except ValueError:
                raise ValueError('ERROR: Argument --gpus must be a comma-separated list of integers only')
            available_gpus = torch.cuda.device_count()
            for dev_id in args.gpus:
                if dev_id >= available_gpus:
                    raise ValueError('ERROR: GPU device ID {0} requested, but only {1} devices available'.format(devid, available_gpus))
            torch.cuda.set_device(args.gpus[0])


    model = create_model(args.pretrained, args.dataset, args.arch, parallel = True, device_ids = args.gpus)
    msglogger.info(model)

    optimizer = None
    best_top1 = 0
    best_epoch = 0
    if args.resumed_checkpoint_path:
        model, optimizer, start_epoch, extras = utils.load_checkpoint(model, args.resumed_checkpoint_path, None, model_device=args.device)
        if not extras is None:
            best_top1 = extras['best_top1']
            best_epoch = extras['best_epoch']
    
    if args.reset_optimizer:
        start_epoch = 0
        if optimizer is not None:
            optimizer = None
            msglogger.info('\nreset_optimizer flag set: Overriding resumed optimizer and resetting epoch count to 0')

    if optimizer is None:
        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum = args.momentum, weight_decay = args.weight_decay)
        msglogger.info('Optimzer Type: %s', type(optimizer))
        msglogger.info('Optimzer Args: %s', optimizer.defaults)

    criterion = nn.CrossEntropyLoss().to(args.device)
    
    if start_epoch >= ending_epoch:
        msglogger.error('epoch count is too low, starting epoch is {} but total epochs set to {}'.format(start_epoch, ending_epoch))
        raise ValueError('Epochs parameter is too low. Nothing to do.')

    train_loader, val_loader, test_loader, _ = utils.load_data(
        args.dataset, os.path.expanduser(args.data), args.batch_size,
        args.workers, args.validation_split, args.deterministic, 
        args.effective_train_size, args.effective_valid_size, args.effective_test_size)
    msglogger.info('Dataset sizes:\n\ttraining={}\n\tvalidation={}\n\ttest={}'.format(len(train_loader.sampler), len(val_loader.sampler), len(test_loader.sampler)))

    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)

    if args.evaluate:
        evaluate_model(model, criterion, test_loader, None, args)
        return 

    for epoch in range(start_epoch, ending_epoch):
        train(train_loader, model, criterion, optimizer, epoch, None, args=args)
        # scheduler.step()
        top1, top5, vloss = validate(val_loader, model, criterion, None, args, epoch)
       
        is_best = top1 > best_top1
        best_top1 = top1 if is_best else best_top1
        best_epoch = epoch if is_best else best_epoch
        checkpoint_extras = {'current_top1': top1,
                            'best_top1': best_top1,
                            'best_epoch': best_epoch}
        msglogger.info('==> best epoch: %d best_top1: %.3f', best_epoch, best_top1)
        utils.save_checkpoint(epoch, args.arch, model, optimizer=optimizer, extras=checkpoint_extras, is_best=is_best, name=args.name, dir=msglogger.logdir)

    test(test_loader, model, criterion, None, args)


def train(train_loader, model, criterion, optimizer, epoch, loggers, args):
    losses = OrderedDict([('Overall Loss', meter.AverageValueMeter()),
                          ('Objective Loss', meter.AverageValueMeter())])
    classerr = meter.ClassErrorMeter(accuracy=True, topk=(1,5))
    batch_time = meter.AverageValueMeter()
    data_time = meter.AverageValueMeter()

    total_samples = len(train_loader.sampler)
    batch_size = train_loader.batch_size
    steps_per_epoch = math.ceil(total_samples / batch_size)

    msglogger.info("{} samples ({} per mini-batch)".format(total_samples, batch_size))

    model.train()
    acc_stats = []
    end = time.time()
    for train_step, (inputs, target) in enumerate(train_loader):
        data_time.add(time.time() - end)

        inputs, target = inputs.to(args.device), target.to(args.device)
        output = model(inputs)
        loss = criterion(output, target)

        classerr.add(output.data, target)
        acc_stats.append([classerr.value(1), classerr.value(5)])
        losses['Objective Loss'].add(loss.item())
        losses['Overall Loss'].add(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        batch_time.add(time.time()-end)
        steps_completed = train_step + 1

        if steps_completed % args.print_freq == 0:
            errs = OrderedDict()
            errs['Top1'] = classerr.value(1)
            errs['Top5'] = classerr.value(5)

            stats_dict = OrderedDict()
            for loss_name, loss_value in losses.items():
                stats_dict[loss_name] = loss_value.mean
            stats_dict.update(errs)
            stats_dict['LR'] = optimizer.param_groups[0]['lr']
            stats_dict['Time'] = batch_time.mean
            stats = ('Performance/Training/', stats_dict)
            msglogger.info('Train epoch: %d [%5d/%5d]  Top1: %.3f  Top5: %.3f  Loss: %.3f',
                       epoch, steps_completed, steps_per_epoch,  errs['Top1'], errs['Top5'], losses['Objective Loss'].mean)

        end = time.time()
    return acc_stats




def _validate(data_loader, model, criterion, loggers, args, epoch=-1):
    losses = {'objective_loss': meter.AverageValueMeter()}
    classerr = meter.ClassErrorMeter(accuracy=True, topk=(1,5))
    print(type(meter))
    if args.earlyexit_thresholds:
       raise ValueError('Error: earlyexit function has not been completed')

    batch_time = meter.AverageValueMeter()
    total_samples = len(data_loader.sampler)
    batch_size = data_loader.batch_size
    total_steps = total_samples / batch_size
    msglogger.info("{} samples ({} per mini-batch)".format(total_samples, batch_size))

    model.eval()

    end = time.time()
    
    for validation_step, (inputs, target) in enumerate(data_loader):
        with torch.no_grad():
            inputs, target = inputs.to(args.device), target.to(args.device)
            output = model(inputs)

            loss = criterion(output, target)
            losses['objective_loss'].add(loss.item())
            classerr.add(output.data, target)
        batch_time.add(time.time()-end)
        end = time.time()
        steps_completed = validation_step + 1
        if steps_completed % args.print_freq == 0:
            if not args.earlyexit_thresholds:
                stats = ('',
                        OrderedDict([('Loss', losses['objective_loss'].mean),
                                     ('Top1', classerr.value(1)),
                                     ('Top5', classerr.value(5))]))
                msglogger.info("Validation epoch: %d [%d/%d]", epoch, validation_step, total_steps)
            else:
                pass
    
    if not args.earlyexit_thresholds:
        msglogger.info('==> Validation epoch: %d Top1: %.3f  Top5: %.3f  Loss: %.3f',
                        epoch, classerr.value()[0], classerr.value()[1], losses['objective_loss'].mean)
        return classerr.value(1), classerr.value(5), losses['objective_loss'].mean



def validate(val_loader, model, criterion, loggers, args, epoch=-1):
    msglogger.info('-----------validate (epoch={})----------'.format(epoch))
    return _validate(val_loader, model, criterion, loggers, args, epoch)
        

def test(test_loader, model, criterion, loggers, args):
    msglogger.info('----------test----------')
    top1, top5, losses = _validate(test_loader, model, criterion, loggers, args)
    return top1, top5, losses
    
def evaluate_model(model, criterion, test_loader, loggers, args):
    top1,_,_ = test(test_loader, model, criterion, loggers, args)

if __name__=="__main__":
    main()

