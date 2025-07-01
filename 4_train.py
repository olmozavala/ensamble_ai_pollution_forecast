# %%
import argparse
import collections
import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer
from utils import prepare_device
# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

# %% Read the args
args = argparse.ArgumentParser(description='PyTorch Template')
args.add_argument('-c', '--config', default='config.json', type=str,
                    help='config file path (default: None)')
args.add_argument('-r', '--resume', default=None, type=str,
                    help='path to latest checkpoint (default: None)')
args.add_argument('-d', '--device', default=None, type=str,
                    help='indices of GPUs to enable (default: all)')

# custom cli options to modify configuration from default values given in json file.
CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
options = [
    CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
    CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
]
config = ConfigParser.from_args(args, options)

# %%
logger = config.get_logger('train')

# Override test data loader's auto_regressive_steps to always be 24
config.config['test']['data_loader']['auto_regresive_steps'] = 24

# Log the batch_size, the auto_regressive_steps, the learning rate, the metrics, the loss function, the optimizer, the lr_scheduler, the model, the device, the data_loader, the valid_data_loader, the trainer
logger.info(f"Batch Size: {config['data_loader']['args']['batch_size']}")
logger.info(f"AutoRegressiveSteps: {config['trainer']['auto_regresive_steps']}")
logger.info(f"Learning Rate: {config['optimizer']['args']['lr']}")
logger.info(f"Metrics: {config['metrics']}")
logger.info(f"Loss Function: {config['loss']}")
logger.info(f"Optimizer: {config['optimizer']['type']}")
logger.info(f"LR Scheduler: {config['lr_scheduler']['type']}")

# build model architecture, then print to console
model = config.init_obj('arch', module_arch)
logger.info(model)

# prepare for (multi-device) GPU training
device, device_ids = prepare_device(config['n_gpu'])
model = model.to(device)
if len(device_ids) > 1:
    model = torch.nn.DataParallel(model, device_ids=device_ids)

# get function handles of loss and metrics
criterion = getattr(module_loss, config['loss'])
metrics = [getattr(module_metric, met) for met in config['metrics']]

# build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
trainable_params = filter(lambda p: p.requires_grad, model.parameters())
optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

# Print in which gpu is the model
print(f"Model is in GPU: {next(model.parameters()).device}")

# %%
trainer = Trainer(model, criterion, metrics, optimizer,
                    config=config,
                    device=device,
                    lr_scheduler=lr_scheduler)

trainer.train()