"""
Stage 1: VQ training with Autoencoder

Run `python stage1.py`
"""
import os
from argparse import ArgumentParser
import torch
import wandb
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from preprocessing.supernova_dataset import SupernovaDataset  
from experiments.exp_stage01 import ExpStage1  
from utils import get_root_dir, load_yaml_param_settings, str2bool


def load_args():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, help="Path to the config data file.",
                        default=get_root_dir().joinpath('configs', 'config.yaml'))
    parser.add_argument('--dataset_path', type=str, help="Path to the supernova dataset CSV.", required=True)
    parser.add_argument('--gpu_device_ind', nargs='+', default=[0], type=int, help='Indices of GPU devices to use.')
    return parser.parse_args()


def train_stage1(config: dict,
                 dataset_path: str,
                 train_data_loader: DataLoader,
                 test_data_loader: DataLoader,
                 gpu_device_ind):
    """
    Train TimeVQVAE Stage 1 using an Autoencoder instead of STFT.
    """
    project_name = 'TimeVQVAE-stage1'


    sample_data = next(iter(train_data_loader)) 
    print(f"Tipo de sample_data: {type(sample_data)}")  # Debugging
    print(f"Shape de sample_data: {sample_data.shape}")  # Debugging

    _, input_length, _ = sample_data.shape  
    in_channels = 1  

    train_exp = ExpStage1(in_channels, input_length, config) 
    
    n_trainable_params = sum(p.numel() for p in train_exp.parameters() if p.requires_grad)
    wandb_logger = WandbLogger(project=project_name, name=None, config={**config, 'dataset_path': dataset_path, 'n_trainable_params': n_trainable_params})

    # Check if GPU is available
    if not torch.cuda.is_available():
        print('GPU is not available. Using CPU...')
        device = 1
        accelerator = 'cpu'
    else:
        accelerator = 'gpu'
        device = gpu_device_ind
        
    trainer = pl.Trainer(logger=wandb_logger,
                         enable_checkpointing=False,
                         callbacks=[LearningRateMonitor(logging_interval='step')],
                         max_steps=config['trainer_params']['max_steps']['stage1'],
                         devices=device,
                         accelerator=accelerator,
                         val_check_interval=config['trainer_params']['val_check_interval']['stage1'],
                         accumulate_grad_batches=1,
                         )
    trainer.fit(train_exp, train_dataloaders=train_data_loader, val_dataloaders=test_data_loader)

    # Test
    print('Closing...')
    wandb.finish()

    # Save model checkpoint
    print('Saving the model...')
    save_path = get_root_dir().joinpath('saved_models')
    os.makedirs(save_path, exist_ok=True)
    trainer.save_checkpoint(os.path.join(save_path, f'stage1-{os.path.basename(dataset_path)}.ckpt'))


if __name__ == '__main__':
    # Load config
    args = load_args()
    config = load_yaml_param_settings(args.config)

    # Load dataset
    batch_size = config['dataset']['batch_sizes']['stage1']
    dataset = SupernovaDataset(args.dataset_path)  # carga datos

    #Split
    train_size = int(0.8 * len(dataset))  # 80% train, 20% test
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    #DataLoaders
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Train Stage 1
    train_stage1(config, args.dataset_path, train_data_loader, test_data_loader, args.gpu_device_ind)
