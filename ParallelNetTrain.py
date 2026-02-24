import yaml
import torch
import utils
import random
import models
import losses
import datasets
import argparse
import numpy as np
import train_funcs
import copy

from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

#Set gpu visibility, for debbug purposes
import os

# Enable anomaly detection in PyTorch autograd. Anomaly detection helps in finding operations that
# are not supported by autograd and can be useful for debugging. It is often used during development
# and debugging phases.
torch.autograd.set_detect_anomaly(True)

# Set the per-process GPU memory fraction to 90%. This means that the GPU will allocate a maximum
# of 90% of its available memory for this process. This can be useful to limit the GPU memory usage
# when running multiple processes or to prevent running out of GPU memory.
# torch.cuda.set_per_process_memory_fraction(0.9, 0)

# Clear the GPU memory cache. This releases GPU memory that is no longer in use and can help free up
# memory for other operations. It's particularly useful when working with limited GPU memory.
torch.cuda.empty_cache()


def _normalize_state_dict_keys(state_dict):
    if not isinstance(state_dict, dict) or not state_dict:
        return state_dict
    if all(k.startswith('module.') for k in state_dict.keys()):
        return {k[len('module.'):]: v for k, v in state_dict.items()}
    return state_dict


def _extract_model_state_dict(ckpt):
    if isinstance(ckpt, dict) and isinstance(ckpt.get('model'), dict) and 'sd' in ckpt['model']:
        return ckpt['model']['sd']
    if isinstance(ckpt, dict) and isinstance(ckpt.get('ocr_model'), dict) and 'sd' in ckpt['ocr_model']:
        return ckpt['ocr_model']['sd']
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        return ckpt['state_dict']
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        return ckpt['model_state_dict']
    if isinstance(ckpt, dict):
        return ckpt
    raise ValueError('Unsupported checkpoint format for model weights.')


def _load_pretrained_weights(model, ckpt_path, tag):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = _normalize_state_dict_keys(_extract_model_state_dict(ckpt))
    if not isinstance(state_dict, dict):
        raise ValueError(f"[{tag}] invalid state_dict format in checkpoint: {ckpt_path}")
    target_state = model.state_dict()
    matched = sum(1 for k, v in state_dict.items() if k in target_state and target_state[k].shape == v.shape)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(
        f"[INFO] {tag}: loaded checkpoint={ckpt_path} "
        f"matched={matched} missing={len(missing)} unexpected={len(unexpected)}"
    )
    if missing:
        print(f"[WARN] {tag}: missing keys when loading pretrained weights: {len(missing)}")
    if unexpected:
        print(f"[WARN] {tag}: unexpected keys when loading pretrained weights: {len(unexpected)}")
    if tag == 'ocr' and matched == 0:
        raise RuntimeError(
            f"[ERROR] {tag}: no matching parameters loaded from checkpoint: {ckpt_path}. "
            "Please verify checkpoint format/path."
        )

def make_dataloader(spec, tag=''):
    # Create the dataset based on the provided specification
    dataset = datasets.make(spec['dataset'])
    # Create a dataset wrapper based on the provided specification and the previously created dataset
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    
    loader = DataLoader(
        dataset,
        batch_size=spec['batch'],
        shuffle=(tag == 'train'), # Shuffle the data if the tag is 'train'
        num_workers=spec.get('num_workers', 0), # Number of worker processes for data loading (0 means data is loaded in the main process)
        pin_memory=torch.cuda.is_available(), # Whether to use pinned memory for faster host->device transfer on CUDA
        collate_fn=dataset.collate_fn # A function used to collate (combine) individual samples into batches
    )

    # Return the DataLoader        
    return loader 

def make_dataloaders():
    # Create data loaders for the training and validation datasets
    # These data loaders are typically created using custom functions (e.g., make_dataloader)
    train_loader = make_dataloader(config['train_dataset'], tag='train')
    val_loader = make_dataloader(config['val_dataset'], tag='val')

    # Return the created data loaders
    return train_loader, val_loader
    
def prepare_training():
    ocr_train = bool(config.get('MODEL_OCR', {}).get('OCR_TRAIN', True))
    # Check if a training checkpoint is specified in the configuration (resuming training)
    if config.get('resume') is not None:
        # Load the saved checkpoint files
        if config['resume'][1]:
            ocr_model_sv_file = torch.load(config['resume'][1])
            ocr_model = models.make(ocr_model_sv_file['model'], load_model=True).cuda()
            ocr_opt = utils.make_optimizer(ocr_model.parameters(), ocr_model_sv_file['optimizer'], load_optimizer=True)
        else:
            ocr_model = models.make(config['MODEL_OCR']).cuda()
            ocr_opt = utils.make_optimizer(ocr_model.parameters(), config['optimizer_ocr'])
        sr_model_sv_file = torch.load(config['resume'][0])
        
        # Create the model using the configuration from the checkpoint and move it to the GPU
        sr_model = models.make(sr_model_sv_file['model'], load_model=True).cuda()
        
        # Create an optimizer with parameters from the checkpoint and load its state
        sr_opt = utils.make_optimizer(sr_model.parameters(), sr_model_sv_file['optimizer'], load_optimizer=True)
        
        # Create an EarlyStopping object using settings from the checkpoint
        early_stopper = utils.Early_stopping(**sr_model_sv_file['early_stopping'])
        
        # Get the starting epoch from the checkpoint and set the random number generator state
        epoch_start = sr_model_sv_file['epoch'] + 1     
        state = sr_model_sv_file['state']                
        torch.set_rng_state(state)
        
        # Print a message indicating that training is resuming
        print(f'Resuming from epoch {epoch_start}...')
        log(f'Resuming from epoch {epoch_start}...')
        
        # Check if a learning rate scheduler (ReduceLROnPlateau) is specified in the configuration
        
        # Create a learning rate scheduler using settings from the configuration
        ocr_lr_scheduler = StepLR(ocr_opt, step_size=5, gamma=0.9) if ocr_train else None
        sr_lr_scheduler = StepLR(sr_opt, step_size=5, gamma=0.9)
        
        # Set the learning rate scheduler's last_epoch to the resumed epoch
        if ocr_lr_scheduler is not None:
            ocr_lr_scheduler.last_epoch = epoch_start - 1
        sr_lr_scheduler.last_epoch = epoch_start - 1
       
    # If no checkpoint is specified, start training from scratch
    else:
        print('Training from start...')
        
        # Create the model using the configuration and move it to the GPU
        
        sr_model = models.make(config['MODEL_SR']).cuda()
        ocr_model = models.make(config['MODEL_OCR']).cuda()
        loaded_pretrained = False

        if config.get('LOAD_PRE_TRAINED_SR') is not None:
            _load_pretrained_weights(sr_model, config['LOAD_PRE_TRAINED_SR'], tag='sr')
            loaded_pretrained = True
        
        # Create optimizers using the configuration
        if config['LOAD_PRE_TRAINED_OCR'] is not None:
            _load_pretrained_weights(ocr_model, config['LOAD_PRE_TRAINED_OCR'], tag='ocr')
            loaded_pretrained = True
        ocr_opt = utils.make_optimizer(ocr_model.parameters(), config['optimizer_ocr'])
        sr_opt = utils.make_optimizer(sr_model.parameters(), config['optimizer_sr'])

        if loaded_pretrained:
            print('Loaded pretrained SR/OCR weights (fresh optimizer).')
            log('Loaded pretrained SR/OCR weights (fresh optimizer).')
        
        # Create an EarlyStopping object using settings from the configuration for the sr model
        early_stopper = utils.Early_stopping(**config['early_stopper'])

        # Set the starting epoch to 1
        epoch_start = 1
        
        # Create a learning rate scheduler using settings from the configuration
        ocr_lr_scheduler = StepLR(ocr_opt, step_size=5, gamma=0.9) if ocr_train else None
        sr_lr_scheduler = StepLR(sr_opt, step_size=5, gamma=0.9)

    if not ocr_train:
        for p in ocr_model.parameters():
            p.requires_grad = False
        ocr_model.eval()
            
    # For epochs prior to the starting epoch, step the learning rate scheduler
    for _ in range(epoch_start - 1):
        if ocr_lr_scheduler is not None:
            ocr_lr_scheduler.step()
        sr_lr_scheduler.step()
            
    # Log the number of model parameters and model structure
    log('model ocr: #params={}'.format(utils.compute_num_params(ocr_model, text=True)))
    log('model sr: #struct={}'.format(ocr_model))
    log('model ocr: trainable={}'.format(ocr_train))
    
    log('model sr: #params={}'.format(utils.compute_num_params(sr_model, text=True)))
    log('model sr: #struct={}'.format(sr_model))
    
    # Return the model, optimizer, starting epoch, learning rate scheduler, and EarlyStopping object
    return ocr_model, sr_model, ocr_opt, sr_opt, epoch_start, ocr_lr_scheduler, sr_lr_scheduler, early_stopper, ocr_train

def main(config_, save_path):
    # Declare global variables
    global config, log, writer
    config = config_
    
    # Create log and writer for logging training progress
    log, writer = utils.make_log_writer(save_path)
    
    # Create data loaders for training and validation datasets
    train_loader, val_loader = make_dataloaders()
    
    # Initialize the model, optimizer, learning rate scheduler, and early stopper
    ocr_model, sr_model, ocr_opt, sr_opt, epoch_start, ocr_lr_scheduler, sr_lr_scheduler, early_stopper, ocr_train = prepare_training()
    n_gpus = torch.cuda.device_count()
    train = train_funcs.make(config['func_train'])
    validation = train_funcs.make(config['func_val'])
    # Create the loss function for training    
    ocr_loss_fn = losses.make(config['loss_ocr'])
    sr_loss_fn = losses.make(config['loss_sr'])
        
    # Get maximum number of epochs and epoch save interval from configuration
    epoch_max = config['epoch_max']
    epoch_save = config.get('epoch_save')
    
    # Create a timer to measure training time
    timer = utils.Timer()  
    confusing_pairs = []
    # Loop over epochs for training
    for epoch in range(epoch_start, epoch_max+1):
        # Initialize timer for the current epoch
        print(f"epoch {epoch}/{epoch_max}")
        t_epoch_init = timer._get()
        
        # Prepare logging information for the current epoch
        log_info = ['epoch {}/{}'.format(epoch, epoch_max)]
        
        # Log the learning rate and add it to the writer
        lr_ocr = ocr_opt.param_groups[0]['lr'] if ocr_train else 0.0
        writer.add_scalar('lr_ocr', lr_ocr, epoch)
        log_info.append('lr_ocr:{}'.format(lr_ocr))
        writer.add_scalar('lr_sr', sr_opt.param_groups[0]['lr'], epoch)
        log_info.append('lr_sr:{}'.format(sr_opt.param_groups[0]['lr']))
        
        # Perform training for the current epoch and get the training loss
        train_loss = train(train_loader, ocr_model, sr_model, ocr_loss_fn, sr_loss_fn, ocr_opt, sr_opt, confusing_pairs, config)
        log_info.append('train: loss={:.4f}'.format(train_loss))
        writer.add_scalar('train_loss', train_loss, epoch)
        
        # Perform validation for the current epoch and get the validation loss
        val_loss, neg_acc_sr, confusing_pair = validation(val_loader, ocr_model, sr_model, ocr_loss_fn, sr_loss_fn, confusing_pairs, config)             
        log_info.append('val: loss={:.4f}'.format(val_loss))
        writer.add_scalar('val_train_loss', val_loss, epoch)
        writer.add_scalar('acc_sr', (1-neg_acc_sr)*100, epoch)
        
        # Adjust the learning rate using the learning rate scheduler if it's defined
        if ocr_lr_scheduler is not None:
            ocr_lr_scheduler.step()
        sr_lr_scheduler.step()         
        
        # Calculate and log elapsed times for the current epoch
        t = timer._get()        
        t_epoch = timer.time_text(t - t_epoch_init )
        t_elapsed = timer.time_text(t)
        log_info.append('{} / {}'.format(t_epoch, t_elapsed))
        
        # Check for early stopping and log the status
        stop, bm = early_stopper.early_stop(neg_acc_sr)
        log_info.append('Early stop {} / Best model {}'.format(stop, bm))
        
        # Get the underlying model (without DataParallel) if multiple GPUs are used
       
        def save_model(model, opt, opt_dict, name_model):
            if n_gpus > 1:
                model_ = model.module
            else:
                model_ = model
            
            # Prepare model and optimizer specifications for saving
            model_spec = copy.deepcopy(name_model)
            model_spec['sd'] = model_.state_dict()
            optimizer_spec = copy.deepcopy(opt_dict)
            optimizer_spec['sd'] = opt.state_dict() if opt is not None else {}
            early_stopper_ = vars(early_stopper)
            
            # Get the current random number generator state
            state = torch.get_rng_state()
            
            # Create a dictionary to save the model checkpoint
            sv_file = {
                'model': model_spec, 
                'optimizer': optimizer_spec, 
                'epoch': epoch, 
                'state': state, 
                'early_stopping': early_stopper_
                }
            
            # Save the model checkpoint if it's the best model so far
            if bm:
                torch.save(sv_file, save_path / Path(f"best_model_{name_model['name']}_Epoch_{epoch}.pth"))
            else:
                torch.save(sv_file, save_path / Path(f"epoch-last-{name_model['name']}.pth"))
        
            # Save the model checkpoint if it's an epoch save interval
            if (epoch_save is not None) and (epoch % epoch_save == 0):
                torch.save(sv_file, save_path / Path('epoch-{}-{}.pth'.format(epoch, name_model['name'])))
         
        save_model(ocr_model, ocr_opt, config['optimizer_ocr'], config['MODEL_OCR'])
        save_model(sr_model, sr_opt, config['optimizer_sr'], config['MODEL_SR'])
        # Log the training progress for the current epoch
        log(', '.join(log_info))
        writer.flush()
        
        # Check for early stopping and break the loop if early stopping criteria are met
        if stop:
            
            print('Early stop: {}'.format(stop))
            break


if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--save', default=None)    
    parser.add_argument('--tag', default=None)
    args = parser.parse_args()
    
    
    # Define a function to set random seeds for reproducibility
    def setup_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)  # sets the seed for cpu
        torch.cuda.manual_seed(seed)  # Sets the seed for the current GPU.
        torch.cuda.manual_seed_all(seed)  #  Sets the seed for the all GPU.
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
    
    # Set a fixed random seed (for reproducibility)
    setup_seed(1996)
    
    # Read the configuration file (usually in YAML format)
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # Determine the save name for checkpoints
    config_stem = Path(args.config).stem
    if args.save is None:
        save_name = '_' + config_stem
    else:
        save_name = str(args.save)
    if args.tag is not None:
        save_name = f'{save_name}_{args.tag}'
        
    # Create a save path for model checkpoints and ensure the directory exists
    save_path = Path('./save') / Path(save_name)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Call the main training function with the configuration and save path
    main(config, save_path)
