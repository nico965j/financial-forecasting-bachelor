import pandas as pd
import numpy as np
import json
import time
import os
import argparse
import yaml
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts, CyclicLR, OneCycleLR
from sklearn.preprocessing import RobustScaler, MinMaxScaler

from preprocessing_utils import DF2Tensors, ScaleData, FetchDataLoader
tqdm.write("Modules loaded.")

parser = argparse.ArgumentParser()
# parser.add_argument("--exp_dir", help="Full path to save best validation model")
parser.add_argument("--conf_path", default="configs/Cycl/conf_1dh.yml", help="Path of configuration file (.../conf___.yml)")


def DataLoadFromPath(path):
    start_time = time.time()
    tqdm.write("Data load...")

    # load csv
    raw_stock_data = pd.read_csv(path, index_col=0, parse_dates=True)
    stock_tickers = pd.Series(raw_stock_data.Ticker.unique())

    # change doubles/64 bit floats into 32bit floats
    float_cols = list(raw_stock_data.select_dtypes(include=['float64', 'int64']))
    raw_stock_data[float_cols] = raw_stock_data[float_cols].astype('float32')

    end_time = time.time()
    tqdm.write(f"Data load complete. Time elapsed: {end_time - start_time:.2f} seconds.")

    return raw_stock_data, stock_tickers

def AssertSeed(seed):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

def OpenConfig(path):
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        return config


##### MODEL
## Define model architecture
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_layer_size, num_layers, output_size, dropout_rate, MCD=False):
        super(LSTM, self).__init__()
        
        # Define the dimensions of the LSTM layer
        self.hidden_dim = hidden_layer_size
        self.num_layers = num_layers
        
        # Define the LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, batch_first=True, dropout=dropout_rate if MCD else 0)
        
        # dropout layer
        self.dropout = nn.Dropout(dropout_rate)

        # fc output layer
        self.fc = nn.Linear(hidden_layer_size, output_size)
        
    def forward(self, x):
        # Init hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        dropout_out = self.dropout(out)
        
        # Decode the last generated hidden state
        pred = self.fc(dropout_out[:, -1, :])

        return pred.squeeze() # single dimension output

def Unscale(databatches, target_scaler): 
    unscaled_batches = []
    for data in databatches:
        data_numpy = data.detach().cpu().numpy().reshape(-1, 1)
        unscaled_data_numpy = target_scaler.inverse_transform(data_numpy).squeeze()
        unscaled_batches.append(torch.from_numpy(unscaled_data_numpy).to(data.device))
    return unscaled_batches

## Setup trainer function
def train_model(model, train_loader, test_loader, target_scaler, criterion, optimizer, scheduler, pytorch_device, clip_value,
                num_epochs=50, n_epochs_stop=10, min_val_loss=np.Inf, epochs_no_improve=0, 
                model_save=True, save_dir='', verbose=True, early_stopping=True, MCD=False):
    
    scheduler_type = type(scheduler).__name__
    os.makedirs(save_dir, exist_ok=True)
    epoch_val_losses = []
    epoch_train_losses = []
    epoch_val_means = []
    epoch_val_stds = []

    pbar = tqdm(range(num_epochs), desc="Training", leave=True, unit='epochs')
    for epoch in pbar:
        model.train()

        train_losses = []
        pbar_inner = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False, unit='batches')
        for X_batch, y_batch in pbar_inner:
            X_batch = X_batch.to(pytorch_device)
            y_batch = y_batch.to(pytorch_device)

            # forward propagation
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            
            # backward propagation
            optimizer.zero_grad()
            
            # clipping for exploding gradients
            if clip_value > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

            # update weights
            optimizer.step()

            # unscaling and saving losses
            # unscaled_outputs, unscaled_y = Unscale([outputs, y_batch], target_scaler) #TODO: problems with unscaling, doesnt work...
            # unscaled_loss = criterion(unscaled_outputs, unscaled_y)

            train_losses.append(loss.item()) # TODO: change to unscaled_loss when working...

        # calculate average losses
        train_loss = np.mean(train_losses)
        epoch_train_losses.append(train_loss.item()) # make it a python scalar
        

        model.train() # NOT .eval() because we want to keep dropout on for MC Dropout sampling
        n_samples = 50 if MCD else 1 # activates Monte-Carlo Dropout
        val_losses = []
        val_preds = np.zeros((n_samples, len(test_loader.dataset)))
        with torch.no_grad():
            for pred in range(n_samples):
                start_idx = 0
                for X_val_batch, y_val_batch in test_loader:
                    X_val_batch = X_val_batch.to(pytorch_device)
                    y_val_batch = y_val_batch.to(pytorch_device)

                    batch_outputs = model(X_val_batch.to(pytorch_device))
                    # unscaled_outputs, unscaled_y = Unscale([outputs, y_batch], target_scaler)
                    # batch_loss = criterion(unscaled_outputs, unscaled_y)
                    batch_loss = criterion(batch_outputs, y_val_batch) #TODO: passing scaled loss for testing
                    val_losses.append(batch_loss) #?saving unscaled loss

                    n_samples_per_batch = batch_outputs.shape[0]
                    end_idx = start_idx + n_samples_per_batch

                    val_preds[pred, start_idx:end_idx] = batch_outputs.detach().cpu().numpy()
                    start_idx = end_idx
                
        val_loss = np.mean(val_losses)
        prediction_mean = val_preds.mean(axis=0)
        prediction_std = val_preds.std(axis=0)
        epoch_val_losses.append(val_loss.item()) # makes it a python scalar
        epoch_val_means.append(prediction_mean.tolist()) # append all test set predictions for each epoch.
        epoch_val_stds.append(prediction_std.tolist())

        pbar.set_postfix({'Train Loss': train_loss, 'Val Loss': val_loss, 'Min Val Loss': min_val_loss})

        ## Utilities (saving/progress/early stopping)
        # check if validation loss has decreased (significantly)
        if min_val_loss > val_loss and abs(min_val_loss - val_loss) > 0.001:
            tqdm.write(f'Validation loss decreased ({min_val_loss:.3f} --> {val_loss:.3f}).') if verbose else None
            
            if model_save == True: # we save model and optimizer for the best validation loss
                tqdm.write('Saving model and optimizer state...') if verbose else None
                torch.save(model.state_dict(), f'{save_dir}/model.pt')
                torch.save(optimizer.state_dict(), f'{save_dir}/optimizer.pt')
                
            epochs_no_improve = 0
            min_val_loss = val_loss
        else:
            epochs_no_improve += 1
            # check if early stopping conditions are met
            if epochs_no_improve == n_epochs_stop and early_stopping == True:
                tqdm.write('Early stopping!')
                break
        
        # step the scheduler
        param = val_loss if scheduler_type == 'ReduceLROnPlateau' else None
        # we dont need to feed in epoch for other schedulers, but val_loss is needed for ReduceLROnPlateau
        scheduler.step(param)

        tqdm.write("\nTraining reached end, consider increasing the number of epochs.") if epoch == num_epochs-1 else None

    
    print(f"---~| Saving run attributes to {save_dir} |~---")
    # Save model hyperparameters
    with open(f'{save_dir}/model_params.json', 'w') as f:
        json.dump({'input_size': model.lstm.input_size, 
                'hidden_layer_size': model.lstm.hidden_size, 
                'num_layers': model.lstm.num_layers, 
                'output_size': model.fc.out_features, 
                'dropout_rate': model.dropout.p}, 
                f)

    # helper hyperparameters
    
    if scheduler_type == 'ReduceLROnPlateau':
        add_params = {'scheduler_type': scheduler_type,
                    'patience': scheduler.patience,
                    'factor': scheduler.factor,}
    elif scheduler_type == 'CosineAnnealingWarmRestarts':
        add_params = {'scheduler_type': scheduler_type,
                    'T_0': scheduler.T_0,
                    'T_mult': scheduler.T_mult,
                    'eta_min': scheduler.eta_min}
    elif scheduler_type == 'CyclicLR':
        add_params = {'scheduler_type': scheduler_type,
                    'step_size_up': scheduler.total_size * scheduler.step_ratio,
                    'base_lr': scheduler.base_lrs[0],
                    'max_lr': scheduler.max_lrs[0],
                    'mode': scheduler.mode,
                    'cycle_momentum': scheduler.cycle_momentum,}
    
    utils_dict = {'lr': optimizer.param_groups[0]['lr'], 'num_epochs': num_epochs}
    utils_dict.update(add_params)

    with open(f'{save_dir}/util_params.json', 'w') as f:
        json.dump(utils_dict, f)

    # Save losses and forecasts
    with open(f'{save_dir}/model_results.json', 'w') as f:
        json.dump({'train_losses': epoch_train_losses, 
                    'val_losses': epoch_val_losses,
                    'val_means': epoch_val_means, # with MCD, this is mean of n forecasts, without MCD its just a single forecast
                    'val_stds': epoch_val_stds}, # MCD: std, no MCD: 0
                    f)    

    return None


## Model initializer
def init_model(input_size, hidden_layer_size, num_layers, output_size, dropout_rate, lr, pytorch_device, iter_var, scheduler=''):
    """
    iter_var: controls the scheduler iteration interaction. E.g., for a step_lr, how many iterations before decreasing the learning rate.
    """
    
    # init model
    model = LSTM(input_size, hidden_layer_size, num_layers, output_size, dropout_rate)
    model.to(pytorch_device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    if scheduler == 'ReduceLROnPlateau':
        LRScheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=iter_var)
    elif scheduler == 'CosineAnnealingWarmRestarts':
        LRScheduler = CosineAnnealingWarmRestarts(optimizer, T_0=iter_var, T_mult=1, eta_min=0.00001)
    elif scheduler == 'CyclicLR':
        LRScheduler = CyclicLR(optimizer, base_lr=0.00001, max_lr=0.001, step_size_up=iter_var, mode='triangular2', cycle_momentum=False)
    
    # ? note: the scheduler has its factor changed to 0.5
    # ? note: more schedulers for testing

    return model, criterion, optimizer, LRScheduler


if __name__ == "__main__":
    from datetime import datetime as dt
    
    conf_path = parser.parse_args().conf_path
    config = OpenConfig(conf_path)
    pytorch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    

    raw_stock_data, stock_tickers = DataLoadFromPath(config['data']['data_path'])

    # training variables
    scaler = RobustScaler() if config['data']['scaler'] == 'robust' else MinMaxScaler(feature_range=(0, 1))
    scalers = {}

    # datesplitting for use in validation
    fold_dates = [dt.strptime(d, '%Y-%m-%d') for d in config['data']['date_splits']]
    folds = range(len(fold_dates))

    # Early stopping initialization
    min_val_loss = np.Inf
    epochs_no_improve = 0
    
    num_epochs = config['train']['num_epochs']

    start_time = time.time()
    # Training, evaluation and validation. Saving models and attributes to ticker folders
    for fold in folds:
        fold_time = time.time()

        fold_scalers = {}
        fold_losses = {}

        split_date = fold_dates[fold]

        for ticker in stock_tickers:
            ticker_time = time.time()
            tqdm.write(f"Training model for {ticker} on fold {fold}.")

            # Scaling all data for the fold split. Turn them into tensors
            ticker_data = raw_stock_data[raw_stock_data.Ticker == ticker] 
            scaled_ticker_data, feature_scaler, target_scaler = ScaleData(df=ticker_data, split_date=split_date, scaler=scaler)
            fold_scalers[ticker] = (feature_scaler, target_scaler)
            print("Data scaled.")
            
            # sequencial dict of tensors with train- and test-features and -targets
            sequencial_tensors_dict = DF2Tensors(scaled_ticker_data.drop(columns=['Ticker', 'Sector']), split_date=split_date)
            print('Data turned into tensors.')

            train_loader, test_loader = FetchDataLoader(sequencial_tensors_dict, 
                                                        batch_size=config['train']['batch_size'], 
                                                        num_workers=0)
            print('Data loaders ready.')

            input_size = train_loader.dataset.X.shape[2] # calls for number of features.
            model, criterion, optimizer, scheduler = init_model(input_size=input_size, 
                                                                hidden_layer_size=config['train']['hidden_layer_size'], 
                                                                num_layers=config['train']['num_layers'], 
                                                                output_size=config['train']['output_size'], 
                                                                dropout_rate=config['train']['dropout_rate'], 
                                                                lr=config['optim']['lr'],
                                                                pytorch_device=pytorch_device, 
                                                                iter_var=config['train']['iter_var'], 
                                                                scheduler=config['train']['scheduler'])
            print('Model initialized.')
                
            save_dir = f"experiments/{config['data']['exp_output_dir']}/fold{fold}/{ticker}"
            
            # train tha model
            train_model(model=model,
                        train_loader=train_loader, 
                        test_loader=test_loader,
                        target_scaler=target_scaler,
                        criterion=criterion, 
                        optimizer=optimizer, 
                        scheduler=scheduler,
                        pytorch_device=pytorch_device,
                        clip_value=config['train']['clip_value'],
                        num_epochs=config['train']['num_epochs'],
                        n_epochs_stop=config['train']['n_epochs_stop'],
                        model_save=True,
                        save_dir=save_dir, 
                        verbose=True, 
                        early_stopping=True,
                        MCD=config['train']['MCD'],)

            tqdm.write(f"Finished training for {ticker} on fold {fold}. \nTime of training: {time.time() - ticker_time:.2f} seconds.\n")
        
        scalers[fold] = fold_scalers
        
        tqdm.write(f"Finished training for fold {fold}. \nTime for fold: {time.time() - fold_time:.2f} seconds.\n")

    tqdm.write(f"{'-'*30}\nFinished training and validation for all folds. \nElapsed time: {time.time() - start_time:.2f} seconds. \n{'-'*30}")