import pandas as pd
import numpy as np
import json
import time
import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from tqdm import tqdm

from sklearn.preprocessing import RobustScaler, MinMaxScaler

from preprocessing_utils import DF2Tensors, ScaleData, FetchDataLoader
tqdm.write("Modules loaded.")

parser = argparse.ArgumentParser()
# parser.add_argument("--exp_dir", help="Full path to save best validation model")
parser.add_argument("--conf_path", default="conf_base.yml", help="Name of configuration file (conf___.yml)")

##### MODEL
## Define model architecture
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_layer_size, num_layers, output_size, dropout_rate):
        super(LSTM, self).__init__()
        
        # Define the dimensions of the LSTM layer
        self.hidden_dim = hidden_layer_size
        self.num_layers = num_layers
        
        # Define the LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, batch_first=True)
        
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


## Setup trainer function
def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, pytorch_device,
                num_epochs=50, n_epochs_stop=10, min_val_loss=np.Inf, epochs_no_improve=0, 
                model_save=True, save_dir='', verbose=True, early_stopping=True):
    
    os.makedirs(save_dir, exist_ok=True)
    epoch_val_losses = []
    epoch_train_losses = []

    pbar = tqdm(range(num_epochs), desc="Training", leave=True, unit='epochs')
    for epoch in pbar:
        model.train()

        train_losses = []
        pbar_inner = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False, unit='batches')
        for batch_X, batch_y in pbar_inner:
            batch_X = batch_X.to(pytorch_device)
            batch_y = batch_y.to(pytorch_device)

            # forward propagation
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # backward propagation
            optimizer.zero_grad()
            loss.backward()

            # update weights
            optimizer.step()

            # save losses
            train_losses.append(loss.item())

        # calculate average losses
        train_loss = np.mean(train_losses)
        epoch_train_losses.append(train_loss)
        
        
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_val_batch, y_val_batch in test_loader:
                val_outputs = model(X_val_batch.to(pytorch_device))
                val_loss = criterion(val_outputs, y_val_batch.to(pytorch_device)).item()
                val_losses.append(val_loss)
        val_loss = np.mean(val_losses)
        epoch_val_losses.append(val_loss)
        
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
        scheduler.step(val_loss)

        tqdm.write("\nTraining reached end, consider increasing the number of epochs.") if epoch == num_epochs-1 else None


    # Save model hyperparameters
    with open(f'{save_dir}/model_params.json', 'w') as f:
        json.dump({'input_size': input_size, 
                'hidden_layer_size': hidden_layer_size, 
                'num_layers': num_layers, 
                'output_size': output_size, 
                'dropout_rate': dropout_rate}, 
                f)

    # helper hyperparameters
    with open(f'{save_dir}/util_params.json', 'w') as f:
        json.dump({'lr': lr, 
                'patience': patience, 
                'num_epochs': num_epochs}, 
                f)
    
    # Save the losses for the current ticker
    with open(f'{save_dir}/model_losses.json', 'w') as f:
        json.dump({'train_losses': epoch_train_losses, 
                'val_losses': epoch_val_losses}, 
                f)

    return epoch_train_losses, epoch_val_losses, min_val_loss


## Model initializer
def init_model(input_size, hidden_layer_size, num_layers, output_size, dropout_rate, lr, patience, pytorch_device, scheduler=True):
    # init model
    model = LSTM(input_size, hidden_layer_size, num_layers, output_size, dropout_rate)
    model.to(pytorch_device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=patience) if scheduler else None

    return model, criterion, optimizer, scheduler


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

if __name__ == "__main__":
    import yaml
    
    conf_path = parser.parse_args().conf_path
    config = OpenConfig(f'configs/{conf_path}')
    pytorch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    

    raw_stock_data, stock_tickers = DataLoadFromPath(config['data_path'])

    input_size = config['input_size']
    hidden_layer_size = config['hidden_layer_size']
    num_layers = config['num_layers']
    output_size = config['output_size']
    dropout_rate = config['dropout_rate']
    lr = config['lr']
    patience = config['patience']
    # Early stopping details
    n_epochs_stop = config['n_epochs_stop']
    min_val_loss = np.Inf
    epochs_no_improve = 0
    # Epochs
    num_epochs = config['num_epochs']

    scalers = {}

    # init variables for use in the loop
    fold_dates = config['date_splits'] # yaml detects datetime
    folds = range(len(fold_dates))

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
            scaled_ticker_data, ticker_scaler = ScaleData(df=ticker_data, ticker=ticker, split_date=split_date, scaler=MinMaxScaler())
            fold_scalers[ticker] = ticker_scaler
            
            # sequencial dict of tensors with train- and test-features and -targets
            sequencial_tensors_dict = DF2Tensors(scaled_ticker_data.drop(columns=['Ticker', 'Sector']), split_date=split_date)
            
            train_loader, test_loader = FetchDataLoader(sequencial_tensors_dict, 
                                                        batch_size=config['train']['batch_size'], 
                                                        num_workers=0)

            model, criterion, optimizer, scheduler = init_model(input_size=config['train']['input_size'], 
                                                                hidden_layer_size=config['train']['hidden_layer_size'], 
                                                                num_layers=config['train']['num_layers'], 
                                                                output_size=config['train']['output_size'], 
                                                                dropout_rate=config['train']['dropout_rate'], 
                                                                lr=config['optim']['lr'],
                                                                optim_type=config['optim']['type'], 
                                                                patience=config['train']['patience'], 
                                                                pytorch_device=pytorch_device, 
                                                                scheduler=True)
                
            save_dir = f"experiments/{config['data']['exp_dir_out']}/fold{fold}/{ticker}"
            
            # train tha model
            epoch_train_losses, epoch_val_losses, min_val_loss = train_model(model=model,
                                                                            train_loader=train_loader, 
                                                                            test_loader=test_loader,
                                                                            criterion=criterion, 
                                                                            optimizer=optimizer, 
                                                                            scheduler=scheduler,
                                                                            pytorch_device=pytorch_device,
                                                                            num_epochs=config['train']['num_epochs'],
                                                                            model_save=True,
                                                                            save_dir=save_dir, 
                                                                            verbose=True, 
                                                                            early_stopping=True)

            tqdm.write(f"Finished training for {ticker} on fold {fold}. \nTime of training: {time.time() - ticker_time:.2f} seconds.\n")
        
        scalers[fold] = fold_scalers
        
        tqdm.write(f"Finished training for fold {fold}. \nTime for fold: {time.time() - fold_time:.2f} seconds.\n")

    tqdm.write(f"{'-'*30}\nFinished training and validation for all folds. \nElapsed time: {time.time() - start_time:.2f} seconds. \n{'-'*30}")