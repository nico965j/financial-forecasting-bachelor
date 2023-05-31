import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import time
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

from sklearn.preprocessing import RobustScaler, MinMaxScaler

from preprocessing_utils import DF2Tensors, ScaleData, FetchDataLoader
print("Modules loaded.")

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
                model_save=True, verbose=True, early_stopping=True):
    
    epoch_val_losses = []
    epoch_train_losses = []

    for epoch in range(num_epochs):
        model.train()

        train_losses = []
        for batch_X, batch_y in train_loader:
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

        # progress metering
        if epoch % 5 == 0 and verbose:
            print(f"Epoch {epoch+1}/{num_epochs}.. Train loss: {train_loss:.3f}, Validation loss: {val_loss:.3f}")

        # check if validation loss has decreased (significantly)
        if val_loss < min_val_loss and abs(min_val_loss - val_loss) > 0.001:
            print(f'Validation loss decreased ({min_val_loss:.3f} --> {val_loss:.3f}).') if verbose else None
            if model_save == True: # we save model and optimizer for the best validation loss
                print('Saving model and optimizer state...') if verbose else None
                torch.save(model.state_dict(), 'model.pt')
                torch.save(optimizer.state_dict(), 'optimizer.pt')

            epochs_no_improve = 0
            min_val_loss = val_loss
        else:
            epochs_no_improve += 1
            # check if early stopping conditions are met
            if epochs_no_improve == n_epochs_stop and early_stopping == True:
                print('Early stopping!')
                break

        # step the scheduler
        scheduler.step(val_loss)

        print("\nTraining reached end, consider increasing the number of epochs.") if epoch == num_epochs-1 else None

    return epoch_train_losses, epoch_val_losses, min_val_loss


## Model initializer
def init_model(input_size, hidden_layer_size, num_layers, output_size, dropout_rate, lr, patience, pytorch_device, scheduler=True):
    # init model
    model = LSTM(input_size, hidden_layer_size, num_layers, output_size, dropout_rate)
    model.to(pytorch_device)
    # print(model)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=patience, verbose=True) if scheduler else None

    return model, criterion, optimizer, scheduler



if __name__ == "__main__":

    ### Initializations

    # set device
    pytorch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # set seed
    def assert_seed(seed=42):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    start_time = time.time()
    print("Data load...")
    # load csv
    raw_stock_data = pd.read_csv('data/SP500_stock_prices_log_clean_3monthreturn.csv', index_col=0, parse_dates=True)
    stock_tickers = pd.Series(raw_stock_data.Ticker.unique())

    # change doubles/64 bit floats into 32bit floats
    float_cols = list(raw_stock_data.select_dtypes(include=['float64', 'int64']))
    raw_stock_data[float_cols] = raw_stock_data[float_cols].astype('float32')
    end_time = time.time()
    print(f"Data load complete. Time elapsed: {end_time - start_time} seconds.")


    input_size = 6
    hidden_layer_size = 32
    num_layers = 1
    output_size = 1
    dropout_rate = 0.4
    lr = 0.001
    patience = 3

    # Early stopping details
    n_epochs_stop = 10
    min_val_loss = np.Inf
    epochs_no_improve = 0

    # Epochs
    num_epochs = 100

    scalers = {} # will have structure: scalers[fold_num][ticker]
    # losses = dict() #or we read in the relevant csv/json files created

    # init variables for use in the loop
    fold_dates = [pd.to_datetime(str) for str in ["2019-01-01", "2019-04-01"]]
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

            print(f"Training model for {ticker} on fold {fold}.")
            # Scaling all data for the fold split. Turn them into tensors
            ticker_data = raw_stock_data[raw_stock_data.Ticker == ticker]
            scaled_ticker_data, ticker_scaler = ScaleData(df=ticker_data, split_date=split_date, scaler=MinMaxScaler())
            fold_scalers[ticker] = ticker_scaler
            
            # sequencial dict of tensors with train- and test-features and -targets
            sequencial_tensors = DF2Tensors(scaled_ticker_data, split_date=split_date)
            
            train_loader, test_loader = FetchDataLoader(sequencial_tensors, batch_size=16, num_workers=0)

            model, criterion, optimizer, scheduler = init_model(input_size, 
                                                                hidden_layer_size, 
                                                                num_layers, 
                                                                output_size, 
                                                                dropout_rate, 
                                                                lr, 
                                                                patience, 
                                                                pytorch_device, 
                                                                scheduler=True)
                
            # train tha model
            epoch_train_losses, epoch_val_losses, min_val_loss = train_model(model=model,
                                                                            train_loader=train_loader, 
                                                                            test_loader=test_loader,
                                                                            criterion=criterion, 
                                                                            optimizer=optimizer, 
                                                                            scheduler=scheduler,
                                                                            pytorch_device=pytorch_device,
                                                                            num_epochs=num_epochs,
                                                                            model_save=False, 
                                                                            verbose=True, 
                                                                            early_stopping=True)

            # Common directory path for saving model, optimizer, losses, and hyperparameters
            save_dir = f'models/{fold}/{ticker}'
            os.makedirs(save_dir, exist_ok=True)

            # Save the model and hyperparameters
            torch.save(model.state_dict(), f'{save_dir}/model.pt')
            with open(f'{save_dir}/model_params.json', 'w') as f:
                json.dump({'input_size': input_size, 
                        'hidden_layer_size': hidden_layer_size, 
                        'num_layers': num_layers, 
                        'output_size': output_size, 
                        'dropout_rate': dropout_rate}, 
                        f)

            # optimizer
            torch.save(optimizer.state_dict(), f'{save_dir}/optimizer.pt')
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
            
            print(f"Finished training for {ticker} on fold {fold}. Time of training: {time.time() - ticker_time:.2f} seconds.")
        
        scalers[fold] = fold_scalers
        
        print(f"Finished training for fold {fold}. Time for fold: {time.time() - fold_time:.2f} seconds.")

    print(f"Finished training and validation for all folds. Elapsed time: {time.time() - start_time:.2f} seconds.")