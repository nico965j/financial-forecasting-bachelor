


# training loop

# we will have to save some variables for later use. -scalers, -losses,
scalers = dict() #will have structure: scalers[fold_num][ticker]

losses = dict() #or we read in the relevant csv/json files created

# init variables for use in the loop
fold_dates = ["___","___",...]


# outermost, we have different "folds", like cross-validation (differnet splits of data) based on date.
for fold in folds:
    fold_scalers = dict() #containing the scalers for each ticker in this particular fold
    fold_losses = dict()...

    split_date = fold_dates[fold]

    for ticker in tickers:
        --- Scaling all data for the fold split. Turn them into tensors
        scaled_ticker_data, ticker_scaler = ScaleData(data[fold][ticker], split_date=split_date, scalertype)
        fold_scalers[ticker] = ticker_scaler
        
        sequencial_tensors = DF2Tensors(scaled_ticker_data, split_date=split_date)
        
        train_loader, test_loader = GetDataLoader(sequencial_tensors, batch_size=16, num_workers=2)

        --- init model
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
                                                                        num_epochs=num_epochs,
                                                                        model_save=True, 
                                                                        verbose=True, 
                                                                        early_stopping=True)

        # save model and optimizer

        # save model losses

        # save model params

        # save model hyperparameters

    end

scalers[fold] = fold_scalers

end