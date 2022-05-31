import torch
import pickle
import os
from sklearn.metrics import accuracy_score
import pandas as pd
from IPython.display import clear_output
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (8.0, 8.0) 
plt.rcParams['image.interpolation'] = 'nearest'


def train(model, iterator, optimizer, criterion, device, train_history=None, valid_history=None, accuracy_history=None):
    model.train()
    
    epoch_loss = 0
    history = []
    for i, (imgs, labels) in enumerate(iterator):
        
        imgs = imgs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        output = model(imgs)
        
        loss = criterion(output, labels)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
        history.append(loss.cpu().data.numpy()) # см доки detach 
        if (i+1)%10==0:
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 8))

            clear_output(True)
            ax[0].plot(history, label='train loss')
            ax[0].set_xlabel('Batch')
            ax[0].set_title('Train loss')
            if train_history is not None:
                ax[1].plot(train_history, label='general train history')
                ax[1].set_xlabel('Epoch')
            if valid_history is not None:
                ax[1].plot(valid_history, label='general valid history')
            if accuracy_history is not None:
                ax[1].plot(accuracy_history, label='general accuracy history')
            plt.legend()
            
            plt.show()

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion, num_to_name, device):
    
    model.eval()
    
    epoch_loss = 0
    epoch_accuracy = 0
    
    predictions = pd.DataFrame(columns=['true_num', 'pred_num', 'true_name', 'pred_name'])
    
    with torch.no_grad():
    
        for i, (imgs, labels) in enumerate(iterator):

            imgs = imgs.to(device)

            output = model(imgs).detach().cpu()
            preds = output.argmax(axis=1).numpy()
        
            batch_preds = pd.DataFrame({'true_num': labels.numpy(), 'pred_num': preds})
            predictions = predictions.append(batch_preds, ignore_index=True)

            loss = criterion(output, labels)
            
            epoch_loss += loss.item()
            
            
        epoch_accuracy += accuracy_score(predictions['true_num'].astype(int), predictions['pred_num'].astype(int))
        
        predictions['pred_name'] = predictions['pred_num'].apply(lambda x: num_to_name[x])
        predictions['true_name'] = predictions['true_num'].apply(lambda x: num_to_name[x])
        
    return epoch_loss / len(iterator), epoch_accuracy, predictions


def train_pipeline(n_epochs, model, train_iterator, val_iterator, optimizer, criterion,
                   saves_path, num_to_name, device, scheduler=None, model_name='vgg11'):
    train_history = []
    valid_history = []
    accuracy_history = []
    
    best_valid_loss = float('inf')
    ckpt_path = os.path.join(saves_path, 'ckpts')
    losses_path = os.path.join(saves_path, 'losses')
    predictions_path = os.path.join(saves_path, 'predictions')
    
    for epoch in range(n_epochs):
    
        train_loss = train(model, train_iterator, optimizer, criterion, device, train_history, valid_history, accuracy_history)
        valid_loss, epoch_accuracy, epoch_predictions = evaluate(model, val_iterator, criterion, num_to_name, device)
        if scheduler is not None:
            scheduler.step(valid_loss)
            
        
        train_history.append(train_loss)
        valid_history.append(valid_loss)
        accuracy_history.append(epoch_accuracy)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), os.path.join(ckpt_path, f'{model_name}_model.pt'))

            with open(os.path.join(losses_path, f'train_loss_{model_name}.pickle'), 'wb') as f:
                pickle.dump(train_history, f)
            with open(os.path.join(losses_path, f'val_loss_{model_name}.pickle'), 'wb') as f:
                pickle.dump(valid_history, f)
            with open(os.path.join(losses_path, f'accuracy_{model_name}.pickle'), 'wb') as f:
                pickle.dump(accuracy_history, f)
            with open(os.path.join(predictions_path, f'predictions_{model_name}.csv'), 'w') as f:
                epoch_predictions.to_csv(f)

