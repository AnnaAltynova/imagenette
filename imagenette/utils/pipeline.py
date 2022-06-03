import torch
import json
import os
from sklearn.metrics import accuracy_score
from IPython.display import clear_output
import matplotlib.pyplot as plt
 

def make_plot(epoch_history, train_history, valid_history, accuracy_history):
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(18, 8))
    clear_output(True)
    
    ax[0].plot(epoch_history, label='epoch train loss')
    ax[0].set_xlabel('Batch')
    ax[0].set_title('Epoch train loss')
    ax[0].legend()
    
    if train_history is not None:
        ax[1].plot(train_history, label='general train history')
        ax[1].set_xlabel('Epoch')
        
    if valid_history is not None:
        ax[1].set_title('Loss')
        ax[1].plot(valid_history, label='general valid history')
        ax[1].legend()
        
    if accuracy_history is not None:
        ax[2].set_title('Accuracy')
        ax[2].plot(accuracy_history, label='general accuracy history')
        ax[2].set_xlabel('Epoch')
        ax[2].legend()

    plt.show()



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
        
        history.append(loss.cpu().data.numpy())
        if (i + 1) % 10 == 0:
            make_plot(history, train_history, valid_history, accuracy_history)
            
    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion, device):
    
    model.eval()
    
    epoch_loss = 0
    epoch_accuracy = 0
    
    with torch.no_grad():
    
        for i, (imgs, labels) in enumerate(iterator):

            imgs = imgs.to(device)
            labels_device = labels.to(device)

            output = model(imgs)
            predictions = output.argmax(axis=1).detach().cpu().numpy().astype(int)

            loss = criterion(output, labels_device)
            
            epoch_loss += loss.item()
            
            epoch_accuracy += accuracy_score(labels.numpy().astype(int), predictions)
         
    return epoch_loss / len(iterator), epoch_accuracy / len(iterator)


def train_procedure(n_epochs, model, train_iterator, val_iterator, optimizer, criterion,
                   saves_path, device, start_epoch=0, scheduler=None, model_name='vgg11'):
    logs_path = os.path.join(saves_path, 'logs.json')
    ckpt_path = os.path.join(saves_path, f'ckpts/{model_name}_model.pt')
    
    if os.path.isfile(logs_path):
        with open(logs_path, 'r') as f:
            logs = json.load(f)
    else:
        logs = {}
    
    if model_name in logs:
        model_logs = logs[model_name]
    else:
        model_logs = {'train_history': [], 'valid_history': [], 'accuracy_history': []}
        logs[model_name] = model_logs
   
    best_valid_loss = float('inf')
    
    for epoch in range(start_epoch, n_epochs):
    
        train_loss = train(model, train_iterator, optimizer, criterion, device,  model_logs['train_history'],
                           model_logs['valid_history'], model_logs['accuracy_history']) 
        valid_loss, accuracy = evaluate(model, val_iterator, criterion, device)
        
        if scheduler is not None:
            scheduler.step(valid_loss)
        
        model_logs['train_history'].append(train_loss)
        model_logs['valid_history'].append(valid_loss)
        model_logs['accuracy_history'].append(accuracy)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        }, ckpt_path)
            
        with open(logs_path, 'w') as f:
            json.dump(logs, f)
        
        
def resume_training(model, optimizer, ckpt_path):
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cuda()
            
    return epoch
