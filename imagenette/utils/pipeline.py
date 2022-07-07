import torch
import torchvision.models as models
import json
import os
from sklearn.metrics import accuracy_score
from IPython.display import clear_output
import matplotlib.pyplot as plt
 

def make_plot(epoch_history, train_history, valid_history, accuracy_history):
    """plot training process"""
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
    

def plot_models(models_logs=['logs_resnet18_aug_lr4.json'], saves_path='saves/logs', title=None):
    """plot training logs of list of models"""
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 8))
    if title is not None:
        plt.suptitle(title, fontsize='xx-large', x=0.5, y=0.95)
    ax[0].set_title('Loss')
    ax[0].set_xlabel('Epoch')
    ax[1].set_title('Accuracy')
    ax[1].set_xlabel('Epoch')

    colors = list('bgrcmyk')
    lines = []
    line_labels = []

    for i, filelog in enumerate(models_logs):
        with open(os.path.join(saves_path, filelog), 'r') as f:
            logs = json.load(f)

        model_name = list(logs.keys())[0]
        color = colors[i]
        lines.append(ax[0].plot(logs[model_name]['train_history'], '-', color=color)[0])
        line_labels.append(f'{model_name}')
        lines.append(ax[0].plot(logs[model_name]['valid_history'], '--', color=color)[0])
        line_labels.append(f'val_{model_name}')
        ax[1].plot(logs[model_name]['accuracy_history'], '-', color=color)[0]


    fig.legend(handles=lines,     
               labels=line_labels,   
               loc="center right",   
               borderaxespad=0.1,    
               title="Legend"  
               )

    plt.subplots_adjust(right=0.88)
    return 


def train(model, iterator, optimizer, criterion, device, train_history=None, valid_history=None, accuracy_history=None, staged=False):
    model.train()
    
    epoch_loss = 0
    history = []
    for i, (imgs, labels) in enumerate(iterator):
        
        imgs = imgs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        output = model(imgs)
        if staged:
            output = output[-1] 
        loss = criterion(output, labels)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
        history.append(loss.cpu().data.numpy())
        
        if (i + 1) % 10 == 0:
            make_plot(history, train_history, valid_history, accuracy_history)
            
    return epoch_loss / len(iterator)


def accuracy2d_mean(preds, mask):
    not_background_mask = mask[mask != 0]
    preds_correct = preds[(mask == preds)]
    preds_correct_not_background = preds_correct[preds_correct != 0]
    return preds_correct_not_background.size / not_background_mask.size


def evaluate(model, iterator, criterion, device, segmentation=False, staged=False):
    
    model.eval()
    
    epoch_loss = 0
    epoch_accuracy = 0
    
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(iterator):
            imgs = imgs.to(device)
            labels_device = labels.to(device)

            output = model(imgs)
            if staged:
                output = output[-1]
            predictions = output.argmax(axis=1).detach().cpu().numpy().astype(int)

            loss = criterion(output, labels_device)

            epoch_loss += loss.item()

            if segmentation:
                epoch_accuracy += accuracy2d_mean(predictions, labels.numpy())
            else:
                epoch_accuracy += accuracy_score(labels.numpy().astype(int), predictions)
         
    return epoch_loss / len(iterator), epoch_accuracy / len(iterator)


def train_procedure(n_epochs, model, train_iterator, val_iterator, optimizer, criterion,
                   saves_path, device, start_epoch=0, scheduler=None, model_name='vgg11',
                    segmentation=False, staged=False):
    
    logs_path = os.path.join(saves_path, f'logs/logs_{model_name}.json')
    ckpt_path = os.path.join(saves_path, f'ckpts3/{model_name}_model.pt')
    
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
    
        train_loss = train(model=model, iterator=train_iterator, optimizer=optimizer,
                           criterion=criterion, device=device,  train_history=model_logs['train_history'],
                           valid_history=model_logs['valid_history'],
                           accuracy_history=model_logs['accuracy_history'],
                           staged=staged)
        
        valid_loss, accuracy = evaluate(model=model, iterator=val_iterator,
                                        criterion=criterion, device=device, 
                                        segmentation=segmentation,
                                        staged=staged)
        
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
        
        
def load_model(model, ckpt_path, optimizer=None, device=None):
    checkpoint = torch.load(ckpt_path, device)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda(device)

    return epoch
