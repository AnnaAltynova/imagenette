import torch
import torchvision.models as models
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
    

def plot_models(models_logs=['logs_resnet18_aug_lr4.json'], saves_path='saves', title=None):
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
    logs_path = os.path.join(saves_path, f'logs_{model_name}.json')
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
        
        
def load_model(model, optimizer, ckpt_path, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
    checkpoint = torch.load(ckpt_path, device)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cuda(device)
            
    return epoch



def check_parameters():
    resnet18 = Resnet(out_dim=1000)
    my_18_param_cnt = resnet18.count_parameters()
    resnet18_torch = models.resnet18(pretrained=False)
    torch_18_param_cnt = sum(p.numel() for p in resnet18_torch.parameters() if p.requires_grad)
    print(f'torch resnet18 param: {torch_18_param_cnt}, custom resnet18 param: {my_18_param_cnt}')
    del resnet18
    del resnet18_torch
    resnet34 = Resnet(num_layers=34, out_dim=1000)
    my_34_param_cnt = resnet34.count_parameters()
    resnet34_torch = models.resnet34(pretrained=False)
    torch_34_param_cnt = sum(p.numel() for p in resnet34_torch.parameters() if p.requires_grad)
    print(f'torch resnet34 param: {torch_34_param_cnt}, custom resnet34 param: {my_34_param_cnt}')
    del resnet34
    del resnet34_torch
    resnet50 = Resnet(num_layers=50, out_dim=1000)
    my_50_param_cnt = resnet50.count_parameters()
    resnet50_torch = models.resnet50(pretrained=False)
    torch_50_param_cnt = sum(p.numel() for  p in resnet50_torch.parameters() if p.requires_grad)
    print(f'torch resnet50 param: {torch_50_param_cnt}, custom resnet50 param: {my_50_param_cnt}')