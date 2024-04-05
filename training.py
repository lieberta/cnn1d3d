import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models import CNN1D3D, CNN3D, CNN3D1D, Conv
from dataset import Dataset_x4_y1, Dataset_x1_y1
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import math
import time
import os # to check if a plot already exists
from torchsummary import summary


#for name, param in model.inception.named_parameters():
#    if "fc.weight" in name or "fc.bias" in name:
#        param.requires_grad = True
#    else:
#        param.requires_grad = train_CNN

#print(f'Länge des Datensets:{len(dataset)}')


def train(l, b, e, model_type):
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    print("Device = "+device)
    num_epochs = e
    learning_rate = l #0.001 #0.001  # 0.00001
    # train_CNN = False
    batch_size = b
    shuffle = True
    pin_memory = True
    num_workers = 1

    # choses right dataset based on modeltype
    if model_type == 'CNN1D3D':
        dataset = Dataset_x4_y1()     # input dimensions = (n_samples, 4, 1, 61, 81, 31)
        model = CNN1D3D().to(device)
        summary(model, (4, 1, 61, 81, 31))

    if model_type == 'CNN3D1D':
        dataset = Dataset_x4_y1()
        model = CNN3D1D().to(device)
        summary(model, (4, 1, 61, 81, 31))

    if model_type == 'CNN3D':
        dataset = Dataset_x1_y1()      # input: (batch_size, channels = 1, x = 61, y = 81, z = 31)
        model = CNN3D().to(device)

    if model_type == 'Conv':
        dataset = Dataset_x4_y1()
        model = Conv().to(device)


    dfs = dataset
    print("Modeltype:"+model_type)

    train_set, validation_set = torch.utils.data.random_split(dfs, [math.ceil(len(dfs)*0.8), math.floor(len(dfs)*0.2)])
    train_loader = DataLoader(dataset=train_set, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                              pin_memory=pin_memory)
    val_loader = DataLoader(dataset=validation_set, shuffle=shuffle, batch_size=batch_size,
                                   num_workers=num_workers, pin_memory=pin_memory)




    model = model.double()

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = dict(train=[], val=[])

    tic = time.perf_counter() # set starting timestamp

    for epoch in range(num_epochs):
        model.train()
        #for visualizing the training process

        loop = tqdm(train_loader, total = len(train_loader), leave = True)
        #if epoch % 2 == 1:
        #    print(f'Epoch {epoch} \n Loss: {loss}')
        train_losses=[]
        for input, target in loop:
        #for i, (input,target) in enumerate(train_loader):
            input = input.to(device)
            target = target.to(device)
            outputs = model(input.double())
            loss = criterion(outputs, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
            train_losses.append(loss.item())


            loop.set_postfix(trainloss = np.mean(train_losses))

        val_losses=[]
        model = model.eval()
        with torch.no_grad():
            for ind, (input, target) in enumerate(val_loader):
                input = input.to(device)
                target = target.to(device)
                outputs = model(input.double())
                loss = criterion(outputs, target)
                val_losses.append(loss.item())






        val_loss=np.mean(val_losses)
        train_loss = np.mean(train_losses)
        history['train'].append(train_loss)
        history['val'].append(val_loss)
    toc = time.perf_counter() # set ending timestamp

    #plot one validation output frame in comparison with its supposed target
    with torch.no_grad():
        for ind, (input, target) in enumerate(val_loader):
            input = input.to(device)
            target = target.to(device)
            output = model(input.double())


    # saves loss history
    np.savez(r'./Plots/numpysaves/'+model_type+'_train_val_loss'+'_epochs'+str(num_epochs)+'_lr'+str(learning_rate)+'_batch'+str(batch_size)+'.npz', trainx = range(len(history['train'])), trainy=history['train'],
                 valx=range(len(history['val'])), valy=history['val'])

    # saves loss history for comparison in evaluation.py
    np.savez(r'./Plots/numpysaves/'+model_type+'.npz', trainx = range(len(history['train'])), trainy=history['train'],
                 valx=range(len(history['val'])), valy=history['val'])


    #plot the train
    range(len(history['train'])), history['train']
    plt.plot(range(len(history['train'])),history['train'],label='Trainloss')
    plt.plot(range(len(history['val'])), history['val'],label='Validationloss')
    plt.legend()
    plt.title('Loss per Epoch')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')

    # save the plot files
    plotfilename = r'./Plots/'+model_type+r'/Loss_Plot_Temp_'+model_type+'_epochs'+str(num_epochs)+'_lr'+str(learning_rate)+'_batch'+str(batch_size)+'.png'
    i = 0 # filename additional number
    while os.path.exists(plotfilename):
        # if the file exists, add a number to the file name
        i+=1
        plotfilename = r'./Plots/'+model_type+r'/Loss_Plot_Temp_'+model_type+'_epochs'+str(num_epochs)+'_lr'+str(learning_rate)+'_batch'+str(batch_size)+f'version_{i}.png'
    plt.savefig(plotfilename)

    # save model
    torch.save(model.state_dict(),
               r'./saved_models/'+model_type+r'/' + str(model_type) + f'_epochs{num_epochs}_lr{learning_rate}_batch{batch_size}'+f'_trainloss{history["train"][-1]}')

    # save model an additional time for comparison in evaluation
    torch.save(model.state_dict(), r'./saved_models/saved'+model_type)


    plt.show()
    proc_time = toc - tic
    print(r'Model trained for {} seconds'.format(proc_time))

    return history, model


if __name__ == "__main__":
    print('Okaaay - Let\'s go...')

# execute main.py








