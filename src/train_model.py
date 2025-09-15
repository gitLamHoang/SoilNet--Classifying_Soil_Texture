from torch import optim
from data_preprocess import *
from model import CNN
input = '/content/drive/MyDrive/Dataset'
label_arr = ['handwriting', 'print_fixed']
device = 'cuda'
model = CNN().to(device)
optimizer = optim.Adam(params= model.parameters(), lr = 0.001)
loss_function = nn.CrossEntropyLoss()

def train_model(num_epoch):
    train_loader, _ = convert_to_tensor(input, label_arr)
    loss_plot = []
    accuracy_plot = []
    for epoch in range(num_epoch):
        model.train()
        sum_loss = 0
        train_acc = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            xb = xb.permute(0, 3, 1, 2).float().to(device)
            xb, yb = xb.to(device), yb.to(device)
            y_pred = model(xb)

            loss = loss_function(y_pred,yb)
            print('epoch' + str(epoch) + ' ' + str(loss.item()))
            sum_loss += loss.item()
            loss.backward()
            optimizer.step()

            pred_label = y_pred.argmax(dim=1)
            yb=yb.argmax(dim=1)
            train_Acc +=torch.sum(pred_label ==yb)

        loss_plot.append(sum_loss)
        accuracy_plot.append(train_acc/train_loader.size())
        print('Loss in epoch ' + str(epoch+1) + ': ' + str(sum_loss))
