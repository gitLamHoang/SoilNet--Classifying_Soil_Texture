from torch import optim
from data_preprocess import *
from model import SoilNet
input = '/content/drive/MyDrive/Dataset'
label_arr = ['coarse', 'fine', 'medium']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SoilNet(num_classes=3).to(device)
optimizer = optim.Adam(params= model.parameters(), lr = 0.001)
loss_function = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
num_epoch = 50

def train_model(num_epoch):
    train_loader, _ = convert_to_tensor(input, label_arr)
    loss_plot = []
    accuracy_plot = []
    for epoch in range(num_epoch):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        print(f"\n===== Epoch {epoch+1}/{num_epoch} =====")
        for batch_idx, (xb, yb) in enumerate(train_loader):
            xb, yb = xb.to(device), yb.to(device).long()

            # Forward
            y_pred = model(xb)
            loss = loss_function(y_pred, yb)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Metrics
            epoch_loss += loss.item()
            pred_label = y_pred.argmax(dim=1)
            correct += (pred_label == yb).sum().item()
            total += yb.size(0)

            # Log batch
            if (batch_idx+1) % 10 == 0:
                acc_batch = (pred_label == yb).float().mean().item()
                print(f"  Batch {batch_idx+1}/{len(train_loader)} - "
                    f"loss: {loss.item():.4f}, acc: {acc_batch:.4f}")

        # Epoch metrics
        avg_loss = epoch_loss / len(train_loader)
        avg_acc = correct / total
        loss_plot.append(avg_loss)
        accuracy_plot.append(avg_acc)
        scheduler.step()

        print(f"==> Epoch {epoch+1}: avg_loss={avg_loss:.4f}, avg_acc={avg_acc:.4f}")