from model import *
from train_model import *
from data_preprocess import *
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

loaded_model = SoilNet().to(device)
loaded_model.load_state_dict(torch.load("cnn_soil.pth"))
loaded_model.eval()
_, test_loader = train_and_test_loader(input_path = input, label_array = label_arr, train_ratio=0.8)
correct, total, test_loss = 0, 0, 0.0

all_preds = []
all_labels = []

with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        y_pred = loaded_model(xb)

        loss = loss_function(y_pred, yb)
        test_loss += loss.item()

        pred_label = y_pred.argmax(dim=1)

        all_preds.extend(pred_label.cpu().numpy())
        all_labels.extend(yb.cpu().numpy())

        correct += (pred_label == yb).sum().item()
        total += yb.size(0)

test_acc = correct / total
test_loss = test_loss / len(test_loader)

print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()