# **Malaria Detection Using CNN**  

This project implements a **Convolutional Neural Network (CNN) from scratch** to classify malaria-infected and uninfected blood cell images. The pipeline includes **feature selection, data preprocessing, model training, and evaluation** using performance metrics like precision, recall, and F1-score.  

## **1️⃣ Feature Selection**  
Malaria detection is based on microscopic images of blood smears. The key features used for classification include:  
✅ **Cell Color & Texture** – Differentiating parasitized and uninfected cells  
✅ **Cell Shape & Size** – Infected cells may appear deformed or irregular  
✅ **Edge Features** – Parasites within cells create unique patterns  

Since CNNs automatically learn hierarchical **spatial features**, no manual feature extraction is required. Instead, the model extracts **low-level features** (edges, textures) in early layers and **high-level features** (shapes, objects) in deeper layers.  

---

## **2️⃣ Data Preprocessing & Transformations**  
Before training the CNN, the images undergo several transformations to improve generalization and reduce overfitting.  

### **📌 Image Transformations**
Using **Torchvision**, we apply:  
✔ **Resizing**: Standardizing images to **64x64 pixels** for uniform input size  
✔ **Normalization**: Scaling pixel values to **(-1, 1)** for stable gradient updates  
✔ **Data Augmentation**:  
   - **Random Horizontal Flip** → Helps model learn position invariance  
   - **Random Rotation** → Prevents orientation bias  
   - **Color Jitter** → Simulates variations in staining and lighting  

```python
import torchvision.transforms as transforms

train_transforms = transforms.Compose([
    transforms.Resize((64, 64)),  # Standardize size
    transforms.RandomHorizontalFlip(),  
    transforms.RandomRotation(10),  
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

val_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
```

✔ **Batching**: Uses a **batch size of 64** for efficient training  
✔ **Shuffling**: Randomizes order of training data for better generalization  

---

## **3️⃣ CNN Model Architecture**  
The CNN is designed to learn spatial features from input images using convolutional layers followed by fully connected layers.  

```python
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(8*8*64, 500),
            nn.ReLU(),
            nn.Dropout(p=0.4),  
            nn.Linear(500, 100),
            nn.ReLU(),
            nn.Dropout(p=0.4),  
            nn.Linear(100, 2)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = x.view(x.shape[0], -1)  # Flatten
        x = self.fc(x)
        return x
```

✔ **Conv Layers**: Extract hierarchical features  
✔ **MaxPooling**: Reduces dimensionality while retaining key patterns  
✔ **Dropout (0.4)**: Prevents overfitting by randomly deactivating neurons  
✔ **Fully Connected Layers**: Classifies into **Parasitized vs. Uninfected**  

---

## **4️⃣ Training the Model**  
The model is trained using **Cross-Entropy Loss** (suitable for classification) and **Adam Optimizer** (for adaptive learning rates).  

```python
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
```

✔ **Learning Rate**: `0.001`, reduced dynamically if validation loss stops improving  
✔ **Weight Decay**: `1e-4` to prevent large weight values  
✔ **Early Stopping**: Stops training if validation loss stagnates  

---

## **5️⃣ Model Evaluation**  
After training, the model is tested on unseen validation data. Key performance metrics include:  
![Confusion Matrix](https://github.com/bonsoul/Malaria_Detection_CNN/blob/main/Training%20Validation%20Loss%20with%20CNN%20from%20scratch.png)

### **📌 Confusion Matrix**
```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Parasitized', 'Uninfected'], yticklabels=['Parasitized', 'Uninfected'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('plots/confusion_matrix.png')
plt.show()
```
✔ **Visualizes model performance** on classification  

### **📌 Precision, Recall & F1-Score**
```python
from sklearn.metrics import precision_score, recall_score, f1_score

prec = precision_score(y_true, y_pred, average='macro')
rec = recall_score(y_true, y_pred, average='macro')
f1 = f1_score(y_true, y_pred, average='macro')

print(f"Precision: {prec:.4f}, Recall: {rec:.4f}, F1-score: {f1:.4f}")
```
✔ **Precision**: Measures accuracy of positive predictions  
✔ **Recall**: Measures ability to find all positive cases  
✔ **F1-Score**: Balances precision & recall  

---

## **6️⃣ Results & Improvements**
### **📌 Current Observations**
✔ Model achieves **high accuracy on training data** but validation loss **increases after epoch 4**, indicating **overfitting**.  
✔ **Precision and recall scores suggest** that false positives & false negatives need to be minimized.  

### **📌 Improvements**
✅ **Increase Dropout to 0.5** → Reduces reliance on specific neurons  
✅ **Use Data Augmentation** → Helps generalization  
✅ **Reduce Learning Rate Dynamically** → Avoids overshooting minima  
✅ **Increase Training Data** → Helps reduce model bias  

---

## **7️⃣ Conclusion**
This CNN-based model provides **promising results** for malaria detection using blood smear images. However, further tuning is required to **minimize overfitting** and **improve real-world performance**. Future work includes **hyperparameter tuning, deeper architectures, and transfer learning** with pre-trained models like **ResNet**.

---

### **🚀 How to Run?**
1️⃣ **Install Dependencies**  
```bash
pip install torch torchvision matplotlib seaborn scikit-learn
```
2️⃣ **Train Model**  
```python
model_scratch, train_losses, val_losses = train(20, dataloaders, model, optimizer, criterion, device, 'models/model_scratch.pt')
```
3️⃣ **Evaluate Performance**  
```python
y_pred = model.predict(test_loader)
cm = confusion_matrix(y_test, y_pred)
print("Precision:", precision_score(y_test, y_pred))
```

