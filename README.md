# Developing a Neural Network Regression Model
## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Explain the problem statement

## Neural Network Model
<img width="1591" height="987" alt="image" src="https://github.com/user-attachments/assets/c73c087f-797a-46a4-9848-4c3ce3b55404" />


## DESIGN STEPS
### STEP 1: 

Create your dataset in a Google sheet with one numeric input and one numeric output.

### STEP 2: 

Split the dataset into training and testing

### STEP 3: 

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4: 

Build the Neural Network Model and compile the model.

### STEP 5: 

Train the model with the training data.

### STEP 6: 

Plot the performance plot

### STEP 7: 

Evaluate the model with the testing data.

### STEP 8: 

Use the trained model to predict  for a new input value .

## PROGRAM

### Name:KARTHICK S

### Register Number:212224230114

```python
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        elf.fc1 = nn.Linear(1, 8)
        self.fc2 = nn.Linear(8, 10)  
        self.fc3 = nn.Linear(10, 1)
        self.relu = nn.ReLU()
        self.history = {'loss': []}



# Initialize the Model, Loss Function, and Optimizer
ai_brain = NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(ai_brain.parameters(), lr=0.001)


def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = criterion(ai_brain(X_train), y_train)
        loss.backward()
        optimizer.step()

        ai_brain.history['loss'].append(loss.item())
        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')

train_model(ai_brain, X_train_tensor, y_train_tensor, criterion, optimizer)

with torch.no_grad():
    test_loss = criterion(ai_brain(X_test_tensor), y_test_tensor)
    print(f'Test Loss: {test_loss.item():.6f}')

import matplotlib.pyplot as plt
loss_df.plot()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss during Training")
plt.show()

X_n1_1 = torch.tensor([[9]], dtype=torch.float32)
prediction = ai_brain(torch.tensor(scaler.transform(X_n1_1), dtype=torch.float32)).item()
print(f'Prediction: {prediction}')

```

### Dataset Information
<img width="219" height="304" alt="image" src="https://github.com/user-attachments/assets/3cb0f1ca-ef4c-4581-95dd-a1be80b86065" />


### OUTPUT
<img width="887" height="435" alt="image" src="https://github.com/user-attachments/assets/2fb9df4f-8b62-40ef-93d3-6a239e948162" />


### Training Loss Vs Iteration Plot
<img width="580" height="455" alt="image" src="https://github.com/user-attachments/assets/cfae8f2b-0b27-42aa-94b1-1aab5807942d" />


### New Sample Data Prediction
<img width="992" height="150" alt="image" src="https://github.com/user-attachments/assets/e47bc629-9bd9-4eaa-81f3-d5fc04d32b28" />


## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
