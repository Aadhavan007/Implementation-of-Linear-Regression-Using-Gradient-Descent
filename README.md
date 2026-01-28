## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
~~~
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("ex3.csv")

x = data['R&D Spend'].values
y = data['Profit'].values

import numpy as np
import matplotlib.pyplot as plt

w = 0.0
b = 0.0
alpha = 0.1
epochs = 100
n = len(x)
x = (x - np.mean(x)) / np.std(x)
losses = []


for _ in range(epochs):
    y_hat = w * x + b
    loss = np.mean((y_hat - y) ** 2)
    losses.append(loss)

    dw = (2/n) * np.sum((y_hat - y) * x)
    db = (2/n) * np.sum(y_hat - y)

    w -= alpha * dw
    b -= alpha * db

plt.figure(figsize=(15,5))
plt.subplot(1, 2, 1)
plt.plot(losses, color="blue")
plt.xlabel("No of Iterations")
plt.ylabel("Loss")
plt.title("LOSS VS ITERATIONS")

plt.figure(figsize=(11, 5))
plt.subplot(1, 2, 2)
plt.scatter(x, y, color="red", label="Data")
plt.plot(x, w * x + b, color="green", label="Regression Line")
plt.xlabel("R&D Spend")
plt.ylabel("Profit")
plt.title("PROFIT VS R&D SPEND")
plt.legend()

plt.tight_layout()
plt.show()

print("Final weight (w):", w)
print("Final bias (b):", b)
~~~
## Output:
<img width="878" height="572" alt="image" src="https://github.com/user-attachments/assets/a4b0b304-ed43-4f53-b425-122cd12da159" />
<img width="944" height="678" alt="image" src="https://github.com/user-attachments/assets/2ba333fd-598c-4d62-ab91-72c4350ef7cd" />







## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
