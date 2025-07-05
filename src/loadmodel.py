import joblib
import matplotlib.pyplot as plt
from sklearn import datasets

# Load the trained model
clf = joblib.load('digits_svm_model.pkl')
print("Model loaded successfully!")

# Load some sample digits
digits = datasets.load_digits()

# Show the first 10 images
plt.figure(figsize=(8, 2))
for i in range(10):
    plt.subplot(1, 10, i + 1)
    plt.imshow(digits.images[i], cmap='gray')
    plt.axis('off')
    plt.title(str(digits.target[i]))
plt.suptitle("Sample Digits from Dataset")
plt.show()