from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
import joblib

digits = datasets.load_digits()

X_train, X_test, y_train, y_test = train_test_split(
    digits.data, 
    digits.target, 
    test_size=0.2, 
    random_state=42
)

clf = svm.SVC(gamma=0.001)

clf.fit(X_train, y_train)

joblib.dump(clf, 'digits_svm_model.pkl')

# Evaluate on the test set
predicted = clf.predict(X_test)


import matplotlib.pyplot as plt
images = X_test.reshape(-1, 8, 8)
for index, (image, prediction, label) in enumerate(zip(images, predicted, y_test)):
    if index == 10:  # Show 10 examples
        break
    plt.subplot(2, 5, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title(f'Pred: {prediction}\nTrue: {label}')
plt.show()