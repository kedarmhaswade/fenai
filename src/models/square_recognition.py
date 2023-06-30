import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split


features_file_name = "../../data/processed/chessboard_squares.txt"
targets_file_name = "../../data/processed/chessboard_square_classes.txt"
data_x = np.loadtxt(features_file_name, dtype='int', delimiter=' ')
print(data_x.shape)
data_y = np.loadtxt(targets_file_name, dtype='str')
print(data_y.shape)
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.3, shuffle=False)
clf = svm.SVC(kernel='linear', gamma=0.001)
clf.fit(x_train, y_train)
predicted = clf.predict(x_test)
print(
    f"Classification report for classifier {clf}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n"
)

disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")
plt.show()
