from sklearn.svm import SVC
import time

def train_svm(X_train, y_train):
    svm = SVC(kernel='linear')
    start_time = time.time()
    svm.fit(X_train, y_train)
    training_time = time.time() - start_time
    return svm, training_time
