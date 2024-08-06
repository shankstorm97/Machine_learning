from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import time

def train_lda(X_train, y_train):
    lda = LinearDiscriminantAnalysis()
    start_time = time.time()
    lda.fit(X_train, y_train)
    training_time = time.time() - start_time
    return lda, training_time
