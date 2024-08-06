import time
from sklearn.metrics import confusion_matrix, classification_report

def evaluate_model(model, X_test, y_test):
    start_time = time.time()
    y_pred = model.predict(X_test)
    testing_time = time.time() - start_time

    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    return conf_matrix, class_report, testing_time
