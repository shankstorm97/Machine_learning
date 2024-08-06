from data.prepare_data import prepare_data
from models.train_lda import train_lda
from models.train_svm import train_svm
from utils.evaluate import evaluate_model

def main():
    # Prepare the data
    X_train, X_test, y_train, y_test = prepare_data()

    # Train Fisher's Linear Discriminant
    lda, training_time_lda = train_lda(X_train, y_train)
    conf_matrix_lda, class_report_lda, testing_time_lda = evaluate_model(lda, X_test, y_test)

    # Train Linear Support Vector Machine
    svm, training_time_svm = train_svm(X_train, y_train)
    conf_matrix_svm, class_report_svm, testing_time_svm = evaluate_model(svm, X_test, y_test)

    # Print results
    print(f'Training time (LDA): {training_time_lda:.4f} seconds')
    print(f'Testing time (LDA): {testing_time_lda:.4f} seconds')
    print('Confusion Matrix (LDA):')
    print(conf_matrix_lda)
    print('Classification Report (LDA):')
    print(class_report_lda)

    print(f'Training time (SVM): {training_time_svm:.4f} seconds')
    print(f'Testing time (SVM): {testing_time_svm:.4f} seconds')
    print('Confusion Matrix (SVM):')
    print(conf_matrix_svm)
    print('Classification Report (SVM):')
    print(class_report_svm)

if __name__ == "__main__":
    main()
