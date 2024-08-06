import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def prepare_data():
    # Load your dataset
    df = pd.read_csv('data/MathE_dataset.csv', delimiter=';')
    
    # Print the first few rows of the dataframe
    print(df.head())
    
    # Define the feature columns (excluding non-numeric and non-relevant columns)
    feature_columns = ['Student Country', 'Question ID', 'Question Level', 'Topic', 'Subtopic']
    
    # Extract features and target
    X = df[feature_columns]
    y = df['Type of Answer']
    
    # Convert categorical features to numeric
    X = pd.get_dummies(X)
    
    # Encode target variable if it's categorical
    y = LabelEncoder().fit_transform(y)
    
    # Print the shapes of the arrays to confirm
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    # Split the data into training (750+) and testing (250+) sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Normalize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test
