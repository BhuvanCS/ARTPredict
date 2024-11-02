import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# Load the preprocessed dataset
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df


def evaluate_models(X, y):
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'Support Vector Machine': SVC(),
        'Gradient Boosting': GradientBoostingClassifier(),
        'MLP': MLPClassifier(max_iter=1000),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    }
    
    results = {}
    
    for name, model in models.items():
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        results[name] = cv_scores.mean()
    
    nn_accuracy = evaluate_neural_network(X, y)
    results['Neural Network'] = nn_accuracy
    
    return results

def evaluate_neural_network(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Build the Neural Network model
    model = Sequential()
    model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Assuming binary classification (0/1)

    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    # Set up early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=32, callbacks=[early_stopping])

    _, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Neural Network model accuracy: {accuracy:.4f}")

    return accuracy

def main(file_path):
    df = load_data(file_path)

    df.dropna(inplace=True)

    X = df.iloc[:, 1:-1]  
    y = df.iloc[:, -1]    # Target variable: last column only

    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    
    results = evaluate_models(X, y)
    
    
    for model_name, accuracy in results.items():
        print(f"{model_name}: {accuracy:.4f}")

    
    best_model = max(results, key=results.get)
    print(f"\nBest model: {best_model} with an accuracy of {results[best_model]:.4f}")

if __name__ == "__main__":
    file_path = 'app\data\preprocessed_data\preprocessed_clinical_data.csv'  # Replace with your CSV file path
    main(file_path)
