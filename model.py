import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# 1. Automatically find the dataset path
base_path = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_path, '..', 'dataset', 'student_data.csv')

try:
    df = pd.read_csv(csv_path)
    print("✅ Dataset loaded successfully!")

    # 2. Binary target: 1 for Dropout, 0 for others
    df['Target_Binary'] = df['Target'].apply(lambda x: 1 if x == 'Dropout' else 0)

    # 3. Top 6 Features from your CSV
    features = [
        'Curricular units 2nd sem (approved)',
        'Curricular units 2nd sem (grade)',
        'Curricular units 1st sem (approved)',
        'Curricular units 1st sem (grade)',
        'Tuition fees up to date',
        'Age at enrollment'
    ]

    X = df[features]
    y = df['Target_Binary']

    # 4. Scale and Train
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    # 5. Save the 'Brain' files into the backend folder
    with open(os.path.join(base_path, 'model.pkl'), 'wb') as f:
        pickle.dump(model, f)
    with open(os.path.join(base_path, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    print(f"✅ Success! Model trained. Accuracy: {round(model.score(X_test, y_test)*100, 2)}%")

except FileNotFoundError:
    print(f"❌ Error: Could not find student_data.csv at {csv_path}. Check your folder names!")
except Exception as e:
    print(f"❌ An error occurred: {e}")