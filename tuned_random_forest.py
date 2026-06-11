import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

os.chdir(os.path.dirname(os.path.abspath(__file__)))
print("--- SCRIPT 2: TUNED RANDOM FOREST ---")
print("1. Loading dataset...")
df = pd.read_csv('fixed_segments_with_ml_features.csv')

features_to_keep = [
    'Road_Type', 'Sinuosity_Index', 'Dist_to_Intersection_m', 'No_of_Lanes', 
    'Speed_Limit_kmh', 'Has_Zebra_Crossing', 'Has_Streetlight_Infrastructure', 
    'Baseline_Traffic_Volume', 'Elevation_Gradient_Pct', 'Topography'
]

X = pd.get_dummies(df[features_to_keep], drop_first=True)
y = df['Is_Hotspot']

print("2. Splitting Data (80% Train, 20% Test)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("3. Tuning Random Forest with GridSearch...")
rf = RandomForestClassifier(class_weight='balanced', random_state=42)
param_grid = {'n_estimators': [100, 200, 300], 'max_depth': [3, 5, 7]}

grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='f1', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_
print(f"Best Parameters: {grid_search.best_params_}")

print("4. Generating Predictions...")
train_pred = best_rf.predict(X_train)
test_pred = best_rf.predict(X_test)

print("\n=== METRICS ===")
print(f"Training Accuracy: {accuracy_score(y_train, train_pred) * 100:.2f}%")
print(f"Testing Accuracy:  {accuracy_score(y_test, test_pred) * 100:.2f}%")

print("\n5. Saving QGIS CSV...")
test_df = df.loc[X_test.index].copy()
test_df['Predicted_Hotspot'] = test_pred
test_df.to_csv('QGIS_Predictions_2_Tuned.csv', index=False)
print("Saved -> 'QGIS_Predictions_2_Tuned.csv'")

print("\n6. Saving Confusion Matrix Image...")
cm = confusion_matrix(y_test, test_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', 
            xticklabels=["Predicted Safe", "Predicted Hotspot"], 
            yticklabels=["Actually Safe", "Actually Hotspot"])
plt.title('Tuned Random Forest - Confusion Matrix')
plt.ylabel('True Reality (Map)')
plt.xlabel('Algorithm Prediction')
plt.tight_layout()
plt.savefig('Matrix_2_Tuned_RF.png', dpi=300)
print("Saved -> 'Matrix_2_Tuned_RF.png'")