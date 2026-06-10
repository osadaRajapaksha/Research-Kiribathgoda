import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline 

print("--- SCRIPT 3: SMOTE + TUNED RANDOM FOREST ---")
print("1. Loading dataset...")
# Make sure to use your correct dataset file name here!
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

print("3. Tuning SMOTE + Random Forest with GridSearch...")
pipeline = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('rf', RandomForestClassifier(class_weight='balanced', random_state=42))
])

param_grid = {
    'smote__sampling_strategy': [0.3, 0.5, 0.7, 'auto'], 
    'rf__n_estimators': [100, 200, 300],         
    'rf__max_depth': [3, 5, 7]                   
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print(f"Best Parameters: {grid_search.best_params_}")

print("\n4. Generating Predictions...")
train_pred = best_model.predict(X_train)
test_pred = best_model.predict(X_test)

print("\n=== METRICS ===")
print(f"Training Accuracy: {accuracy_score(y_train, train_pred) * 100:.2f}%")
print(f"Testing Accuracy:  {accuracy_score(y_test, test_pred) * 100:.2f}%")

print("\n5. Saving QGIS CSV (Unseen Test Data)...")
test_df = df.loc[X_test.index].copy()
test_df['Predicted_Hotspot'] = test_pred

def categorize_prediction(row):
    if row['Is_Hotspot'] == 1 and row['Predicted_Hotspot'] == 1: return 'True Hotspot'
    if row['Is_Hotspot'] == 0 and row['Predicted_Hotspot'] == 0: return 'True Safe'
    if row['Is_Hotspot'] == 0 and row['Predicted_Hotspot'] == 1: return 'False Alarm'
    if row['Is_Hotspot'] == 1 and row['Predicted_Hotspot'] == 0: return 'Missed Hotspot'

test_df['Prediction_Category'] = test_df.apply(categorize_prediction, axis=1)
test_df.to_csv('QGIS_Predictions_3_SMOTE_Tuned.csv', index=False)
print("Saved -> 'QGIS_Predictions_3_SMOTE_Tuned.csv'")

print("\n6. Saving Real-World Confusion Matrix Image...")
cm = confusion_matrix(y_test, test_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=["Predicted Safe", "Predicted Hotspot"], 
            yticklabels=["Actually Safe", "Actually Hotspot"])
plt.title('Real-World Test Data - Confusion Matrix')
plt.ylabel('True Reality (Map)')
plt.xlabel('Algorithm Prediction')
plt.tight_layout()
plt.savefig('Matrix_3_SMOTE_Tuned_RF.png', dpi=300)
print("Saved -> 'Matrix_3_SMOTE_Tuned_RF.png'")

# ==========================================
# --- NEW: APPLY SMOTE TO TESTING DATA ---
# ==========================================
print("\n7. Extracting and Saving SMOTE Synthetic TEST Data...")
best_smote_ratio = grid_search.best_params_['smote__sampling_strategy']

# Safely calculate k_neighbors based on how many hotspots ended up in the tiny test set
test_hotspot_count = y_test.sum()
safe_k_neighbors = min(5, max(1, test_hotspot_count - 1))

# Re-run SMOTE manually to get the synthetic rows on the TEST data
manual_smote_test = SMOTE(sampling_strategy=best_smote_ratio, random_state=42, k_neighbors=safe_k_neighbors)
X_test_resampled, y_test_resampled = manual_smote_test.fit_resample(X_test, y_test)

# Make predictions on this new mixed test dataset (Real Test + Fake SMOTE)
resampled_test_predictions = best_model.predict(X_test_resampled)

smote_test_export_df = X_test_resampled.copy()
smote_test_export_df['Is_Hotspot_Actual'] = y_test_resampled
smote_test_export_df['Predicted_Hotspot'] = resampled_test_predictions

# Tag which rows are real and which are synthetic
num_real_test_rows = len(X_test)
num_synthetic_test_rows = len(X_test_resampled) - num_real_test_rows
smote_test_export_df['Data_Type'] = ['Real'] * num_real_test_rows + ['Synthetic'] * num_synthetic_test_rows

smote_test_export_df.to_csv('SMOTE_Combined_Testing_Data.csv', index=False)
print(f"Saved -> 'SMOTE_Combined_Testing_Data.csv'")
print(f"         ({num_real_test_rows} Real rows and {num_synthetic_test_rows} Synthetic rows)")

print("\n8. Saving SMOTE TESTING Confusion Matrix Image...")
cm_smote_test = confusion_matrix(y_test_resampled, resampled_test_predictions)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_smote_test, annot=True, fmt='d', cmap='Greens', 
            xticklabels=["Predicted Safe", "Predicted Hotspot"], 
            yticklabels=["Actually Safe", "Actually Hotspot"])
plt.title('Testing Data (with SMOTE) - Confusion Matrix')
plt.ylabel('SMOTE Test Reality (Real + Fake)')
plt.xlabel('Algorithm Prediction')
plt.tight_layout()
plt.savefig('Matrix_4_SMOTE_Testing_Data.png', dpi=300)
print("Saved -> 'Matrix_4_SMOTE_Testing_Data.png'")
print("\nSuccess! All files generated perfectly.")