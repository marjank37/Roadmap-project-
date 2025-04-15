# Data Cleaning & Preparation for Classification

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Load the Excel file
df = pd.read_excel(r"D:\roadmap\Data_Pathrise.xlsx", sheet_name='data')

# Drop irrelevant columns
df = df.drop(columns=['id', 'cohort_tag'])

# Remove any trailing spaces in column names
df.columns = df.columns.str.strip()

# Drop rows with missing target
df = df.dropna(subset=['placed'])

# Fill missing categorical values with 'Unknown'
categorical_cols = [
    'gender', 'work_authorization_status', 'employment_status',
    'professional_experience', 'length_of_job_search',
    'highest_level_of_education', 'biggest_challenge_in_search', 'race'
]

for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].fillna('Unknown')

# Fill missing numeric columns
if 'number_of_interviews' in df.columns:
    df['number_of_interviews'] = df['number_of_interviews'].fillna(df['number_of_interviews'].median())

if 'program_duration_days' in df.columns:
    df['program_duration_days'] = df['program_duration_days'].fillna(df['program_duration_days'].median())

if 'number_of_applications' in df.columns:
    df['number_of_applications'] = df['number_of_applications'].replace(0, 1)

# One-hot encode categorical variables
df_encoded = pd.get_dummies(df, drop_first=True)

# Fill any remaining NaN values with 0
df_encoded = df_encoded.fillna(0)

# Save cleaned version (optional)
df_encoded.to_csv("cleaned_classification_data.csv", index=False)
print("✅ Data cleaned and ready for classification.")

# ------------------------------
# Classification
# ------------------------------

# Load cleaned data
df = pd.read_csv("cleaned_classification_data.csv")

# Define features and target
X = df.drop('placed', axis=1)
y = df['placed']

# Ensure no NaNs remain
X = X.fillna(0)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define classification models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(),
    "Support Vector Machine (SVM)": SVC(),
    "K-Nearest Neighbors (KNN)": KNeighborsClassifier()
}

# Train and evaluate each model
for name, model in models.items():
    print(f"\n🔍 Evaluating {name}...")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

# Set default style
sns.set(style="whitegrid")

# 1. Correlation Heatmap for numerical columns
plt.figure(figsize=(12, 8))
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# 2. Distribution of program duration (target for regression)
plt.figure(figsize=(10, 6))
sns.histplot(df['program_duration_days'], bins=30, kde=True)
plt.title("Distribution of Program Duration (Days)")
plt.xlabel("Program Duration")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# 3. Gender vs Placed
plt.figure(figsize=(8, 5))
sns.countplot(x="gender_Male", hue="placed", data=df)
plt.title("Gender vs Placement")
plt.xlabel("Is Male")
plt.ylabel("Count")
plt.legend(title="Placed")
plt.tight_layout()
plt.show()

# 4. Race vs Placed
# NOTE: Race is one-hot encoded — this just shows one example
race_cols = [col for col in df.columns if col.startswith("race_")]
for col in race_cols:
    plt.figure(figsize=(8, 4))
    sns.countplot(x=col, hue="placed", data=df)
    plt.title(f"{col.replace('race_', '')} vs Placement")
    plt.xlabel(col)
    plt.tight_layout()
    plt.show()

# 5. Primary Track
track_cols = [col for col in df.columns if col.startswith("primary_track_")]
for col in track_cols:
    plt.figure(figsize=(8, 4))
    sns.countplot(x=col, hue="placed", data=df)
    plt.title(f"{col.replace('primary_track_', '')} Track vs Placement")
    plt.xlabel(col)
    plt.tight_layout()
    plt.show()

# 6. Number of Applications vs Placement
plt.figure(figsize=(10, 6))
sns.boxplot(x="placed", y="number_of_applications", data=df)
plt.title("Applications vs Placement")
plt.tight_layout()
plt.show()

# 7. Number of Interviews vs Placement
plt.figure(figsize=(10, 6))
sns.boxplot(x="placed", y="number_of_interviews", data=df)
plt.title("Interviews vs Placement")
plt.tight_layout()
plt.show()

# 8. Highest Level of Education vs Placement
edu_cols = [col for col in df.columns if col.startswith("highest_level_of_education_")]
for col in edu_cols:
    plt.figure(figsize=(8, 4))
    sns.countplot(x=col, hue="placed", data=df)
    plt.title(f"Education: {col.replace('highest_level_of_education_', '')} vs Placement")
    plt.tight_layout()
    plt.show()


# Select numeric columns for correlation
numeric_cols = ['program_duration_days', 'number_of_applications', 'number_of_interviews', 'placed']
df_corr = df[numeric_cols]

# Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df_corr.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Heatmap of Correlation Between 4 Columns and placed")
plt.tight_layout()
plt.show()

# Pairplot
sns.pairplot(df_corr)
plt.suptitle("Pairplot of Numeric Features", y=1.02)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.kdeplot(df[df['placed'] == 0]['program_duration_days'], label='placed 0', color='blue', fill=True)
sns.kdeplot(df[df['placed'] == 1]['program_duration_days'], label='placed 1', color='red', fill=True)

# Mark golden time points (optional)
plt.axvline(x=60, color='yellow', lw=2, linestyle='--')
plt.axvline(x=380, color='yellow', lw=2, linestyle='--')

plt.title("Density Plot of program_duration_days Based on placed")
plt.xlabel("program duration days")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.show()


# Load the original file (not one-hot encoded for better labeling)
original_df = pd.read_excel(r"D:\roadmap\Data_Pathrise.xlsx", sheet_name='data')
original_df = original_df.dropna(subset=['placed'])
original_df['employment_status '] = original_df['employment_status '].fillna('Unknown')

plt.figure(figsize=(10, 6))
sns.violinplot(x='employment_status ', y='placed', data=original_df)
plt.title("Violin Plot of Placed by Employment Status")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()


models = {
    "LogR": LogisticRegression(max_iter=1000),
    "SVM": SVC(probability=True),
    "DT": RandomForestClassifier(max_depth=5),
    "KNN": KNeighborsClassifier(),
    "RF": RandomForestClassifier(),
}

X = df.drop("placed", axis=1)
y = df["placed"]

# Fill any missing values
X = X.fillna(0)

# Split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Plot ROC
plt.figure(figsize=(10, 6))
auc_scores = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    auc_scores[name] = roc_auc
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.legend()
plt.tight_layout()
plt.show()

# AUC Bar Chart
plt.figure(figsize=(8, 5))
sorted_auc = dict(sorted(auc_scores.items(), key=lambda item: item[1]))
sns.barplot(x=list(sorted_auc.keys()), y=list(sorted_auc.values()), color="green")
plt.ylabel("AUC Values")
plt.title("AUC Values for Different Models (Sorted)")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

df.to_csv("filename.csv", index=False)



