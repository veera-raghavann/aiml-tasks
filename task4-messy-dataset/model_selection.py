"""Second-year extension: fair model comparison with cross-validation."""

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

DATA_URL = "https://raw.githubusercontent.com/MLSA-SRM/recruit-task-messy-dataset/main/recruitment_engagement.csv"


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    clean = df.drop_duplicates().copy()

    clean["domain"] = clean["domain"].astype("string").str.strip().str.lower().str.title()
    clean["subdomain"] = clean["subdomain"].astype("string").str.strip()
    clean["subdomain"] = clean["subdomain"].replace({"ai/ml": "AI/ML", "web dev": "Web Dev"})
    clean["subdomain"] = clean["subdomain"].str.title().replace({"Ai/Ml": "AI/ML", "Pr": "PR"})
    clean["year"] = clean["year"].astype("string").str.strip().str.lower().map({
        "first": "1st Year", "1": "1st Year", "1st year": "1st Year",
        "second": "2nd Year", "2": "2nd Year", "2nd year": "2nd Year",
    })
    clean["signup_source"] = clean["signup_source"].astype("string").str.strip().str.title()
    clean["prior_experience"] = clean["prior_experience"].astype("string").str.strip().str.lower().map({
        "yes": "Yes", "y": "Yes", "no": "No", "n": "No"
    })

    numeric_cols = ["prep_hours_last_week", "quiz_score", "days_since_signup"]
    for col in numeric_cols:
        clean[col] = pd.to_numeric(clean[col], errors="coerce")

    clean.loc[clean["prep_hours_last_week"] < 0, "prep_hours_last_week"] = np.nan
    clean.loc[clean["days_since_signup"] < 0, "days_since_signup"] = np.nan
    clean.loc[~clean["quiz_score"].between(0, 100), "quiz_score"] = np.nan
    clean["completed_task"] = clean["completed_task"].astype("string").str.strip().str.title()
    return clean


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical = X.select_dtypes(include=["object", "string"]).columns.tolist()
    numeric = X.select_dtypes(include=["number"]).columns.tolist()

    return ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]), numeric),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
    ])


df = pd.read_csv(DATA_URL)
clean = clean_data(df)
X = clean.drop(columns=["completed_task", "applicant_id", "name"])
y = clean["completed_task"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

models = {
    "Logistic Regression": Pipeline([
        ("preprocess", make_preprocessor(X_train)),
        ("model", LogisticRegression(max_iter=2000, random_state=42)),
    ]),
    "Random Forest": Pipeline([
        ("preprocess", make_preprocessor(X_train)),
        ("model", RandomForestClassifier(
            n_estimators=200, random_state=42, class_weight="balanced"
        )),
    ]),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = []

for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")
    cv_results.append({
        "Model": name,
        "CV F1 mean": scores.mean(),
        "CV F1 std": scores.std(),
    })

cv_table = pd.DataFrame(cv_results).sort_values("CV F1 mean", ascending=False)
print("5-fold training CV:")
print(cv_table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

selected_name = cv_table.iloc[0]["Model"]
selected_model = models[selected_name]
selected_model.fit(X_train, y_train)
pred = selected_model.predict(X_test)

print(f"\nSelected for final test inspection: {selected_name}")
print(f"Accuracy : {accuracy_score(y_test, pred):.3f}")
print(f"Precision: {precision_score(y_test, pred, pos_label='Yes'):.3f}")
print(f"Recall   : {recall_score(y_test, pred, pos_label='Yes'):.3f}")
print(f"F1       : {f1_score(y_test, pred, pos_label='Yes'):.3f}")
print("\nClassification report:")
print(classification_report(y_test, pred))

errors = X_test.copy()
errors["actual"] = y_test
errors["predicted"] = pred
errors = errors[errors["actual"] != errors["predicted"]]
print(f"Misclassified test rows: {len(errors)}")
print(errors.to_string())
