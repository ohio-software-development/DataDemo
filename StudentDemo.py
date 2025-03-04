import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error

df = pd.read_csv('StudentsPerformance.csv')

label_encoders = {}
categorical_features = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']
for col in categorical_features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le


X = df.drop(columns=['math score', 'reading score', 'writing score'])
y_math = df['math score']
y_reading = df['reading score']
y_writing = df['writing score']

X_train, X_test, y_math_train, y_math_test = train_test_split(X, y_math, test_size=0.2, random_state=42)
X_train, X_test, y_reading_train, y_reading_test = train_test_split(X, y_reading, test_size=0.2, random_state=42)
X_train, X_test, y_writing_train, y_writing_test = train_test_split(X, y_writing, test_size=0.2, random_state=42)

models = {}
for subject, y_train in zip(['math', 'reading', 'writing'], [y_math_train, y_reading_train, y_writing_train]):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    models[subject] = model

for subject, model in models.items():
    feature_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print(f'Feature importance for {subject} score:\n{feature_importance}\n')

def predict_scores(user_input):
    user_df = pd.DataFrame([user_input])
    for col, le in label_encoders.items():
        user_df[col] = le.transform(user_df[col])
    predictions = {subject: model.predict(user_df)[0] for subject, model in models.items()}
    return predictions

user_input = {
    'gender': 'girl',
    'race/ethnicity': 'group C',
    'parental level of education': "master's degree",
    'lunch': 'standard',
    'test preparation course': 'none'
}

print("My socioeconomic factors have predispositioned myself to probably score like this: ")
print(predict_scores(user_input))
print(" ")

df['gender'] = label_encoders['gender'].inverse_transform(df['gender'])
gender_avg_scores = df.groupby('gender')[['math score', 'reading score', 'writing score']].mean()
print("Average scores by gender:\n", gender_avg_scores)

df['parental level of education'] = label_encoders['parental level of education'].inverse_transform(df['parental level of education'])
parental_avg_scores = df.groupby('parental level of education')[['math score', 'reading score', 'writing score']].mean()
print("Average scores by parental level of education:\n", parental_avg_scores)

df['race/ethnicity'] = label_encoders['race/ethnicity'].inverse_transform(df['race/ethnicity'])
parental_avg_scores = df.groupby('race/ethnicity')[['math score', 'reading score', 'writing score']].mean()
print("Average scores by race/ethnicity:\n", parental_avg_scores)

