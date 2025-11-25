"""
Preprocessing script for Titanic dataset.
Saves a cleaned CSV to titanic_cleaned.csv
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("Titanic-Dataset.csv")
# Drop Cabin due to many missing values
if 'Cabin' in df.columns:
    df = df.drop(columns=['Cabin'])
# Fill missing
if 'Age' in df.columns:
    df['Age'] = df['Age'].fillna(df['Age'].median())
if 'Embarked' in df.columns:
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
if 'Fare' in df.columns:
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())

# Feature engineering
def extract_title(name):
    if pd.isnull(name):
        return "Unknown"
    parts = name.split(',')
    if len(parts) > 1:
        title_part = parts[1].strip().split(' ')[0]
        return title_part.replace('.', '')
    return "Unknown"

if 'Name' in df.columns:
    df['Title'] = df['Name'].apply(extract_title)
    title_counts = df['Title'].value_counts()
    rare_titles = title_counts[title_counts < 10].index.tolist()
    df['Title'] = df['Title'].apply(lambda t: 'Rare' if t in rare_titles else t)

# Categorical encoding
if 'Sex' in df.columns:
    df['Sex'] = df['Sex'].map({'male':0, 'female':1})
if 'Embarked' in df.columns:
    df = pd.concat([df, pd.get_dummies(df['Embarked'], prefix='Embarked')], axis=1)
    df = df.drop(columns=['Embarked'])
if 'Title' in df.columns:
    df = pd.concat([df, pd.get_dummies(df['Title'], prefix='Title')], axis=1)
    df = df.drop(columns=['Title'])

# Scaling numerical
num_cols = [c for c in ['Age','Fare','SibSp','Parch'] if c in df.columns]
if num_cols:
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

# Remove outliers using IQR on original values
orig = pd.read_csv("Titanic-Dataset.csv")
to_remove = pd.Series(False, index=df.index)
for col in num_cols:
    Q1 = orig[col].quantile(0.25)
    Q3 = orig[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5*IQR
    upper = Q3 + 1.5*IQR
    mask = (orig[col] < lower) | (orig[col] > upper)
    to_remove = to_remove | mask

df_clean = df.loc[~to_remove].reset_index(drop=True)
df_clean.to_csv("titanic_cleaned.csv", index=False)
print("Saved titanic_cleaned.csv")
