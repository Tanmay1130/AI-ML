# Titanic Dataset - Preprocessing & Cleaning

This repository contains the cleaned and preprocessed Titanic dataset and a preprocessing script.

## Files
- `Titanic-Dataset.csv` - Original dataset (uploaded by user).
- `titanic_cleaned.csv` - Cleaned and preprocessed dataset (generated).
- `preprocessing_script.py` - Python script that reproduces the preprocessing steps.
- `README.md` - This file.

## Preprocessing Steps Applied
1. **Imported dataset** and inspected missing values and data types.
2. **Dropped `Cabin`** column due to a high proportion of missing values.
3. **Imputed missing values**:
   - `Age` -> median.
   - `Embarked` -> mode.
   - `Fare` -> median.
4. **Extracted `Title`** from `Name` and grouped rare titles.
5. **Encoded categorical variables**:
   - `Sex` mapped to 0 (male) and 1 (female).
   - `Embarked` and `Title` one-hot encoded.
6. **Standardized numerical features**: Age, Fare, SibSp, Parch using `StandardScaler`.
7. **Outlier removal**: Removed rows where `Age` or `Fare` were outside 1.5 * IQR bounds (based on original values).
8. **Saved cleaned dataset** to `titanic_cleaned.csv`.

## How to reproduce
Run the `preprocessing_script.py` in a Python environment with pandas and scikit-learn installed:
```
python preprocessing_script.py
```

## Notes
- The outlier removal used Age and Fare IQR; adjust as needed.
- Cabin was dropped — if you prefer to keep it, consider imputing or extracting deck letter.

