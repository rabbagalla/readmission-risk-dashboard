# preprocessing.py

import pandas as pd
import numpy as np
import re

def clean_column_names(columns):
    return [re.sub(r'[\[\]<>]', '', col) for col in columns]

def preprocess_input(df):
    df.replace('?', np.nan, inplace=True)
    df['race'] = df['race'].fillna(df['race'].mode()[0])
    df.dropna(subset=['diag_1', 'diag_2', 'diag_3'], inplace=True)
    df[['max_glu_serum', 'A1Cresult', 'medical_specialty']] = df[[
        'max_glu_serum', 'A1Cresult', 'medical_specialty'
    ]].fillna('Unknown')

    def map_diag(code):
        if pd.isnull(code): return 0
        code = str(code)
        if code.startswith('V') or code.startswith('E'): return 0
        try: code = float(code)
        except: return 0
        if 390 <= code <= 459 or code == 785: return 1
        elif 460 <= code <= 519 or code == 786: return 2
        elif 520 <= code <= 579 or code == 787: return 3
        elif 250 <= code < 251: return 4
        elif 800 <= code <= 999: return 5
        elif 710 <= code <= 739: return 6
        elif 580 <= code <= 629 or code == 788: return 7
        elif 140 <= code <= 239: return 8
        else: return 9

    for col in ['diag_1', 'diag_2', 'diag_3']:
        df[col] = df[col].apply(map_diag)

    binary_cols = ['gender', 'change', 'diabetesMed']
    for col in binary_cols:
        df[col] = pd.factorize(df[col])[0]

    med_columns = [
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
        'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
        'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
        'insulin', 'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone',
        'metformin-pioglitazone', 'metformin-rosiglitazone'
    ]
    med_map = {'No': 0, 'Steady': 1, 'Up': 2, 'Down': 3}
    for col in med_columns:
        if col in df.columns:
            df[col] = df[col].map(med_map)

    onehot_cols = [
        'race', 'age', 'admission_type_id', 'discharge_disposition_id',
        'admission_source_id', 'max_glu_serum', 'A1Cresult', 'medical_specialty'
    ]
    df = pd.get_dummies(df, columns=onehot_cols, drop_first=True)

    return df
