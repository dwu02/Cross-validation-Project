import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold

def pipeline(FILE_PATH, imputer_strategy = 'median',verbose = False):
    file = open(FILE_PATH,'r').read().splitlines()
    data = [line.split(',') for line in file]
    df = pd.DataFrame(data,columns = ['ID', 'Clump Thickness', 'Uniformity of Cell Size','Uniformity of Cell Shape','Marginal Adhesion','Single Epithelial Cell Size','Bare Nuclei','Bland Chromatin','Normal Nucleoli:','Mitoses','Class'])
    df.replace('?', np.nan, inplace=True)
    missing_val_count_by_col = (df.isnull().sum())
    if verbose: print(f"""
-------------------------------------
Dataset: Breast Cancer Wisconsin Data
State: Before Imputation
Imputation Strategy: {imputer_strategy}
Number of missing values: {sum(list(missing_val_count_by_col[missing_val_count_by_col > 0]))}
Columns with missing values: {missing_val_count_by_col[missing_val_count_by_col > 0].index}
-------------------------------------
    """)
    mean_imputer = SimpleImputer(strategy = imputer_strategy)
    imputed = pd.DataFrame(mean_imputer.fit_transform(df))
    imputed.columns = df.columns
    imputed['Class'] = imputed['Class'].astype(str)
    numeric_cols = [cname for cname in imputed.columns if imputed[cname].dtype in ['int64', 'float64']]
    missing_val_count_by_col = (imputed.isnull().sum())
    if verbose: print(f"""
-------------------------------------
Dataset: Breast Cancer Wisconsin Data
State: After Imputation
Imputation Strategy: {imputer_strategy}
Number of missing values: {sum(list(missing_val_count_by_col[missing_val_count_by_col > 0]))}
-------------------------------------
    """)
    X = imputed[numeric_cols].copy()
    y = imputed.Class
    X.drop(columns=['ID'],inplace=True)
    kf = KFold(n_splits=10, shuffle=True, random_state=1)
    index = 1
    if verbose: 
        for train_index, test_index in kf.split(X, y):
            print(f'Fold:{index}, Train set: {len(train_index)}, Test set:{len(test_index)}')
            index += 1
    return X,y,kf

# take ratio of score with computation time
    

# df.Class.replace('2','benign', inplace=True)
# df.Class.replace('4','malignant', inplace=True)

# # kf = KFold(n_splits=10, shuffle=True, random_state=42)
# # cnt = 1
# # for train_index, test_index in kf.split(X, y):
# #     print(f'Fold:{cnt}, Train set: {len(train_index)}, Test set:{len(test_index)}')
# #     cnt += 1

# imputed[['ID']] = imputed[['ID']].apply(pd.to_numeric)

# print(missing_val_count_by_col[missing_val_count_by_col > 0].index)
# tot_missing = sum(list(missing_val_count_by_col[missing_val_count_by_col > 0]))
# print(tot_missing)
# print(imputed['Bare Nuclei: 1 - 10'][145])