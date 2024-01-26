#pip install prince

import prince
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier


# takes input - X(feature vector), y(target variable) 
# prints significant columns with(p-value<0.05)
# outputs new dataframe with only significant features
# remove print statements when not needed
def get_p_values_significant_features(X,y):
    X = sm.add_constant(X)
    ols_model = sm.OLS(y, X).fit()

    p_features = {}
    for i in X.columns.tolist():
        p_features[f'{i}'] = ols_model.pvalues[i]

    data_pvalue= pd.DataFrame(p_features.items(), columns=['Feature_name', 'p-Value']).sort_values(by = 'p-Value').reset_index(drop=True)
    print('*'*50)
    print('significant features')
    print('*'*50)
    print(data_pvalue[data_pvalue['p-Value']<0.05])
    significant_features = data_pvalue[data_pvalue['p-Value']<0.05]['Feature_name'].tolist()

    return X[significant_features]

# usage of get_p_values_significant_features
#X_significant = get_p_values_significant_features(X,y)
#print(X_significant)

# takes input - X(feature vector), y(target variable), no_of_features(to be selected)
# prints selected columns
# outputs new dataframe with selected features
# remove print statements when not needed
def select_k_best(X,y,no_of_features):
    k_selector = SelectKBest(mutual_info_classif, k=no_of_features)
    X_new = k_selector.fit_transform(X, y)
    chosen_indices = k_selector.get_support(indices=True)
    column_names = X.columns
    print('*'*50)
    print('selected features using mutual_info_classif')
    print('*'*50)
    selected_column_names = column_names[chosen_indices]
    print(selected_column_names)
  
    return X[selected_column_names]

# usage of select_k_best
# X_select = select_k_best(X, y, 40)
# print(X_select)


# this is very similar to PCA but works for both categorical and numerical variables
# input X(feature vector), n(reduced no. of features)
# prints the eigen value, % variance and % cumulative variance
# output tranformed feature vector with reduced number of components
# https://pypi.org/project/prince/0.6.2/#factor-analysis-of-mixed-data-famd
# remove print statements when not needed
def famd_dim_reduction(X, n):
    famd = prince.FAMD(
    n_components=n,
    n_iter=5,
    copy=True,
    check_input=True,
    random_state=42,
    engine="sklearn",
    handle_unknown="error" 
    )
    famd = famd.fit(X)
    print(famd.eigenvalues_summary)
    X_famd = famd.transform(X)

    return X_famd

# usage of famd_dim_reduction
# X_famd = famd_dim_reduction(X, 40)

# recursive feature elimination
# input X(feature vector), y(target), no_of_features(reduced no. of features)
# output feature vector with reduced number of features
def dim_red_using_rfe(X, y, no_of_features):
    model = RandomForestClassifier()
    reduced_features_estimator = RFE(estimator=model, n_features_to_select=no_of_features)
    reduced_features_estimator.fit(X, y)
    reduced_col_names = reduced_features_estimator.get_feature_names_out()

    return X[reduced_col_names]

# usage of dim_red_using_rfe
#X_rfe = dim_red_using_rfe(X, y, 30)

