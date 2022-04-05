
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error
import eli5

def parse_price(val):
    return float(val.replace(" ", "").replace(",", "."))

def get_df(df_train, df_test):    
    ### łączy dane treningowe i testowe, rozpakowuje słownik z danymi kategorycznymi
    
    df_train = df_train[ df_train.index != 106447 ].reset_index(drop=True)  # usuwamy jeden outlier
    df = pd.concat([df_train, df_test])
 
    params = df["offer_params"].apply(pd.Series)
    params = params.fillna(-1)

    df = pd.concat([df, params], axis=1)
    print(df.shape)

    # faktoryzujemy dane kategoryczne
    obj_feats = params.select_dtypes(object).columns

    for feat in obj_feats:
        df["{}_cat".format(feat)] = df[feat].factorize()[0]
            
    return df

def check_model(df, feats, model, cv=5, scoring="neg_mean_absolute_error", show_eli5=True):
    ### dzieli dane na treningowe i testowe, przeprowadza cross validation 
    
    # podział danych
    df_train = df[ ~df["price_value"].isnull() ].copy()
    df_test = df[ df["price_value"].isnull() ].copy()

    X_train = df_train[feats]
    y_train = df_train["price_value"]
    
    # testy
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring)
    result = np.mean(scores), np.std(scores)
    
    if show_eli5: # pokaż dane do debugowania modelu
        model.fit(X_train, y_train)
        print(result)
        return eli5.show_weights(model, feature_names=feats)
    
    return result

def check_log_model(df, feats, model, cv=5, scoring=mean_absolute_error, show_eli5=True):
    ### dzieli dane na treningowe i testowe, przeprowadza cross validation dla cen zlogarytmizowanych
    
    df_train = df[ ~df["price_value"].isnull() ].copy()

    X = df_train[feats]
    y = df_train["price_value"]
    y_log = np.log(y)  # zlogarytmizowanie wartości cen
    
    cv = KFold(n_splits=cv, shuffle=True, random_state=0) # podział na grupy (zwraca indeksy)
    scores = []
    for train_idx, test_idx in cv.split(X):
        # wybór danych treningowych i testowych
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]  
        y_log_train, y_test = y_log.iloc[train_idx], y.iloc[test_idx]

        # trenowanie modelu
        model.fit(X_train, y_log_train)
        y_log_pred = model.predict(X_test)
        y_pred = np.exp(y_log_pred) # powrót do zwykłych cen

        score = scoring(y_test, y_pred)  # domyślnie mean absolute error
        scores.append(score)
        
    result = np.mean(scores), np.std(scores)
    
    if show_eli5:
        model.fit(X, y_log)
        print(result)
        return eli5.show_weights(model, feature_names=feats)

    return result

def reset_outliers(df, feat, prc=99):
    cut_value = np.percentile(df[feat], prc) # taką lub mniejszą pojemność ma 99% samochodów
    
    return df[feat].map(lambda x: x if x < cut_value else -1) # kasujemy tę wartość (outlier)
