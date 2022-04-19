from moleculekit.smallmol.smallmollib import SmallMolLib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import numpy as np

def train_model(train_sdf, target_prop, val_set_fraction=0.1, scale=True):
    vectors = []
    labels = []
    for mol in SmallMolLib(train_sdf):
        metal = mol.strip_salts_and_detect_metals()
        
        if metal==0: #discard molecules with metals
            prop_names = [i for i in mol._mol.GetPropNames()]
            if target_prop in prop_names:
                headers, props  = mol.featurize(2, 1024)
                vectors.append(props)
                labels.append(float(mol._mol.GetProp(target_prop)))

    X = np.array(vectors) 
    y = np.array(labels)
    X = X[:, 1:] #discard smiles column
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=val_set_fraction)

    if scale:
        #should all features be scaled? even binary values?
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    model = XGBRegressor(n_estimators=1500, max_depth=5, n_jobs=-1, subsample=0.7, learning_rate=0.05, 
                        colsample_bytree=0.7, gamma=0.1, min_child_weight=1)
    model.fit(X_train, Y_train)

    y_predicted = model.predict(X_test)
    print('Corr. Coef for test set:', np.corrcoef(Y_test,y_predicted)[0][1])
    return vectors, labels

a, b = train_model(
    'some.sdf',
    'Kd'
    )