import logging

logger = logging.getLogger(__name__)


def train_model(
    train_sdf, target_prop, val_set_fraction=0.1, fp_radius=2, fp_size=1024, scale=True
):
    """
    Trains an xgbregressor model for all the molecules inside `train_sdf` using as label the SDF property specified in
    `target_prop`. The molecules get automatically featurized as a combination of RDKIT computed properties, such as
    number of rings, or number of acceptors, and a combination of 3 different fingerprints (Avalon, MACCSkeys and Morgan).

    Parameters
    ----------
    train_sdf: str
        path to an SDF file with molecules, must contain the target label you want to train your model on
    target_prop: str
        The name of the SDF property field to use as label, like "IC50" or "Activity"
    val_set_fraction: float
        Fraction of the molecules inside the SDF to use for model validation.
    fp_radius: int
        radius of the morgan fingerprint
    fp_size: int
        the size of the fingerprint for the Avalon and Morgan fingerprints.
    scale: bool
        wheter to use some form of feature scaling or not

    Returns
    -------
    the XGBRegressor model trained
    """
    from moleculekit.smallmol.smallmollib import SmallMolLib
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from xgboost import XGBRegressor
    import numpy as np
    import pandas as pd

    vectors = []
    labels = []
    smiles = []
    for mol in SmallMolLib(train_sdf):
        metal = mol.strip_salts_and_detect_metals()

        if metal:  # discard molecules with metals
            prop_names = [i for i in mol._mol.GetPropNames()]
            if target_prop in prop_names:
                headers, props = mol.featurize(fp_radius, fp_size)
                smiles.append(props[0])
                vectors.append(props[1:])
                labels.append(float(mol._mol.GetProp(target_prop)))

    smiles = np.array(smiles)
    vectors = np.array(vectors)
    labels = np.array(labels)[:, np.newaxis]
    values = np.append(vectors, labels, axis=1)
    headers = headers[1:]
    headers.append("label")

    df = pd.DataFrame(values, columns=headers)
    # remove +inf, -inf and nans
    df = df.replace([np.inf, -np.inf], np.nan)  # .dropna()

    y = df.iloc[:, -1]
    X = df.iloc[:, :-1]

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, y, test_size=val_set_fraction
    )

    if scale:
        logger.info("Scaling the computed features...")
        # should all features be scaled? even binary values?
        # probally the function should return the scaler for inferences
        scaler = StandardScaler()
        X_train[X_train.columns] = scaler.fit_transform(X_train[X_train.columns])

        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    logger.info("Training the XGB regressor...")
    model = XGBRegressor(
        n_estimators=1500,
        max_depth=5,
        n_jobs=-1,
        subsample=0.7,
        learning_rate=0.05,
        colsample_bytree=0.7,
        gamma=0.1,
        min_child_weight=1,
    )
    model.fit(X_train, Y_train)

    y_predicted = model.predict(X_test)
    print("Corr. Coef for test set:", np.corrcoef(Y_test, y_predicted)[0][1])
    return model


trained_model = train_model("Data_SDF.sdf", "logP")

