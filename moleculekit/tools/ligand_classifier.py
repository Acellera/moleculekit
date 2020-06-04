import argparse
import os
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import average_precision_score
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc,recall_score,precision_score
from ax.service.managed_loop import optimize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import StratifiedKFold
import logging
import pickle
import time


class LigandBinderClassifier:
    def __init__(self, args):

        for arg, value in args.items():
            if value == '':
                args[arg] = None

        logger = logging.getLogger('LigandBinderClassifier')
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        #ch = logging.StreamHandler()
        ch = logging.FileHandler('LigandBasedClassifier.log')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        self.logger = logger

        self.vars = argparse.Namespace(**args)

        if self.vars.predict and not self.vars.model:
            raise Exception("For prediction mode, please provide an XGBoost model using the --model option")

        self.default_hp_params = [
               {"name": "max_depth", "type": "range", "bounds": [2, 12]},
               {"name": "learning_rate", "type": "range", "bounds": [0.0001, 0.03]},
               {"name": "subsample", "type": "range", "bounds": [0.1, 1.0]},
               {"name": "colsample_bytree", "type": "range", "bounds": [0.1, 1.0]},
               {"name": "reg_alpha", "type": "range", "bounds": [0.0, 15.0]},
               {"name": "reg_lambda", "type": "range", "bounds": [0.0, 15.0]}
        ]

    def _load_data(self, csv):
        df = pd.read_csv(csv, header=None)
        if len(df.columns) == 1:
            df.columns = ['smiles']
        elif len(df.columns) == 2:
            df.columns = ["smiles", "truth"]
        return df

    def _compute_fingerprints(self, data):
        fingerprints = []
        self.logger.info('Generating ligand fingerprints...')
        for smile in tqdm(data.smiles.values):
            try:
                m = Chem.MolFromSmiles(smile)
                fingerprints.append(np.array(AllChem.GetMorganFingerprintAsBitVect(m, 2)))
            except:
                fingerprints.append(np.nan)
        data["fingerprint"] = fingerprints
        data = data.dropna()
        return data

    def _generate_input_labels(self, df):
        if len(df.columns) > 1:
            y = df.truth.values
        else:
            y = np.nan
        X = np.array([np.array(x) for x in df.fingerprint.values])
        return X, y

    def _run_averaging_ensemble(self, models, X_test):
        preds = []
        models = {k: v for k, v in sorted(models.items(), key=lambda item: item[0], reverse=True)}
        for model in list(models.values())[:]:
            preds_submodels = []
            for submod in model:
                pr = submod.predict_proba(X_test)
                preds_submodels.append(pr)
            preds.append(np.mean(preds_submodels, axis=0))
        preds = np.mean(preds, axis=0)
        return preds

    def _calculate_enrichment(self, prob, truth, quantile=0.1):
        top_ligs = truth[np.argsort(prob[:, 1])[::-1]][:int(quantile * len(truth))]
        if len(top_ligs) == 0:
            return np.nan
        a = len(top_ligs[top_ligs == 1])
        b = len(top_ligs)
        self.logger.info("Test top-{}% enrichment:\t {}\t ({} out of {} were positives)".format( round(quantile*100,2), round(a / b, 4), a, b))
        return a / b

    def _calculate_auc(self, y_test, preds):
        fpr, tpr, _ = roc_curve(y_test, preds[:, 1])
        roc_auc = auc(fpr, tpr)
        self.logger.info("AUC: {}".format(round(roc_auc,4)))
        return roc_auc

    def _run_training(self, X, y):
        self.logger.info("Running training...")

        def _hp_evaluate(parameters):
            split = StratifiedKFold(10)
            res = []
            ms = []
            for train_idx, val_idx in split.split(X, y):
                X_train_ = X[train_idx]
                X_val_ = X[val_idx]
                y_train_ = y[train_idx]
                y_val_ = y[val_idx]
                clf = XGBClassifier(learning_rate=parameters['learning_rate'],
                                    max_depth=int(parameters['max_depth']),
                                    n_estimators=10000,
                                    subsample=parameters['subsample'],
                                    colsample_bytree=parameters['colsample_bytree'],
                                    reg_alpha=parameters['reg_alpha'],
                                    reg_lambda=parameters['reg_lambda'],
                                    n_jobs=-1)
                clf.fit(X_train_, y_train_, early_stopping_rounds=20, verbose=False, eval_set=[(X_val_, y_val_)],
                        eval_metric=self.vars.optimization_metric)
                res.append(clf.best_score)

                ms.append(clf)

            self.logger.info("Optimization metric ({}) in validation set: {}".format(self.vars.optimization_metric, round(np.mean(res), 4)))
            opt_models[np.mean(res)] = ms
            return abs(np.mean(res))

        self.logger.info("Running hyperparameter tuning on 10-fold cross-validation: maximizing {}...".format(self.vars.optimization_metric))
        opt_models = {}
        best_parameters, values, experiment, model = optimize(
            parameters=self.default_hp_params,
            evaluation_function=_hp_evaluate,
            objective_name='map',
            total_trials=self.vars.hp_trials
        )
        output_model_name = 'model_{}.pickle'.format('fragalysis')
        self.logger.info("Saving pre-trained model at: {}".format(output_model_name))
        pickle.dump(opt_models, open(output_model_name, "wb"))

    def _run_predict(self, X, data):
        opt_models = pickle.load(open(self.vars.model.name, "rb"))
        preds = self._run_averaging_ensemble(opt_models, X)
        output_df = pd.DataFrame(data.smiles)
        output_df['predictions'] = preds[:, 1]
        output_name = "predictions_fragalysis.csv"
        self.logger.info("Writing predictions to {}".format(output_name))
        output_df.to_csv(output_name, index=False, header=False)

    def _run_validation(self, X, y):
        self.logger.info("Running validation: splitting data into training and test sets...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.vars.seed)
        self.logger.info("Proportion of binders in test set: {}".format(round(len(y_test[y_test == 1]) / len(y_test),2)))
        self.logger.info("Proportion of binders in training set: {}".format(round(len(y_train[y_train == 1]) / len(y_train), 2)))
        self.logger.info("Overall proportion of binders in the whole data: {}".format(round(len(y[y == 1]) / len(y), 2)))

        def _hp_evaluate(parameters):
            split = StratifiedKFold(10)
            res = []
            ms = []
            for train_idx, val_idx in split.split(X_train, y_train):
                X_train_ = X_train[train_idx]
                X_val_ = X_train[val_idx]
                y_train_ = y_train[train_idx]
                y_val_ = y_train[val_idx]
                clf = XGBClassifier(learning_rate=parameters['learning_rate'],
                                    max_depth=int(parameters['max_depth']),
                                    n_estimators=10000,
                                    subsample=parameters['subsample'],
                                    colsample_bytree=parameters['colsample_bytree'],
                                    reg_alpha=parameters['reg_alpha'],
                                    reg_lambda=parameters['reg_lambda'],
                                    n_jobs=-1)
                clf.fit(X_train_, y_train_, early_stopping_rounds=20, verbose=False, eval_set=[(X_val_, y_val_)],
                        eval_metric=self.vars.optimization_metric)
                res.append(clf.best_score)

                ms.append(clf)

            self.logger.info("Optimization metric ({}) in validation set: {}".format(self.vars.optimization_metric, round(np.mean(res), 4)))
            opt_models[np.mean(res)] = ms
            return abs(np.mean(res))

        self.logger.info("Running hyperparameter tuning on 10-fold cross-validation: maximizing {}...".format(self.vars.optimization_metric))
        opt_models = {}
        best_parameters, values, experiment, model = optimize(
            parameters=self.default_hp_params,
            evaluation_function=_hp_evaluate,
            objective_name='map',
            total_trials=self.vars.hp_trials
        )
        self.logger.info("Validating model performance in the test set...")
        preds = self._run_averaging_ensemble(opt_models, X_test)
        enrichments = []
        for quantile in [0.01, 0.05]+list(np.arange(0.1, 1.1, 0.1)):
            enrichments.append(self._calculate_enrichment(preds, y_test, quantile))
        auc = self._calculate_auc(y_test, preds)
        output_model_name = 'model_{}.pickle'.format('fragalysis')
        self.logger.info("Saving pre-trained model at: {}".format(output_model_name))
        pickle.dump(opt_models, open(output_model_name, "wb"))


    def run(self):
        data = self._load_data(self.vars.csv)
        data = self._compute_fingerprints(data)
        X, y = self._generate_input_labels(data)

        if self.vars.validate is True:
            self._run_validation(X, y)
        elif self.vars.train is True:
            self._run_training(X, y)
        elif self.vars.predict is True:
            self._run_predict(X, data)

