import pandas as pd
import xgboost as xgb
import optuna
from optuna import create_study, logging
from optuna.pruners import MedianPruner
from optuna.integration import XGBoostPruningCallback
from optuna.samplers import TPESampler

import warnings
warnings.filterwarnings('ignore')

optuna.logging.set_verbosity(optuna.logging.WARNING) 

# number of jobs for optimisation
N_JOBS = 1

# The following Optuna Optimisation call is heavily insprired by JP (@para24) on Kaggle, see:
# https://www.kaggle.com/code/para24/xgboost-stepwise-tuning-using-optuna/notebook#7.-Stepwise-Hyperparameter-Tuning

def _objective(trial, X, y, num_classes, group, score, params=dict()):
    """
    `X` and `y` MUST be pd.DataFrames - NOT `xgb.DMatrix`.  
    """
    dtrain = xgb.DMatrix(X, label=y)


    if group == '1':
        params['max_depth'] = trial.suggest_int('max_depth', 2, 30)
        params['min_child_weight'] = trial.suggest_loguniform('min_child_weight', 1e-10, 1e10)
    
    if group == '2':
        params['subsample'] = trial.suggest_uniform('subsample', 0, 1)
        params['colsample_bytree'] = trial.suggest_uniform('colsample_bytree', 0, 1)

    if group == '3':
        params['num_boost_round'] = trial.suggest_int('num_boost_round', 100, 600)
        params['learning_rate'] = trial.suggest_uniform('learning_rate', 0.005, 0.1)

    if group == '4':
        params['gamma'] = trial.suggest_loguniform('gamma', 1e-3, 19)

    pruning_callback = XGBoostPruningCallback(trial, 'test-' + score.__name__)

    xgb_params = params.copy()
    del xgb_params['num_boost_round']

    cv_scores = xgb.cv(xgb_params, dtrain, nfold=5,
                       stratified=True,
                       feval=score,
                       num_boost_round=params['num_boost_round'],
                       early_stopping_rounds=10,
                       callbacks=[pruning_callback])
    
    return cv_scores['test-' + score.__name__ + '-mean'].values[-1]

def _execute_optimisation(X_train, y_train, num_classes, study_name, group, score, trials, data:str, params=dict(), direction='maximize',):
    ## use pruner to skip trials that aren't doing so well
    pruner = MedianPruner(n_warmup_steps=20)

    ## use sampler to use past results from db
    sampler = TPESampler(n_startup_trials=20, multivariate=True, warn_independent_sampling=False)

    study = create_study(
        direction=direction,
        study_name=study_name,
        storage=f'sqlite:///db/optuna_{data}.db',
        load_if_exists=True,
        pruner=pruner,
        sampler=sampler
    )

    study.optimize(
        lambda trial: _objective(trial, X_train, y_train, num_classes, group, score, params),
        n_trials=trials,
        n_jobs=N_JOBS
    )

    print('STUDY NAME: ', study_name)
    print('-------------------------------------------------------')
    print('EVALUATION METRIC: ', score.__name__)
    print('-------------------------------------------------------')
    print('BEST CV SCORE: ', study.best_value)
    print('-------------------------------------------------------')
    print(f'OPTIMAL GROUP - {group} PARAMS: ', study.best_params)
    print('-------------------------------------------------------')
    print('BEST TRIAL', study.best_trial)
    print('-------------------------------------------------------')

    updated_params = params.copy()
    updated_params.update(study.best_params)

    return updated_params

def stepwise_optimisation(X_train: pd.DataFrame, y_train: pd.DataFrame, num_classes: int, eval_metric: callable, data: str, trials=9) -> dict:
    """
    Execute stepwise optimisation to find optimal CV parameters for XGBoost given the train set. 

    params:
        `X_train` (pd.DataFrame) - training set - NOT A `xgb.DMatrix` object

        `y_train` (pd.DataFrame) - as above

        `num_classes` (int) - the number of classes used to categorise returns

        `eval_metric` (callable) - evaluation metric to be used in optimisation. See `eval_metrics.py`

        `trials` - number of trials to do for `optimize`

    returns:
        a dictionary containing optimal parameters
    """
    final_params = dict()

    # initial learning params
    final_params['num_boost_round'] = 200
    final_params['learning_rate'] = 0.01
    final_params['objective'] = 'multi:softprob'
    final_params['num_class'] = num_classes

    # use gpu
    final_params['tree_method'] = 'hist'
    final_params['device'] = 'cuda'


    for g in ['1', '2', '3', '4']:
        print(f'====== Optimising Group {g} ======')
        update_params = _execute_optimisation(
            X_train, y_train, num_classes, 'xgboost', g, eval_metric, trials, data, params=final_params, direction='maximize',
        )
        final_params.update(update_params)
        print(f'Params after updating group {g}: ', final_params)
        print('\n\n')

    print(f'====== Final Optimal Parameters ======')
    print(final_params)

    return final_params
