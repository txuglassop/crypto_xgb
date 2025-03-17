# CrytoXGB

*A workflow to develop, train and backtests XGBoost models to predict jumps in crypto prices! Supports custom feature engineering, evaluation metrics, trading strategies, and more!*

> [!INFO]
> Check out `notebooks/wf.ipynb` for a demonstration!

**CryptoXGB is not intended to be used for financial decision making.**

## Getting started

Fork this repo, then run
```sh
pip install -r requirements.txt
```

Then, replace the empty strings in `user_info.py` with your timezone and Binance API details.

To scrape data using Binance's API, use
```sh
./get_data.sh <symbol> <interval> <start_year> [month] [day]
```
and the data will be stored in `input/`.

## Feature Engineering

To features to the XGBoost model, first add a function to `src/feature_engineering.py` that will create a new column in the dataframe containing this feature. It can use other features already in the `df`, and preferably returns `df` with the new column added.

Then, call the function in an appropriate spot in `add_features` in `add_features.py` to actually add the feature to your model.

## Finding optimal hyperparameters

CryptoXGB uses stepwise optimisation with the `Optuna` library for hyperparameter optimisation. The code for this can be found in `src/optimise_xgb.py`.

To achieve good results, a lot of computation is required, so cloud computing is not a bad idea. `colab.py` is an alternative to `wf.ipynb` that has some Google Colab features ready to go, as well as GPU acceleration to speed up this process.

You can also specify a custom evaluation metric as the objective function for XGBoost's training algorithm. See `eval_metrics.py` for more.

Then, to initiate the optimisation, run
```python
from optimise_xgb import stepwise_optimisation
from eval_metrics import <metric>

db_url = f'db/<name_of_sqlite_db>'

params = stepwise_optimisation(X_train, y_train, num_classes, eval_metric, db_url, n_jobs=1, trials=500)
```


## Backtesting

Before we backtest our model, you can make your custom trading strategy in `strategies.py`. See this file for more info.

To run a backtest, create an instance of the `Backtest` class from `backtest.py` and initialise training and test sets as well as the corresponding timestamps. For example,
```python
from backtest import backtest

backtest = Backtest(
    model_trainer, X_train, y_train, X_test, y_test, train_timestamp, test_timestamp, num_classes,
    starting_capital=500_000, commission=0.0002
)
```

Then, to actually run the backtest, use the `.run_backtest` method,
```python
bt_results = backtest.run_backtest(strategy, retrain=True, train_frequency=24*7, progress_bar=True)
```


You can then analyse the backtest by plotting it or printing some common metrics using the functions from `backtests_metrics.py`,
```python
from backtest_metrics import plot_backtest, print_backtest_metrics

plot_backtest(bt_results)

print_backtest_metrics(bt_results)
```


## Logging model hyperparameters, features and backtest results

To keep track of models you've made as well as past backtests, you can use the `Sesssion` object from `session_info.py` to log model specifics and backtests results in `backtest/`. See `wf.ipynb` for an example of how to do this.

















