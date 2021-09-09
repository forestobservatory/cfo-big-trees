import multiprocessing as _mp

from sklearn import model_selection as _model_selection

_ncores = _mp.cpu_count()


class _Types:
    def __init__(self):
        from numbers import Number
        from typing import Dict, List, Union

        from numpy.typing import ArrayLike
        from sklearn.model_selection import BaseCrossValidator, ParameterGrid
        from sklearn.model_selection._search import BaseSearchCV

        self.ArrayLike = ArrayLike
        self.BaseCrossValidator = BaseCrossValidator
        self.BaseSearchCV = BaseSearchCV
        self.ErrorScore = Union[str, Number]
        self.ScorerList = Union[List, Dict]
        self.Number = Number
        self.ParameterGrid = Union[Dict, ParameterGrid]


types = _Types()


def list_scorers(method: str = None) -> types.ScorerList:
    """Returns a list of viable scorers for grid search optimization.

    Args:
        method: Modeling method. Options: ["classification", "regression", "clustering"].
            If `None`, returns a dictionary with all available scorers.

    Returns:
        a list of all scorers by method.
    """
    classification_scorers = [
        "accuracy",
        "average_precision",
        "f1",
        "f1_micro",
        "f1_macro",
        "f1_weighted",
        "f1_samples",
        "neg_log_loss",
        "precision",
        "recall",
        "roc_auc",
    ]
    regression_scorers = [
        "neg_mean_absolute_error",
        "neg_mean_squared_error",
        "neg_median_absolute_error",
        "r2",
    ]
    clustering_scorers = [
        "adjusted_rand_score",
    ]

    if method.lower() == "classification":
        return classification_scorers
    elif method.lower() == "regression":
        return regression_scorers
    elif method.lower() == "clustering":
        return clustering_scorers
    else:
        all_scorers = {
            "classification": classification_scorers,
            "regression": regression_scorers,
            "clustering": clustering_scorers,
        }
        return all_scorers


# a class for model tuning that has the default parameters pre-set
class Tuner(object):
    def __init__(
        self,
        x: types.ArrayLike,
        y: types.ArrayLike,
        optimizer: types.BaseSearchCV = _model_selection.GridSearchCV,
        param_grid: types.ParameterGrid = None,
        scoring: str = None,
        fit_params: dict = None,
        n_jobs: int = _ncores - 1,
        refit: bool = True,
        verbose: int = 1,
        error_score: types.ErrorScore = 0,
        return_train_score: bool = False,
        cv: types.BaseCrossValidator = None,
        n_splits: int = 5,
    ):
        """Performs hyperparameter searches to optimize model performance using `sklearn`.

        Args:
            x : {array-like, sparse matrix} of shape (n_samples, n_features)
                The input samples. Internally, it will be converted to
                ``dtype=np.float32`` and if a sparse matrix is provided
                to a sparse ``csr_matrix``.
            y : array-like of shape (n_samples,)
                Target values (strings or integers in classification, real numbers
                in regression)
                For classification, labels must correspond to classes.
            optimizer:
            param_grid:
            scoring: Performance metric to optimize in model training.
                Get available options with `list_scorers()`
            fit_params: Parameters passed to the fit method of the estimator.
            n_jobs: Number of simultaneous grid search processes to run.
            refit: Whether to fit a new model each time a model class is instantiated.
            error_score: Value to assign to the score if an error occurs in estimator fitting.
                If set to ‘raise’, the error is raised. If a numeric value is given, FitFailedWarning is raised.
            return_train_score: Get insights on how different parameter settings impact overfitting/underfitting.
                But computing scores can be expensive and is not required to select parameters that yield
                the best generalization performance.
            cv:
            n_splits:

        Returns:
            a Tuner object that can perform hyperparameter searches and optimize model performance.
        """

        self.x = x
        self.y = y
        self.optimizer = optimizer
        self.param_grid = param_grid
        self.scoring = scoring
        self.fit_params = fit_params
        self.n_jobs = n_jobs
        self.refit = refit
        self.verbose = verbose
        self.error_score = error_score
        self.return_train_score = return_train_score
        self.cv = cv
        self.n_splits = n_splits

    # function to actually run the tuning process and report outputs
    def _run_grid_search(self, estimator):
        """Helper function to run the grid search, which is called by each of the model clases."""

        gs = self.optimizer(
            estimator,
            param_grid=self.param_grid,
            scoring=self.scoring,
            fit_params=self.fit_params,
            n_jobs=self.n_jobs,
            cv=self.cv,
            refit=self.refit,
            verbose=self.verbose,
            error_score=self.error_score,
            return_train_score=self.return_train_score,
        )

        gs.fit(self.x, self.y)

        self.cv_results = gs.cv_results_
        self.best_estimator = gs.best_estimator_
        self.best_score = gs.best_score_
        self.best_params = gs.best_params_
        self.best_index = gs.best_index_
        self.scorer = gs.scorer_
        self.n_splits = gs.n_splits_
        self.gs = gs

    def LinearRegression(self, optimizer=None, param_grid=None, scoring=None, fit_params=None, cv=None):
        from sklearn import linear_model, metrics, model_selection

        # check if the optimizer has changed, otherwise use default
        if optimizer is not None:
            self.optimizer = optimizer

        # check if the parameter grid has been set, otherwise set defaults
        if param_grid is None:
            if self.param_grid is None:
                param_grid = {"normalize": (True, False), "fit_intercept": (True, False)}
                self.param_grid = param_grid
        else:
            self.param_grid = param_grid

        # set the scoring function
        if scoring is None:
            if self.scoring is None:
                scoring = metrics.explained_variance_score
                self.scoring = scoring
        else:
            self.scoring = scoring

        # set the default fit parameters
        if fit_params is not None:
            self.fit_params = fit_params

        # set the cross validation strategy
        if cv is None:
            if self.cv is None:
                cv = model_selection.StratifiedKFold(n_splits=self.n_splits)
                self.cv = cv
        else:
            self.cv = cv

        # create the estimator and run the grid search
        estimator = linear_model.LinearRegression()
        self._run_grid_search(estimator)

    def LogisticRegression(self, optimizer=None, param_grid=None, scoring=None, fit_params=None, cv=None):
        from sklearn import linear_model, model_selection

        # check if the optimizer has changed, otherwise use default
        if optimizer is not None:
            self.optimizer = optimizer

        # check if the parameter grid has been set, otherwise set defaults
        if param_grid is None:
            if self.param_grid is None:
                param_grid = {"C": (1e-2, 1e-1, 1e0, 1e1), "tol": (1e-3, 1e-4, 1e-5), "fit_intercept": (True, False)}
                self.param_grid = param_grid
        else:
            self.param_grid = param_grid

        # set the scoring function
        if scoring is None:
            if self.scoring is None:
                scoring = "roc_auc"
                self.scoring = scoring
        else:
            self.scoring = scoring

        # set the default fit parameters
        if fit_params is not None:
            self.fit_params = fit_params

        # set the cross validation strategy
        if cv is None:
            if self.cv is None:
                cv = model_selection.StratifiedKFold(n_splits=self.n_splits)
                self.cv = cv
        else:
            self.cv = cv

        # create the estimator and run the grid search
        estimator = linear_model.LogisticRegression()
        self._run_grid_search(estimator)

    def DecisionTreeClassifier(self, optimizer=None, param_grid=None, scoring=None, fit_params=None, cv=None):
        from sklearn import model_selection, tree

        # check if the optimizer has changed, otherwise use default
        if optimizer is not None:
            self.optimizer = optimizer

        # check if the parameter grid has been set, otherwise set defaults
        if param_grid is None:
            if self.param_grid is None:
                param_grid = {
                    "criterion": ("gini", "entropy"),
                    "splitter": ("best", "random"),
                    "max_features": ("sqrt", "log2", None),
                    "max_depth": (2, 5, 10, None),
                    "min_samples_split": (2, 0.01, 0.1),
                    "min_impurity_split": (1e-7, 1e-6),
                }
                self.param_grid = param_grid
        else:
            self.param_grid = param_grid

        # set the scoring function
        if scoring is None:
            if self.scoring is None:
                scoring = "roc_auc"
                self.scoring = scoring
        else:
            self.scoring = scoring

        # set the default fit parameters
        if fit_params is not None:
            self.fit_params = fit_params

        # set the cross validation strategy
        if cv is None:
            if self.cv is None:
                cv = model_selection.StratifiedKFold(n_splits=self.n_splits)
                self.cv = cv
        else:
            self.cv = cv

        # create the estimator and run the grid search
        estimator = tree.DecisionTreeClassifier()
        self._run_grid_search(estimator)

    def SVC(self, optimizer=None, param_grid=None, scoring=None, fit_params=None, cv=None, class_weight=None):
        from sklearn import model_selection, svm

        # check if the optimizer has changed, otherwise use default
        if optimizer is not None:
            self.optimizer = optimizer

        # check if the parameter grid has been set, otherwise set defaults
        if param_grid is None:
            if self.param_grid is None:
                param_grid = {
                    "C": (1e-3, 1e-2, 1e-1, 1e0, 1e1),
                    "kernel": ("rbf", "linear"),
                    "gamma": (1e-3, 1e-4, 1e-5, 1e-6, 1e-7),
                }
        self.param_grid = param_grid

        # set the scoring function
        if scoring is None:
            if self.scoring is None:
                scoring = "roc_auc"
        self.scoring = scoring

        # set the default fit parameters
        if fit_params is not None:
            self.fit_params = fit_params

        # set the cross validation strategy
        if cv is None:
            if self.cv is None:
                cv = model_selection.StratifiedKFold(n_splits=self.n_splits)
        self.cv = cv

        # set the class weight
        if class_weight is not None:
            if self.class_weight is None:
                class_weight = "balanced"
        self.param_grid["class_weight"] = class_weight

        # create the estimator and run the grid search
        estimator = svm.SVC()
        self._run_grid_search(estimator)

    def SVR(self, optimizer=None, param_grid=None, scoring=None, fit_params=None, cv=None):
        from sklearn import metrics, model_selection, svm

        # check if the optimizer has changed, otherwise use default
        if optimizer is not None:
            self.optimizer = optimizer

        # check if the parameter grid has been set, otherwise set defaults
        if param_grid is None:
            if self.param_grid is None:
                param_grid = {
                    "C": (1e-2, 1e-1, 1e0, 1e1),
                    "epsilon": (0.01, 0.1, 1),
                    "kernel": ("rbf", "linear", "poly", "sigmoid"),
                    "gamma": (1e-2, 1e-3, 1e-4),
                }
                self.param_grid = param_grid
        else:
            self.param_grid = param_grid

        # set the scoring function
        if scoring is None:
            if self.scoring is None:
                scoring = metrics.explained_variance_score
                self.scoring = scoring
        else:
            self.scoring = scoring

        # set the default fit parameters
        if fit_params is not None:
            self.fit_params = fit_params

        # set the cross validation strategy
        if cv is None:
            if self.cv is None:
                cv = model_selection.StratifiedKFold(n_splits=self.n_splits)
                self.cv = cv
        else:
            self.cv = cv

        # create the estimator and run the grid search
        estimator = svm.SVR()
        self._run_grid_search(estimator)

    def LinearSVC(self, optimizer=None, param_grid=None, scoring=None, fit_params=None, cv=None):
        from sklearn import model_selection, svm

        # check if the optimizer has changed, otherwise use default
        if optimizer is not None:
            self.optimizer = optimizer

        # check if the parameter grid has been set, otherwise set defaults
        if param_grid is None:
            if self.param_grid is None:
                param_grid = {
                    "C": (1e-2, 1e-1, 1e0, 1e1),
                    "loss": ("hinge", "squared_hinge"),
                    "tol": (1e-3, 1e-4, 1e-5),
                    "fit_intercept": (True, False),
                }
                self.param_grid = param_grid
        else:
            self.param_grid = param_grid

        # set the scoring function
        if scoring is None:
            if self.scoring is None:
                scoring = "roc_auc"
                self.scoring = scoring
        else:
            self.scoring = scoring

        # set the default fit parameters
        if fit_params is not None:
            self.fit_params = fit_params

        # set the cross validation strategy
        if cv is None:
            if self.cv is None:
                cv = model_selection.StratifiedKFold(n_splits=self.n_splits)
                self.cv = cv
        else:
            self.cv = cv

        # create the estimator and run the grid search
        estimator = svm.LinearSVC()
        self._run_grid_search(estimator)

    def LinearSVR(self, optimizer=None, param_grid=None, scoring=None, fit_params=None, cv=None):
        from sklearn import metrics, model_selection, svm

        # check if the optimizer has changed, otherwise use default
        if optimizer is not None:
            self.optimizer = optimizer

        # check if the parameter grid has been set, otherwise set defaults
        if param_grid is None:
            if self.param_grid is None:
                param_grid = {
                    "C": (1e-2, 1e-1, 1e0, 1e1),
                    "loss": ("epsilon_insensitive", "squared_epsilon_insensitive"),
                    "epsilon": (0, 0.01, 0.1),
                    "dual": (False),
                    "tol": (1e-3, 1e-4, 1e-5),
                    "fit_intercept": (True, False),
                }
                self.param_grid = param_grid
        else:
            self.param_grid = param_grid

        # set the scoring function
        if scoring is None:
            if self.scoring is None:
                scoring = metrics.explained_variance_score
                self.scoring = scoring
        else:
            self.scoring = scoring

        # set the default fit parameters
        if fit_params is not None:
            self.fit_params = fit_params

        # set the cross validation strategy
        if cv is None:
            if self.cv is None:
                cv = model_selection.StratifiedKFold(n_splits=self.n_splits)
                self.cv = cv
        else:
            self.cv = cv

        # create the estimator and run the grid search
        estimator = svm.LinearSVR()
        self._run_grid_search(estimator)

    def AdaBoostClassifier(self, optimizer=None, param_grid=None, scoring=None, fit_params=None, cv=None):
        from sklearn import ensemble, model_selection

        # check if the optimizer has changed, otherwise use default
        if optimizer is not None:
            self.optimizer = optimizer

        # check if the parameter grid has been set, otherwise set defaults
        if param_grid is None:
            if self.param_grid is None:
                param_grid = {"n_estimators": (25, 50, 75, 100), "learning_rate": (0.1, 0.5, 1.0)}
                self.param_grid = param_grid
        else:
            self.param_grid = param_grid

        # set the scoring function
        if scoring is None:
            if self.scoring is None:
                scoring = "roc_auc"
                self.scoring = scoring
        else:
            self.scoring = scoring

        # set the default fit parameters
        if fit_params is not None:
            self.fit_params = fit_params

        # set the cross validation strategy
        if cv is None:
            if self.cv is None:
                cv = model_selection.StratifiedKFold(n_splits=self.n_splits)
                self.cv = cv
        else:
            self.cv = cv

        # create the estimator and run the grid search
        estimator = ensemble.AdaBoostClassifier()
        self._run_grid_search(estimator)

    def AdaBoostRegressor(self, optimizer=None, param_grid=None, scoring=None, fit_params=None, cv=None):
        from sklearn import ensemble, metrics, model_selection

        # check if the optimizer has changed, otherwise use default
        if optimizer is not None:
            self.optimizer = optimizer

        # check if the parameter grid has been set, otherwise set defaults
        if param_grid is None:
            if self.param_grid is None:
                param_grid = {
                    "n_estimators": (25, 50, 75, 100),
                    "learning_rate": (0.1, 0.5, 1.0),
                    "loss": ("linear", "exponential", "square"),
                }
                self.param_grid = param_grid
        else:
            self.param_grid = param_grid

        # set the scoring function
        if scoring is None:
            if self.scoring is None:
                scoring = metrics.explained_variance_score
                self.scoring = scoring
        else:
            self.scoring = scoring

        # set the default fit parameters
        if fit_params is not None:
            self.fit_params = fit_params

        # set the cross validation strategy
        if cv is None:
            if self.cv is None:
                cv = model_selection.StratifiedKFold(n_splits=self.n_splits)
                self.cv = cv
        else:
            self.cv = cv

        # create the estimator and run the grid search
        estimator = ensemble.AdaBoostRegressor()
        self._run_grid_search(estimator)

    def GradientBoostClassifier(self, optimizer=None, param_grid=None, scoring=None, fit_params=None, cv=None):
        from sklearn import ensemble, model_selection

        # check if the optimizer has changed, otherwise use default
        if optimizer is not None:
            self.optimizer = optimizer

        # check if the parameter grid has been set, otherwise set defaults
        if param_grid is None:
            if self.param_grid is None:
                param_grid = {
                    "n_estimators": (10, 100, 500),
                    "learning_rate": (0.01, 0.1, 0.5),
                    "max_features": ("sqrt", "log2", None),
                    "max_depth": (1, 10, None),
                    "min_samples_split": (2, 0.01, 0.1),
                    "min_impurity_split": (1e-7, 1e-6),
                }
                self.param_grid = param_grid
        else:
            self.param_grid = param_grid

        # set the scoring function
        if scoring is None:
            if self.scoring is None:
                scoring = "roc_auc"
                self.scoring = scoring
        else:
            self.scoring = scoring

        # set the default fit parameters
        if fit_params is not None:
            self.fit_params = fit_params

        # set the cross validation strategy
        if cv is None:
            if self.cv is None:
                cv = model_selection.StratifiedKFold(n_splits=self.n_splits)
                self.cv = cv
        else:
            self.cv = cv

        # create the estimator and run the grid search
        estimator = ensemble.GradientBoostingClassifier()
        self._run_grid_search(estimator)

    def GradientBoostRegressor(self, optimizer=None, param_grid=None, scoring=None, fit_params=None, cv=None):
        from sklearn import ensemble, model_selection

        # check if the optimizer has changed, otherwise use default
        if optimizer is not None:
            self.optimizer = optimizer

        # check if the parameter grid has been set, otherwise set defaults
        if param_grid is None:
            if self.param_grid is None:
                param_grid = {
                    "n_estimators": (10, 100, 500),
                    "learning_rate": (0.01, 0.1, 0.5),
                    "max_features": ("sqrt", "log2", None),
                    "max_depth": (1, 10, None),
                    "min_samples_split": (2, 0.01, 0.1),
                    "min_impurity_split": (1e-7, 1e-6),
                }
                self.param_grid = param_grid
        else:
            self.param_grid = param_grid

        # set the scoring function
        if scoring is None:
            if self.scoring is None:
                scoring = "neg_mean_absolute_error"
                self.scoring = scoring
        else:
            self.scoring = scoring

        # set the default fit parameters
        if fit_params is not None:
            self.fit_params = fit_params

        # set the cross validation strategy
        if cv is None:
            if self.cv is None:
                cv = model_selection.StratifiedKFold(n_splits=self.n_splits)
                self.cv = cv
        else:
            self.cv = cv

        # create the estimator and run the grid search
        estimator = ensemble.GradientBoostingRegressor()
        self._run_grid_search(estimator)

    def RandomForestClassifier(
        self, optimizer=None, param_grid=None, scoring=None, fit_params=None, cv=None, class_weight=None
    ):
        from sklearn import ensemble, model_selection

        # check if the optimizer has changed, otherwise use default
        if optimizer is not None:
            self.optimizer = optimizer

        # check if the parameter grid has been set, otherwise set defaults
        if param_grid is None:
            if self.param_grid is None:
                param_grid = {
                    "criterion": ("gini", "entropy"),
                    "n_estimators": (10, 100, 500),
                    "max_features": ("sqrt", "log2", None),
                    "max_depth": (1, 10, None),
                    "min_samples_split": (2, 0.01, 0.1),
                    "min_impurity_split": (1e-7, 1e-6),
                }
                self.param_grid = param_grid
        else:
            self.param_grid = param_grid

        # set the scoring function
        if scoring is None:
            if self.scoring is None:
                scoring = "roc_auc"
                self.scoring = scoring
        else:
            self.scoring = scoring

        # set the default fit parameters
        if fit_params is not None:
            self.fit_params = fit_params

        # set the cross validation strategy
        if cv is None:
            if self.cv is None:
                cv = model_selection.StratifiedKFold(n_splits=self.n_splits)
                self.cv = cv
        else:
            self.cv = cv

        # create the estimator and run the grid search
        estimator = ensemble.RandomForestClassifier()
        self._run_grid_search(estimator)

    def RandomForestRegressor(self, optimizer=None, param_grid=None, scoring=None, fit_params=None, cv=None):
        from sklearn import ensemble, model_selection

        # check if the optimizer has changed, otherwise use default
        if optimizer is not None:
            self.optimizer = optimizer

        # check if the parameter grid has been set, otherwise set defaults
        if param_grid is None:
            if self.param_grid is None:
                param_grid = {
                    "n_estimators": (10, 100, 500),
                    "max_features": ("sqrt", "log2", None),
                    "max_depth": (1, 10, None),
                    "min_samples_split": (2, 0.01, 0.1),
                    "min_impurity_split": (1e-7, 1e-6),
                }
                self.param_grid = param_grid
        else:
            self.param_grid = param_grid

        # set the scoring function
        if scoring is None:
            if self.scoring is None:
                scoring = "neg_mean_absolute_error"
                self.scoring = scoring
        else:
            self.scoring = scoring

        # set the default fit parameters
        if fit_params is not None:
            self.fit_params = fit_params

        # set the cross validation strategy
        if cv is None:
            if self.cv is None:
                cv = model_selection.StratifiedKFold(n_splits=self.n_splits)
                self.cv = cv
        else:
            self.cv = cv

        # create the estimator and run the grid search
        estimator = ensemble.RandomForestRegressor()
        self._run_grid_search(estimator)
