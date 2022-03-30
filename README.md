# gridsearchcv
class MyGridSearchCV:
    
    def __init__(self, estimator, param_grid: dict, scoring=None, refit=True, cv=None):
        self.estimator = estimator
        self.param_grid = param_grid
        self.scoring = scoring
        self.refit = refit
        if cv is None:
            self.cv = KFold()
        else:
            self.cv = cv
            
        scoring_dict = {'explained_variance': metrics.explained_variance_score,
                        'max_error': metrics.max_error,
                        'neg_mean_absolute_error': metrics.mean_absolute_error,
                        'neg_mean_squared_error': metrics.mean_squared_error,
                        'neg_root_mean_squared_error': metrics.mean_squared_error, 
                        'neg_mean_squared_log_error': metrics.mean_squared_log_error,
                        'neg_median_absolute_error': metrics.median_absolute_error,
                        'r2': metrics.r2_score,
                        'neg_mean_poisson_deviance': metrics.mean_poisson_deviance,
                        'neg_mean_gamma_deviance': metrics.mean_gamma_deviance}
        
        if self.scoring is not None:
            assert(self.scoring in scoring_dict.keys()),'no such scoring: ' + str(self.scoring) + '.'
            self.scoring = scoring_dict[scoring]
            
        model = estimator.__init__

        sig = signature(model)

        for i in self.param_grid.keys():
            assert(i in sig.parameters.keys()), type(estimator).__name__ + " object has no argument like: " + str(i) 
         

    def fit(self, X, y):
        
        X, y = np.array(X), np.array(y)
        
        all_list = [list(i) for i in self.param_grid.values()]

        permutations = list(product(*all_list)) 

        parameters = list()
        for i in permutations:
            permutation_dict = dict()
            for ind, val in enumerate(i):
                permutation_dict[list(self.param_grid.keys())[ind]] = val
            parameters.append(permutation_dict)

        scores = list()
        for param_dict in parameters:
            for param in list(self.estimator.get_params().keys()):
                if param in param_dict:
                    self.estimator.__setattr__(param, param_dict[param])
            scr = list()
            for train_index, test_index in cv.split(X):
                xTrain, xTest, yTrain, yTest = X[train_index], X[test_index], y[train_index], y[test_index]
                self.estimator.fit(xTrain, yTrain)
                y_pred = self.estimator.predict(xTest)
                if self.scoring is not None:
                    scr.append(self.scoring(y_pred, yTest))
                else:
                    scr.append(self.estimator.score(xTest, yTest))
            scores.append(np.mean(scr))
        scores = np.array(scores)
        self.best_score_ = np.min(scores)
        self.best_params_ = parameters[np.where(scores == np.min(scores))[0][0]]
        
        if self.refit:
            for param in list(self.estimator.get_params().keys()):
                if param in param_dict:
                    self.estimator.__setattr__(param, self.best_params_[param])
            
            self.predict = self.estimator.predict
        
        return self
