import pickle
import itertools

import numpy
import sklearn
import sklearn.cross_validation
import sklearn.linear_model
import statsmodels.api as sm

glaciers = pickle.load(open('glaciers', 'br')).dropna()

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))

features = ['max_elevation', 'median_elevation', 'continentality', 'summer_temperature', 'precipitation', 'winter_precipitation', 'cloud_cover']

glaciers = glaciers.reindex(numpy.random.permutation(glaciers.index))
X = glaciers[features]
y = glaciers['g']
Xnorm = (X - X.mean())/(X.std())

def aic(Xtest, ytest, Xtrain, ytrain):
    Xtest = sm.add_constant(Xtest)
    Xtrain = sm.add_constant(Xtrain)
    results = []
    for subset in powerset(features):
        X2 = Xtrain[list(('const',) + subset)]
        model = sm.OLS(ytrain, X2)
        results.append((model.fit().aic, subset, model))
    aic_score, subset_aic, model_aic = results[numpy.argmin([r[0] for r in results])]
    error = ((ytest - model_aic.fit().predict(Xtest[['const'] + list(subset_aic)]))**2).mean()
    return (subset_aic, error)

def bic(Xtest, ytest, Xtrain, ytrain):
    Xtest = sm.add_constant(Xtest)
    Xtrain = sm.add_constant(Xtrain)
    results = []
    for subset in powerset(features):
        X2 = Xtrain[list(('const',) + subset)]
        model = sm.OLS(ytrain, X2)
        results.append((model.fit().bic, subset, model))
    bic_score, subset_bic, model_bic = results[numpy.argmin([r[0] for r in results])]
    error = ((ytest - model_bic.fit().predict(Xtest[['const'] + list(subset_bic)]))**2).mean()
    return (subset_bic, error)

def cv(Xtest, ytest, Xtrain, ytrain):
    results = []
    clf = sklearn.linear_model.LinearRegression()
    for subset in powerset(features):
        if subset:
            X2 = Xtrain[list(subset)]
            results.append((-sklearn.cross_validation.cross_val_score(clf, X2, ytrain, scoring='mean_squared_error').mean(), subset))
    cv_score, subset_cv = results[numpy.argmin([r[0] for r in results])]
    model_cv = clf.fit(Xtrain[list(subset_cv)], ytrain)
    error = ((ytest - model_cv.predict(Xtest[list(subset_cv)]))**2).mean()
    return (subset_cv, error)

def l1(Xtest, ytest, Xtrain, ytrain):
    results = []
    clf = sklearn.linear_model.LassoCV()
    model = clf.fit(Xtrain, ytrain)
    error = ((ytest - model.predict(Xtest))**2).mean()
    return (tuple(map(lambda i: features[i], model.coef_.nonzero()[0])), error)

def l2(Xtest, ytest, Xtrain, ytrain):
    results = []
    clf = sklearn.linear_model.RidgeCV()
    model = clf.fit(Xtrain, ytrain)
    error = ((ytest - model.predict(Xtest))**2).mean()
    return (tuple(map(lambda i: features[i], model.coef_.nonzero()[0])), error)

def bayesian_ridge(Xtest, ytest, Xtrain, ytrain):
    results = []
    clf = sklearn.linear_model.BayesianRidge()
    model = clf.fit(Xtrain, ytrain)
    error = ((ytest - model.predict(Xtest))**2).mean()
    return (tuple(map(lambda i: features[i], model.coef_.nonzero()[0])), error)

def ard(Xtest, ytest, Xtrain, ytrain):
    results = []
    clf = sklearn.linear_model.ARDRegression()
    model = clf.fit(Xtrain, ytrain)
    error = ((ytest - model.predict(Xtest))**2).mean()
    return (tuple(map(lambda i: features[i], model.coef_.nonzero()[0])), error)

aic_list = []
bic_list = []
cv_list = []
l1_list = []
l2_list = []
bayesian_list = []
ard_list = []

for i in range(10):
    glaciers = glaciers.reindex(numpy.random.permutation(glaciers.index))
    X = glaciers[features]
    y = glaciers['g']
    Xnorm = (X - X.mean())/(X.std())

    for train, test in sklearn.cross_validation.KFold(len(glaciers), n_folds=8):
        Xtrain = Xnorm.iloc[train, :]
        ytrain = y.iloc[train]

        Xtest = Xnorm.iloc[test, :]
        ytest = y.iloc[test]

        aic_list.append(aic(Xtest, ytest, Xtrain, ytrain))
        bic_list.append(bic(Xtest, ytest, Xtrain, ytrain))
        cv_list.append(cv(Xtest, ytest, Xtrain, ytrain))
        l2_list.append(l2(Xtest, ytest, Xtrain, ytrain))
        l1_list.append(l1(Xtest, ytest, Xtrain, ytrain))
        bayesian_list.append(bayesian_ridge(Xtest, ytest, Xtrain, ytrain))
        ard_list.append(ard(Xtest, ytest, Xtrain, ytrain))
