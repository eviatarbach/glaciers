import pickle
import itertools

import numpy
import sklearn
import sklearn.cross_validation
import sklearn.linear_model
import statsmodels.api as sm

glaciers = pickle.load(open('glaciers', 'br')).dropna()

# Mass-balance gradient cannot be negative
glaciers = glaciers[glaciers['g'] > 0]

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))

features = ['max_elevation', 'median_elevation', 'continentality', 'summer_temperature', 'precipitation', 'winter_precipitation', 'cloud_cover', 'lapse_rate']

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
        model = sm.GLM(ytrain, X2, family=sm.families.InverseGaussian(sm.families.links.log))
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
        model = sm.GLM(ytrain, X2, family=sm.families.InverseGaussian(sm.families.links.log))
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

aic_list = []
bic_list = []
cv_list = []

glaciers = glaciers.reindex(numpy.random.permutation(glaciers.index))
X = glaciers[features]
y = glaciers['g']
Xnorm = sm.add_constant((X - X.mean())/(X.std()))

i = 0
for subset in powerset(features):
    print(i)
    subset_err = []
    for train, test in sklearn.cross_validation.KFold(len(glaciers), n_folds=8):
        Xtrain = Xnorm.iloc[train, :]
        ytrain = y.iloc[train]

        Xtest = Xnorm.iloc[test, :]
        ytest = y.iloc[test]

        X2 = Xtrain[['const'] + list(subset)]
        model = sm.GLM(ytrain, X2, family=sm.families.InverseGaussian(sm.families.links.log)).fit()
        error = ((ytest - model.predict(Xtest[['const'] + list(subset)]))**2).mean()

        subset_err.append(error)
    cv_list.append((subset, numpy.mean(subset_err)))
    i += 1
