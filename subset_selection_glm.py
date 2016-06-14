import pickle
import itertools

import numpy
import sklearn
import sklearn.cross_validation
import sklearn.linear_model
import statsmodels.api as sm

glaciers = pickle.load(open('data/serialized/glaciers_climate', 'br')).dropna()

# Mass-balance gradient cannot be negative
# glaciers = glaciers[glaciers['g_acc'] > 0]
glaciers = glaciers[glaciers['g_abl'] > 0]


def power_set(iterable):
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s) + 1))

features = ['max_elevation', 'median_elevation', 'continentality', 'summer_temperature',
            'precipitation', 'winter_precipitation', 'cloud_cover', 'lapse_rate']

cv_list = []

glaciers = glaciers.reindex(numpy.random.permutation(glaciers.index))
X = glaciers[features]
y = glaciers['g_abl']
Xnorm = sm.add_constant((X - X.mean())/(X.std()))

i = 0
for subset in power_set(features):
    print(i)
    subset_err = []
    for train, test in sklearn.cross_validation.KFold(len(glaciers), n_folds=20):
        Xtrain = Xnorm.iloc[train, :]
        ytrain = y.iloc[train]

        Xtest = Xnorm.iloc[test, :]
        ytest = y.iloc[test]

        X2 = Xtrain[['const'] + list(subset)]
        model = sm.GLM(ytrain, X2, family=sm.families.Gamma()).fit()
        error = ((ytest - model.predict(Xtest[['const'] + list(subset)]))**2).mean()

        subset_err.append(error)
    cv_list.append((subset, numpy.mean(subset_err)))
    i += 1
