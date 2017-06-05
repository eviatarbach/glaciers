import pickle

import numpy
import statsmodels.api as sm
import sklearn
import sklearn.neighbors
import sklearn.cross_validation

glaciers = pickle.load(open('data/serialized/glaciers_climate', 'br')).dropna()

glaciers = glaciers.reindex(numpy.random.permutation(glaciers.index))

features = ['max_elevation', 'median_elevation', 'continentality', 'summer_temperature',
            'precipitation', 'winter_precipitation', 'cloud_cover', 'lapse_rate']

Xf = glaciers[features]
Xnorm = sm.add_constant((Xf - Xf.mean())/(Xf.std()))

X = numpy.radians(glaciers[['lat', 'lon']])
y = glaciers['g_abl']

errs = []
for train, test in sklearn.cross_validation.KFold(len(glaciers), n_folds=20):
    Xtrain = X.iloc[train, :]
    ytrain = y.iloc[train]

    Xtest = X.iloc[test, :]
    ytest = y.iloc[test]

    neigh = sklearn.neighbors.KNeighborsRegressor(n_neighbors=20, weights='distance',
                                                  metric='haversine')
    neigh.fit(Xtrain, ytrain)

    res1 = neigh.predict(Xtest)

    X2 = Xnorm.iloc[train, :][['const', 'summer_temperature', 'cloud_cover']]
    model = sm.GLM(ytrain, X2, family=sm.families.Gamma()).fit()

    res2 = model.predict(Xnorm.iloc[test, :][['const', 'summer_temperature', 'cloud_cover']])

    res = 0.5*res1 + 0.5*res2

    error = numpy.sqrt(((ytest - res)**2).mean())

    errs.append(error)
