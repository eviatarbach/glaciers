import itertools

import numpy
import pandas
import sklearn
import sklearn.cross_validation
import sklearn.neighbors
import sklearn.linear_model
import statsmodels.api as sm
from glm import GammaRegressor

glaciers = pandas.read_pickle('data/serialized/glaciers_climate')


def power_set(iterable):
    # From StackOverflow
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s) + 1))

features = ['max_elevation', 'median_elevation', 'continentality', 'summer_temperature',
            'precipitation', 'winter_precipitation', 'cloud_cover', 'lapse_rate']

runs = []

glaciers.loc[glaciers['lat'] < 25, 'region'] = 'low'
glaciers.loc[(glaciers['lat'] >= 25) & (glaciers['lat'] <= 55), 'region'] = 'mid'
glaciers.loc[glaciers['lat'] > 55, 'region'] = 'high'

for i in range(100):
    print(i)
    for gradient in ['g', 'g_abl', 'g_acc']:
        # Timescale cannot be negative
        if gradient in ['g', 'g_abl']:
            data = glaciers[glaciers[gradient] > 0]
        else:
            data = glaciers[glaciers[gradient].notnull()]

        cv_list = []

        data = data.reindex(numpy.random.permutation(data.index))

        X = data[features]
        y = data[gradient]
        X_norm = (X - X.mean())/(X.std())
        X_val, X_test, y_val, y_test = sklearn.cross_validation.train_test_split(X_norm, y,
                                                                                 test_size=0.2)
        coords_val = glaciers.loc[X_val.index][['lat', 'lon']]
        coords_test = glaciers.loc[X_test.index][['lat', 'lon']]

        i = 0
        for subset in power_set(features):
            print(i)
            subset_err = []
            not_null_indices = X_val[list(subset)].notnull().all(axis=1)
            for train_index, test_index in sklearn.cross_validation.StratifiedKFold(glaciers.loc[not_null_indices.index, 'region'][not_null_indices]):
                # print(glaciers.loc[not_null_indices.index, 'region'][not_null_indices].iloc[train_index].value_counts())
                model = GammaRegressor().fit(X_val[list(subset)][not_null_indices].iloc[train_index], y_val[not_null_indices].iloc[train_index])
                error = numpy.mean((model.predict(X_val[list(subset)][not_null_indices].iloc[test_index]) - y_val[not_null_indices].iloc[test_index])**2)
                subset_err.append(error)
            cv_list.append((subset, numpy.mean(subset_err)))
            i += 1

        best_subset = cv_list[numpy.argmin([subset[1] for subset in cv_list])][0]
        not_null_indices_test = X_test[list(best_subset)].notnull().all(axis=1)
        not_null_indices_val = X_val[list(best_subset)].notnull().all(axis=1)
        neighbours_val = sklearn.neighbors.KNeighborsRegressor(n_neighbors=20, weights='distance',
                                                               metric='haversine', algorithm='ball_tree')
        neighbours_val.fit(numpy.radians(coords_val), y_val)
        neighbours_test = sklearn.neighbors.KNeighborsRegressor(n_neighbors=20, weights='distance',
                                                                metric='haversine', algorithm='ball_tree')
        neighbours_test.fit(numpy.radians(coords_test), y_test)
        clf_val = GammaRegressor().fit(X_val[list(best_subset)][not_null_indices_val],
                                       y_val[not_null_indices_val])
        error = (0.5*clf_val.predict(X_test[list(best_subset)][not_null_indices_test])
                 + 0.5*neighbours_val.predict(coords_test[not_null_indices_test])) - y_test[not_null_indices_test]
        error = numpy.sqrt(numpy.mean(error**2))

        try:
            clf_test = GammaRegressor().fit(X_test[list(best_subset)][not_null_indices_test],
                                            y_test[not_null_indices_test])
            r2 = sklearn.metrics.r2_score(y_test[not_null_indices_test],
                                          0.5*clf_test.predict(X_test[list(best_subset)][not_null_indices_test])
                                          + 0.5*neighbours_test.predict(coords_test[not_null_indices_test]))
        except:
            pass
        # error = sklearn.cross_validation.cross_val_score(GammaRegressor(),
        #                                                  X_test[list(best_subset)][not_null_indices],
        #                                                  y_test[not_null_indices], scoring='r2')
        runs.append((gradient, best_subset, r2, error))
