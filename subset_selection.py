import itertools

import numpy
import pandas
import sklearn
import sklearn.model_selection
import sklearn.neighbors
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

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
    for gradient in ['g_abl', 'g_acc']:
        # Timescale cannot be negative
        if gradient == 'g_abl':
            data = glaciers[glaciers[gradient] > 0]
        else:
            data = glaciers[glaciers[gradient].notnull()]

        cv_list = []

        data = data.reindex(numpy.random.permutation(data.index))

        X = data[features]
        y = data[gradient]
        X_norm = (X - X.mean())/(X.std())
        X_val, X_test, y_val, y_test = sklearn.model_selection.train_test_split(X_norm, y,
                                                                                test_size=0.05)
        coords_val = glaciers.loc[X_val.index][['lat', 'lon']]
        coords_test = glaciers.loc[X_test.index][['lat', 'lon']]

        for subset in list(power_set(features))[1:]:
            subset_err = []
            not_null_indices = X_val[list(subset)].notnull().all(axis=1)
            X_val_subset = X_val[list(subset)][not_null_indices]
            y_val_subset = y_val[not_null_indices]
            skf = sklearn.model_selection.StratifiedKFold(n_splits=3)
            for train_index, test_index in skf.split(X_val_subset,
                                                     glaciers.loc[not_null_indices.index,
                                                                  'region'][not_null_indices]):
                model = LinearRegression().fit(X_val_subset.iloc[train_index],
                                               y_val_subset.iloc[train_index])
                error = numpy.mean((model.predict(X_val_subset.iloc[test_index])
                                    - y_val_subset.iloc[test_index])**2)
                subset_err.append(error)
            cv_list.append({'subset': subset, 'err': numpy.mean(subset_err)})

        best_subset = cv_list[numpy.argmin([subset['err'] for subset in cv_list])]['subset']

        runs.append({'gradient': gradient, 'subset': best_subset})

runs = pandas.DataFrame(runs)

for gradient in ['g_abl', 'g_acc']:
    subset = runs[runs['gradient'] == gradient]['subset'].value_counts().argmax()

    if gradient == 'g_abl':
        data = glaciers[glaciers[gradient] > 0]
    else:
        data = glaciers[glaciers[gradient].notnull()]

    X = data[list(subset)]
    y = data[gradient]
    X_norm = (X - X.mean())/(X.std())

    not_null_indices = X_norm[list(subset)].notnull().all(axis=1)
    X_nn = X_norm[list(subset)][not_null_indices]
    y_nn = y[not_null_indices]

    model = LinearRegression().fit(X_nn, y_nn)
    coords = glaciers.loc[X_nn.index][['lat', 'lon']]
    neighbours = sklearn.neighbors.KNeighborsRegressor(n_neighbors=20, weights='distance',
                                                       metric='haversine',
                                                       algorithm='ball_tree')
    neighbours.fit(numpy.radians(coords), y_nn)
    error = (y_nn - (0.5*model.predict(X_nn) + 0.5*neighbours.predict(coords)))
    plt.hist(error, bins=15)
    plt.show()

    kf = sklearn.model_selection.KFold(n_splits=20)
    subset_err = []
    for train_index, test_index in kf.split(X_nn, y_nn):
        model = LinearRegression().fit(X_nn.iloc[train_index], y_nn.iloc[train_index])
        coords_train = glaciers.loc[X_nn.iloc[train_index].index][['lat', 'lon']]
        coords_test = glaciers.loc[X_nn.iloc[test_index].index][['lat', 'lon']]
        neighbours = sklearn.neighbors.KNeighborsRegressor(n_neighbors=20, weights='distance',
                                                           metric='haversine',
                                                           algorithm='ball_tree')
        neighbours.fit(numpy.radians(coords_train), y_nn.iloc[train_index])
        error = ((0.5*model.predict(X_nn.iloc[test_index])
                  + 0.5*neighbours.predict(coords_test)) - y_nn.iloc[test_index])
        subset_err.append(numpy.sqrt(numpy.mean(error**2)))

    print(gradient, numpy.mean(subset_err))
