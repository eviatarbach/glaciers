import itertools

import numpy
import pandas
import sklearn
import sklearn.model_selection
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

glaciers = pandas.read_pickle('data/serialized/glaciers_climate')
glaciers['G'] = glaciers['g_acc']/glaciers['g_abl'] - 1


def power_set(iterable):
    # From itertools documentation
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s) + 1))


features = ['max_elevation', 'median_elevation', 'continentality', 'summer_temperature',
            'precipitation', 'winter_precipitation', 'cloud_cover', 'lapse_rate']

runs = []

glaciers_train, glaciers_test = sklearn.model_selection.train_test_split(glaciers, test_size=0.20)

for i in range(100):
    print(i)
    for gradient in ['g_abl', 'G']:
        # Timescale cannot be negative
        if gradient == 'g_abl':
            data = glaciers_train[glaciers_train[gradient] > 0]
        else:
            data = glaciers_train[glaciers_train[gradient].notnull()]

        cv_list = []

        data = data.reindex(numpy.random.permutation(data.index))

        X = data[features]
        y = data[gradient]
        X_norm = (X - X.mean())/(X.std())

        for subset in list(power_set(features))[1:]:
            subset_err = []
            not_null_indices = X_norm[list(subset)].notnull().all(axis=1)
            X_subset = X_norm[list(subset)][not_null_indices]
            y_subset = y[not_null_indices]
            skf = sklearn.model_selection.KFold(n_splits=6)
            for train_index, test_index in skf.split(X_subset, y_subset):
                model = LinearRegression().fit(X_subset.iloc[train_index],
                                               y_subset.iloc[train_index])
                error = numpy.mean((model.predict(X_subset.iloc[test_index])
                                    - y_subset.iloc[test_index])**2)
                subset_err.append(error)
            cv_list.append({'subset': subset, 'err': numpy.mean(subset_err)})

        best_subset = cv_list[numpy.argmin([subset['err'] for subset in cv_list])]['subset']

        runs.append({'gradient': gradient, 'subset': best_subset})

runs = pandas.DataFrame(runs)

for gradient in ['g_abl', 'G']:
    subset = runs[runs['gradient'] == gradient]['subset'].value_counts().argmax()

    if gradient == 'g_abl':
        data = glaciers_test[glaciers_test[gradient] > 0]
    else:
        data = glaciers_test[glaciers_test[gradient].notnull()]

    X = data[list(subset)]
    y = data[gradient]
    X_norm = (X - X.mean())/(X.std())

    not_null_indices = X_norm[list(subset)].notnull().all(axis=1)
    X_nn = X_norm[list(subset)][not_null_indices]
    y_nn = y[not_null_indices]

    model = LinearRegression().fit(X_nn, y_nn)
    error = y_nn - model.predict(X_nn)
    plt.hist(error, bins=15)
    plt.show()

    kf = sklearn.model_selection.LeaveOneOut()
    subset_err = []
    for train_index, test_index in kf.split(X_nn, y_nn):
        model = LinearRegression().fit(X_nn.iloc[train_index], y_nn.iloc[train_index])
        error = y_nn.iloc[test_index] - model.predict(X_nn.iloc[test_index])
        subset_err.append(numpy.sqrt(numpy.mean(error**2)))

    print(gradient, numpy.mean(subset_err))
