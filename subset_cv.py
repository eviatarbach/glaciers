features = ['max_elevation', 'median_elevation', 'continentality', 'summer_temperature', 'precipitation', 'winter_precipitation', 'cloud_cover']

results = []

glaciers = glaciers.reindex(numpy.random.permutation(glaciers.index))
X = glaciers[features]
y = glaciers['g']
Xnorm = (X - X.mean())/(X.std())

clf = sklearn.linear_model.LinearRegression()
for subset in powerset(features):
    if subset:
        Xtrain = Xnorm.iloc[:85, :]
        ytrain = y.iloc[:85]
        X2 = Xtrain[list(subset)]
        results.append((-sklearn.cross_validation.cross_val_score(clf, X2, ytrain, scoring='mean_squared_error').mean(), subset))
