import numpy
import pandas
import sklearn
import sklearn.neighbors
from sklearn.linear_model import LinearRegression

# Features to use in the accumulation and ablation gradient linear,
# as determined by subset selection
ABL_FEATURES = ['continentality', 'summer_temperature', 'lapse_rate']
ACC_FEATURES = ['max_elevation', 'median_elevation', 'continentality']

glaciers = pandas.read_pickle('data/serialized/glaciers_climate')
all_glaciers = pandas.read_pickle('data/serialized/all_glaciers')

all_glaciers['max_elevation'] = all_glaciers['Zmax']

# Remove negative ablation gradients, since the timescale cannot be
# negative
abl_mask = ~glaciers['g_abl'].isnull() & (glaciers['g_abl'] > 0)
acc_mask = ~glaciers['g_acc'].isnull()

g_abl = glaciers[abl_mask]['g_abl']
g_acc = glaciers[acc_mask]['g_acc']

abl_feature_mask = ~glaciers[ABL_FEATURES].isnull().any(axis=1) & abl_mask
acc_feature_mask = ~glaciers[ACC_FEATURES].isnull().any(axis=1) & acc_mask

abl_data = glaciers[abl_feature_mask][ABL_FEATURES]
acc_data = glaciers[acc_feature_mask][ACC_FEATURES]

abl_mean = abl_data.mean()
abl_std = abl_data.std()

acc_mean = acc_data.mean()
acc_std = acc_data.std()

g_abl_features = (abl_data - abl_mean)/abl_std
g_acc_features = (acc_data - acc_mean)/acc_std

neighbours_abl = sklearn.neighbors.KNeighborsRegressor(n_neighbors=20, weights='distance',
                                                       metric='haversine')
neighbours_acc = sklearn.neighbors.KNeighborsRegressor(n_neighbors=20, weights='distance',
                                                       metric='haversine')

neighbours_abl.fit(numpy.radians(glaciers[abl_mask][['lat', 'lon']]), g_abl)
neighbours_acc.fit(numpy.radians(glaciers[acc_mask][['lat', 'lon']]), g_acc)

model_abl = LinearRegression().fit(g_abl_features, glaciers[abl_feature_mask]['g_abl'])
model_acc = LinearRegression().fit(g_acc_features, glaciers[acc_feature_mask]['g_acc'])

c = 0
for glacier in all_glaciers.index:
    c += 1
    print(c/len(all_glaciers))

    if all_glaciers.loc[glacier, ABL_FEATURES].isnull().any():
        neighbour_abl_res = neighbours_abl.predict(numpy.radians(all_glaciers.loc[[glacier],
                                                                                  ['lat', 'lon']]))
        abl_res = neighbour_abl_res
    else:
        lm_abl_res = model_abl.predict((all_glaciers.loc[[glacier], ABL_FEATURES]
                                        - abl_mean)/abl_std)
        abl_res = lm_abl_res

    if all_glaciers.loc[glacier, ACC_FEATURES].isnull().any():
        neighbour_acc_res = neighbours_acc.predict(numpy.radians(all_glaciers.loc[[glacier],
                                                                                  ['lat', 'lon']]))
        acc_res = neighbour_acc_res
    else:
        lm_acc_res = model_acc.predict((all_glaciers.loc[[glacier], ACC_FEATURES]
                                        - acc_mean)/acc_std)
        acc_res = lm_acc_res

    all_glaciers.loc[glacier, 'g_abl'] = abl_res
    all_glaciers.loc[glacier, 'g_acc'] = acc_res

all_glaciers.to_pickle('data/serialized/all_glaciers')
