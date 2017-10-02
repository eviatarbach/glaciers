import numpy
import pandas
import sklearn
import sklearn.neighbors
from sklearn.linear_model import LinearRegression

# Features to use in the ablation gradient and G regression, as determined by subset selection
ABL_FEATURES = ['continentality', 'summer_temperature', 'lapse_rate']
G_FEATURES = ['max_elevation', 'continentality', 'winter_precipitation', 'cloud_cover']

glaciers = pandas.read_pickle('data/serialized/glaciers_climate')
glaciers['G'] = glaciers['g_acc']/glaciers['g_abl'] - 1
all_glaciers = pandas.read_pickle('data/serialized/all_glaciers')

all_glaciers['max_elevation'] = all_glaciers['Zmax']

# Remove null
mask = ~glaciers['G'].isnull() & (glaciers['g_abl'] > 0)

g_abl = glaciers[mask]['g_abl']
G = glaciers[mask]['G']

abl_feature_mask = ~glaciers[ABL_FEATURES].isnull().any(axis=1) & mask
G_feature_mask = ~glaciers[G_FEATURES].isnull().any(axis=1) & mask

abl_data = glaciers[abl_feature_mask][ABL_FEATURES]
G_data = glaciers[G_feature_mask][G_FEATURES]

abl_mean = abl_data.mean()
abl_std = abl_data.std()

G_mean = G_data.mean()
G_std = G_data.std()

g_abl_features = (abl_data - abl_mean)/abl_std
G_features = (G_data - G_mean)/G_std

neighbours_abl = sklearn.neighbors.KNeighborsRegressor(n_neighbors=20, weights='distance',
                                                       metric='haversine')
neighbours_G = sklearn.neighbors.KNeighborsRegressor(n_neighbors=20, weights='distance',
                                                     metric='haversine')

neighbours_abl.fit(numpy.radians(glaciers[mask][['lat', 'lon']]), g_abl)
neighbours_G.fit(numpy.radians(glaciers[mask][['lat', 'lon']]), G)

model_abl = LinearRegression().fit(g_abl_features, glaciers[abl_feature_mask]['g_abl'])
model_G = LinearRegression().fit(G_features, glaciers[G_feature_mask]['G'])

c = 0
for glacier in all_glaciers.index:
    c += 1
    print(c/len(all_glaciers))

    if all_glaciers.loc[glacier, G_FEATURES].isnull().any():
        neighbour_G_res = neighbours_G.predict(numpy.radians(all_glaciers.loc[[glacier],
                                                                              ['lat', 'lon']]))
        G_res = neighbour_G_res
    else:
        lm_G_res = model_G.predict((all_glaciers.loc[[glacier], G_FEATURES] - G_mean)/G_std)
        G_res = lm_G_res

    if all_glaciers.loc[glacier, ABL_FEATURES].isnull().any():
        neighbour_abl_res = neighbours_abl.predict(numpy.radians(all_glaciers.loc[[glacier],
                                                                                  ['lat', 'lon']]))
        abl_res = neighbour_abl_res
    else:
        lm_abl_res = model_abl.predict((all_glaciers.loc[[glacier], ABL_FEATURES]
                                        - abl_mean)/abl_std)
        abl_res = lm_abl_res

    all_glaciers.loc[glacier, 'G'] = G_res
    all_glaciers.loc[glacier, 'g_abl'] = abl_res
    all_glaciers.loc[glacier, 'g_acc'] = abl_res*(G_res + 1)

all_glaciers.to_pickle('data/serialized/all_glaciers')
