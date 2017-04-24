import numpy
import pandas
import sklearn
import sklearn.neighbors
import statsmodels.api as sm

# Features to use in the accumulation and ablation gradient linear,
# as determined by subset selection
ABL_FEATURES = ['summer_temperature', 'cloud_cover']
ACC_FEATURES = ['max_elevation', 'cloud_cover']
G_FEATURES = ['max_elevation', 'median_elevation', 'continentality']

glaciers = pandas.read_pickle('data/serialized/glaciers_climate')
all_glaciers = pandas.read_pickle('data/serialized/all_glaciers')

# Remove negative ablation gradients, since the timescale cannot be
# negative
abl_mask = ~glaciers['g_abl'].isnull() & (glaciers['g_abl'] > 0)
acc_mask = ~glaciers['g_acc'].isnull()
g_mask = ~glaciers['g'].isnull() & (glaciers['g'] > 0)

g_abl = glaciers[abl_mask]['g_abl']
g_acc = glaciers[acc_mask]['g_acc']
g = glaciers[g_mask]['g']

abl_feature_mask = ~glaciers[ABL_FEATURES].isnull().any(axis=1) & abl_mask
acc_feature_mask = ~glaciers[ACC_FEATURES].isnull().any(axis=1) & acc_mask
g_feature_mask = ~glaciers[G_FEATURES].isnull().any(axis=1) & g_mask

abl_data = glaciers[abl_feature_mask][ABL_FEATURES]
acc_data = glaciers[acc_feature_mask][ACC_FEATURES]
g_data = glaciers[g_feature_mask][G_FEATURES]

abl_mean = abl_data.mean()
abl_std = abl_data.std()

acc_mean = acc_data.mean()
acc_std = acc_data.std()

g_mean = g_data.mean()
g_std = g_data.std()

g_abl_features = sm.add_constant((abl_data - abl_mean)/abl_std)
g_acc_features = sm.add_constant((acc_data - acc_mean)/acc_std)
g_features = sm.add_constant((g_data - g_mean)/g_std)

neighbours_abl = sklearn.neighbors.KNeighborsRegressor(n_neighbors=20, weights='distance',
                                                       metric='haversine')
neighbours_acc = sklearn.neighbors.KNeighborsRegressor(n_neighbors=20, weights='distance',
                                                       metric='haversine')
neighbours_g = sklearn.neighbors.KNeighborsRegressor(n_neighbors=20, weights='distance',
                                                     metric='haversine')

neighbours_abl.fit(numpy.radians(glaciers[abl_mask][['lat', 'lon']]), g_abl)
neighbours_acc.fit(numpy.radians(glaciers[acc_mask][['lat', 'lon']]), g_acc)
neighbours_g.fit(numpy.radians(glaciers[g_mask][['lat', 'lon']]), g)

glm_abl = sm.GLM(glaciers[abl_feature_mask]['g_abl'], g_abl_features,
                 family=sm.families.Gamma()).fit()
glm_acc = sm.GLM(glaciers[acc_feature_mask]['g_acc'], g_acc_features,
                 family=sm.families.Gamma()).fit()
glm_g = sm.GLM(glaciers[acc_feature_mask]['g'], g_acc_features,
               family=sm.families.Gamma()).fit()

c = 0
for glacier in all_glaciers.index:
    c += 1
    print(c/len(all_glaciers))
    neighbour_abl_res = neighbours_abl.predict(numpy.radians(all_glaciers.loc[[glacier],
                                                                              ['lat', 'lon']]))
    if all_glaciers.loc[glacier, ABL_FEATURES].isnull().any():
        abl_res = neighbour_abl_res
    else:
        glm_abl_res = glm_abl.predict(sm.add_constant((all_glaciers.loc[[glacier], ABL_FEATURES]
                                                       - abl_mean)/abl_std))
        abl_res = 0.5*neighbour_abl_res + 0.5*glm_abl_res

    neighbour_acc_res = neighbours_acc.predict(numpy.radians(all_glaciers.loc[[glacier],
                                                                              ['lat', 'lon']]))
    if all_glaciers.loc[glacier, ACC_FEATURES].isnull().any():
        acc_res = neighbour_acc_res
    else:
        glm_acc_res = glm_acc.predict(sm.add_constant((all_glaciers.loc[[glacier], ACC_FEATURES]
                                                       - acc_mean)/acc_std))
        acc_res = 0.5*neighbour_acc_res + 0.5*glm_acc_res

    neighbour_g_res = neighbours_g.predict(numpy.radians(all_glaciers.loc[[glacier],
                                                                          ['lat', 'lon']]))
    if all_glaciers.loc[glacier, G_FEATURES].isnull().any():
        g_res = neighbour_g_res
    else:
        glm_g_res = glm_g.predict(sm.add_constant((all_glaciers.loc[[glacier], G_FEATURES]
                                                   - g_mean)/g_std))
        g_res = 0.5*neighbour_g_res + 0.5*glm_g_res

    all_glaciers.loc[glacier, 'g_abl'] = abl_res
    all_glaciers.loc[glacier, 'g_acc'] = acc_res
    all_glaciers.loc[glacier, 'g'] = g_res

all_glaciers.to_pickle('data/serialized/all_glaciers')
