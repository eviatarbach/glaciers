import pickle

from geopy.distance import distance
import sklearn
import sklearn.neighbors
import statsmodels.api as sm

# Features to use in the accumulation and ablation gradient linear,
# as determined by subset selection
ABL_FEATURES = ['summer_temperature', 'cloud_cover']
ACC_FEATURES = ['max_elevation', 'cloud_cover']


def distance_func(p1, p2):
    """
    Returns distance between two coordinates
    """
    return distance((p1[0], p1[1]), (p2[0], p2[1])).kilometers

with open('data/serialized/glaciers_climate', 'br') as glaciers_file,\
     open('data/serialized/all_glaciers', 'br') as all_glaciers_file:
    glaciers = pickle.load(glaciers_file)
    all_glaciers = pickle.load(all_glaciers_file)

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

    g_abl_features = sm.add_constant((abl_data - abl_mean)/abl_std)
    g_acc_features = sm.add_constant((acc_data - acc_mean)/acc_std)

    neighbours_abl = sklearn.neighbors.KNeighborsRegressor(n_neighbors=20, weights='distance',
                                                           metric=distance_func)
    neighbours_acc = sklearn.neighbors.KNeighborsRegressor(n_neighbors=20, weights='distance',
                                                           metric=distance_func)

    neighbours_abl.fit(glaciers[abl_mask][['lat', 'lon']], g_abl)
    neighbours_acc.fit(glaciers[acc_mask][['lat', 'lon']], g_acc)

    glm_abl = sm.GLM(glaciers[abl_feature_mask]['g_abl'], g_abl_features,
                     family=sm.families.Gamma()).fit()
    glm_acc = sm.GLM(glaciers[acc_feature_mask]['g_acc'], g_acc_features,
                     family=sm.families.Gamma()).fit()

    for glacier in glaciers.index:
        neighbour_abl_res = neighbours_abl.predict(glaciers.loc[[glacier], ['lat', 'lon']])
        if glaciers.loc[glacier, ABL_FEATURES].isnull().any():
            abl_res = neighbour_abl_res
        else:
            glm_abl_res = glm_abl.predict(sm.add_constant((glaciers.loc[[glacier], ABL_FEATURES]
                                                           - abl_mean)/abl_std))
            abl_res = 0.5*neighbour_abl_res + 0.5*glm_abl_res

        neighbour_acc_res = neighbours_acc.predict(glaciers.loc[[glacier], ['lat', 'lon']])
        if glaciers.loc[glacier, ACC_FEATURES].isnull().any():
            acc_res = neighbour_acc_res
        else:
            glm_acc_res = glm_acc.predict(sm.add_constant((glaciers.loc[[glacier], ACC_FEATURES]
                                                           - acc_mean)/acc_std))
            acc_res = 0.6*neighbour_acc_res + 0.4*glm_acc_res

        print(glaciers.loc[glacier, 'g_abl'], abl_res)
        print(glaciers.loc[glacier, 'g_acc'], acc_res)
