hist = all_glaciers['P'].plot(kind='kde', sharex=False, subplots=True)

P = all_glaciers.loc[region_name]['P']
hist[0].set_xlim(P.quantile([0.01, 1]).values)

L = all_glaciers.loc[region_name]['LENGTH']
hist[4].set_xlim(L.quantile([0, 0.99]).values)

H = all_glaciers.loc[region_name]['Thickness']
hist[3].set_xlim(H.quantile([0, 0.99]).values)

zela = all_glaciers.loc[region_name]['zela']
hist[2].set_xlim(zela.quantile([0.01, 1]).values)
