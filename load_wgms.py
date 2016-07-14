with open('data/MAIL_WGMS/00_rgi50_links.20151130_WithCategories.csv', 'r',
       encoding='ISO-8859-1') as id_file:
    # Load slope and aspect data into WGMS data
    # data = data.set_index(['RGIId'])

    data_id = pandas.read_csv(id_file, usecols=['RGIId', 'FoGId'], index_col=['FoGId'])

    glaciers = pandas.read_pickle('data/serialized/glaciers_climate')

    conv = data_id.loc[glaciers.index.values].dropna()

    rgi_ids = list(conv.values.flatten())

    conv['Slope'] = data.loc[rgi_ids]['Slope'].values
    conv['Aspect'] = data.loc[rgi_ids]['Aspect'].values
    glaciers[['slope', 'aspect']] = conv[['Slope', 'Aspect']]
