import pandas
import numpy
import scipy.stats
from sklearn.model_selection import train_test_split

from data import RGI_REGIONS

all_glaciers = pandas.read_pickle('data/serialized/all_glaciers').sortlevel()

MISSING_REGIONS = ['Alaska', 'SouthernAndes']

# These regions have correct mapping between RGI 5 and Huss, Alaska and Southern Andes don't
RGI_REGIONS2 = [r for r in RGI_REGIONS if r not in MISSING_REGIONS]

# We use RGI areas, since we have these correctly assigned for every glacier
areas = all_glaciers.loc[RGI_REGIONS2]['Area']
volumes = all_glaciers.loc[RGI_REGIONS2]['volume']

mask = ~numpy.isnan(areas) & ~numpy.isnan(volumes)

volumes_train, volumes_test, areas_train, areas_test = train_test_split(volumes[mask],
                                                                        areas[mask],
                                                                        test_size=0.10)

line_train = scipy.stats.linregress(numpy.log(areas_train), numpy.log(volumes_train))

# Calculate relative error in prediction
volumes_pred = numpy.exp(line_train.intercept)*areas_test**(line_train.slope)
relative_error = numpy.sqrt(numpy.mean(((volumes_pred - volumes_test)/volumes_pred)**2))
print(relative_error)

line = scipy.stats.linregress(numpy.log(areas[mask]), numpy.log(volumes[mask]))
volumes_interp = numpy.exp(line.intercept)*all_glaciers.loc[MISSING_REGIONS]['Area']**(line.slope)
all_glaciers.loc[MISSING_REGIONS, 'volume'] = volumes_interp.values

# Also interpolate volumes missing for other reasons
missing_mask = all_glaciers['volume'].isnull()
areas_missing = all_glaciers.loc[missing_mask, 'Area']
all_glaciers.loc[missing_mask, 'volume'] = numpy.exp(line.intercept)*areas_missing**(line.slope)

# Mark all glaciers that had their volumes interpolated
all_glaciers['interp_volume'] = False
all_glaciers.loc[MISSING_REGIONS, 'interp_volume'] = True
all_glaciers.loc[missing_mask, 'interp_volume'] = True

# Set quantities for mismatched regions to the RGI quantities
all_glaciers.loc[MISSING_REGIONS, 'area'] = all_glaciers['Area']
all_glaciers.loc[MISSING_REGIONS, 'SLOPE_avg'] = all_glaciers['Slope']

# Fill in null from RGI when missing from Huss data
all_glaciers.loc[all_glaciers['SLOPE_avg'].isnull(), 'SLOPE_avg'] = all_glaciers['Slope']
all_glaciers.loc[all_glaciers['area'].isnull(), 'area'] = all_glaciers['Area']

# Fill in thickness
all_glaciers.loc[all_glaciers['THICK_mean'].isnull(),
                 'THICK_mean'] = all_glaciers['volume']/all_glaciers['area']

all_glaciers.to_pickle('data/serialized/all_glaciers')
