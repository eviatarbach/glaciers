import pandas
import numpy
import scipy.stats

from data import RGI_REGIONS

all_glaciers = pandas.read_pickle('data/serialized/all_glaciers').sortlevel()

MISSING_REGIONS = ['Alaska', 'SouthernAndes']

# These regions have correct mapping between RGI 5 and Huss, Alaska and Southern Andes don't
RGI_REGIONS2 = [r for r in RGI_REGIONS if r not in MISSING_REGIONS]

# We use RGI areas, since we have these correctly assigned for every glacier
areas = all_glaciers.loc[RGI_REGIONS2]['Area']
volumes = all_glaciers.loc[RGI_REGIONS2]['volume']

mask = ~numpy.isnan(areas) & ~numpy.isnan(volumes)

line = scipy.stats.linregress(numpy.log(areas)[mask], numpy.log(volumes)[mask])

# Calculate relative error in prediction
volumes_pred = numpy.exp(line.intercept)*areas**(line.slope)
relative_error = numpy.sqrt(numpy.mean(((volumes_pred - volumes)/volumes_pred)**2))
print(relative_error)

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

# Interpolate lengths
lengths = all_glaciers.loc[RGI_REGIONS2]['LENGTH']

mask = ~numpy.isnan(areas) & ~numpy.isnan(lengths)

line = scipy.stats.linregress(numpy.log(areas)[mask], numpy.log(lengths)[mask])

# Calculate relative error in prediction
lengths_pred = numpy.exp(line.intercept)*areas**(line.slope)
relative_error = numpy.sqrt(numpy.mean(((lengths_pred - lengths)/lengths_pred)**2))
print(relative_error)

# Interpolate missing lengths
missing_mask = all_glaciers['LENGTH'].isnull()
areas_missing = all_glaciers.loc[missing_mask, 'Area']
all_glaciers.loc[missing_mask, 'LENGTH'] = numpy.exp(line.intercept)*areas_missing**(line.slope)

# Mark all glaciers that had their volumes interpolated
all_glaciers['interp_length'] = False
all_glaciers.loc[missing_mask, 'interp_length'] = True

all_glaciers.to_pickle('data/serialized/all_glaciers')
