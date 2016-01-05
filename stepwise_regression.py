import pandas
import statsmodels.formula.api as smf

def forward_selected(data, response):
    '''
    Based roughly on http://planspace.org/20150423-forward_selection_with_statsmodels/
    '''
    remaining = set(data.columns)
    remaining.remove(response)
    selected = []
    while remaining:
        scores_with_candidates = []
        for candidate in remaining:
            formula = "{} ~ {} + 1".format(response,
                                           ' + '.join(selected + [candidate]))
            score = smf.ols(formula, data).fit().pvalues[candidate]
            scores_with_candidates.append((score, candidate))
        scores_with_candidates = sorted(scores_with_candidates)[::-1]
        best_new_score, best_candidate = scores_with_candidates.pop()
        if best_new_score < 0.15:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
        else:
            break
    formula = "{} ~ {} + 1".format(response,
                                   ' + '.join(selected))
    model = smf.ols(formula, data).fit()
    return model

glaciers = pandas.read_pickle('glaciers')

forward_selected(glaciers[['max_elevation', 'median_elevation', 'g', 'continentality', 'summer_temperature', 'precipitation', 'winter_precipitation', 'cloud_cover']], 'g')
