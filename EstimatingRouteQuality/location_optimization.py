# this is a script that finds an optimal position in the US according to "Classic Energy",
# calculated using the potential: -route_quality/(dist^2) where dist is the distance from the current position
# to the route sector, this is analagous to electrostatic or gravitational interaction energy.
# The energy wells created by each route will superimpose on one another, creating wells of 
# optimal positions.

# Did not make the final blog post, I will work on it more maybe for a future post/app.
# - speed up for practical use
# - check reliability: does it always give the same result?

import numpy as np
import pandas as pd
from grade_rank_calculation import calculate_grade_rank
from mpu import haversine_distance
from scipy.optimize import differential_evolution
import plotly.graph_objects as go

class location_optimizer(object):

    def __init__(self, df, metric='ARQI_median', route_type='all', grade_range='all'):

        if route_type != 'all':
            df = df[df['type_string'] == route_type].copy()
            
        if grade_range != 'all':
            lo,hi = grade_range.split('-')
            lo_rank = calculate_grade_rank(lo)
            hi_rank = calculate_grade_rank(hi)                
            df = df[(lo_rank <= df['YDS_rank']) & (df['YDS_rank'] <= hi_rank)].copy()

        df['parent_loc'] = df.apply(lambda row: np.array(row['parent_loc'][::-1]), axis=1)

        self.quals = df[metric]
        self.locs = df['parent_loc']
        
    def energy(self, quality, loc0, loc1, eps=1.0e-4):
    
        dist = haversine_distance(loc0, loc1) + eps
        return -quality/(dist*dist)
    
    def total_energy(self, loc):
        
        TE = 0.0

        for qual, pos in zip(self.quals, self.locs):
            TE += self.energy(qual, pos, loc)

        return TE

    def optimize(self):

        opt_data = []
        def callbackF(xk, convergence=100.0):
            cTE = self.total_energy(xk)
            opt_data.append([xk[0], xk[1], cTE])

        bounds = [(36.5,49), (-123.9157,-69.2246)]
        res = differential_evolution(self.total_energy, bounds, polish=True, disp=True, 
            strategy='randtobest1bin', popsize=20, maxiter=500, callback=callbackF)
        
        return res, opt_data

    def run(self, plot_results=True):

        res, opt_data = self.optimize()
        opt_data = [[l[0], l[1], l[2], 0.0] for l in opt_data]
        final = list(res.x) + [self.total_energy(res.x), 1.0]
        opt_data.append(final)
        res_df = pd.DataFrame(opt_data, columns=['lat', 'lon', 'TE', 'type'])

        if plot_results:

            data = go.Scattermapbox(
                        lat=res_df['lat'],
                        lon=res_df['lon'],
                        mode='markers',
                        marker=dict(color=res_df['type']))
    
            layout = dict(margin=dict(l=0, t=0, r=0, b=0, pad=0),
                          mapbox=dict(center=dict(lat=39,lon=-95),
                                        style='carto-darkmatter',
                                        zoom=3.5),
                          geo=dict(scope='usa',
                                   projection_type='albers usa',
                                   resolution=110))
    
            fig = go.Figure(data=data, layout=layout)
            fig.write_html('test.html')


if __name__ == '__main__':
 
    df = pd.read_pickle('RouteQualityData.pkl.zip', compression='zip')
    LO = location_optimizer(df, grade_range='5.12a-5.13a', route_type='sport')
    LO.run()
