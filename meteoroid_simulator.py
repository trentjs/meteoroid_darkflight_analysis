"""
=============== Meteoroid Simulator ===============
Created on Mon Feb 07 17:13:05 2019
@author: Trent Jansen-Sturgeon

Simulate a meteoroid using the equations of motion and appropriate errors.
Inputs: Requires a config file with meteoroid and viewing parameters.
Output: The simulated observation data in the standard ecsv format.

"""

# Import modules
import configparser
import argparse
import os
import ast
import copy

# Import science modules
import numpy as np
from numpy.linalg import norm
from astropy.table import Table
from astropy.time import Time
from scipy.integrate import ode
import astropy.units as u

# Import custom modules
from DarkFlightTrent import EarthDynamics
from trajectory_utilities import LLH2ECEF, ECEF2LLH, EarthRadius, \
    ECEF2ECI, ECI2ECEF, ENU2ECEF, ECI2ECEF_pos, ECEF2ENU
from CSV2KML import Rays, Path, Points#, merge_trajectory_KMLs

__author__ = "Trent Jansen-Sturgeon"
__copyright__ = "Copyright 2019, Desert Fireball Network"
__license__ = ""
__version__ = "1.0"
__scriptName__ = "meteoroid_simulator.py"

class NotVisibleError(Exception):
    ''' Raised when any observatory cannot see the simulated event.'''
    pass

def generate_random_config(directory):

    # Create the configuration file as it doesn't exist yet
    cfg_filename = os.path.join(directory, 'random_trajectory.cfg')
    if os.path.isfile(cfg_filename):
        print("A random trajectory already exists. We'll use that config.")
        return cfg_filename

    # Add content to the file
    Config = configparser.ConfigParser()
    Config.add_section('identity')
    Config.set('identity', 'name', 'random_trajectory')

    Config.add_section('dynamic_properties')
    Config.set('dynamic_properties', 'initial_datetime', '2000-01-01T12:00:00.000')
    Config.set('dynamic_properties', 'initial_latitude', str(0)) #deg
    Config.set('dynamic_properties', 'initial_longitude', str(0)) #deg
    Config.set('dynamic_properties', 'initial_height', str(100e3)) #m
    Config.set('dynamic_properties', 'initial_velocity', str(np.random.uniform(12e3, 72e3))) #m/s
    Config.set('dynamic_properties', 'initial_slope', str(np.random.uniform(10, 90))) #deg
    Config.set('dynamic_properties', 'initial_bearing', str(np.random.uniform(0, 360))) #deg

    Config.add_section('physical_properties')
    Config.set('physical_properties', 'initial_mass', str(10**(np.random.uniform(-1, 2)))) #kg
    Config.set('physical_properties', 'density', str(3500)) #kg/m3
    Config.set('physical_properties', 'shape', str(1.21))
    Config.set('physical_properties', 'ablation_coeff', str(1e-8))
    Config.set('physical_properties', 'spin_state', str(0.66666))
    Config.set('physical_properties', 'luminous_efficiency', str(0.0001))

    # if np.random.uniform(0,10) < 1: # One-in-ten chance of fragmentation
    #     Config.set('physical_properties', 'mass_loss_height', str([np.random.uniform(30e3, 360)])) #m
    #     Config.set('physical_properties', 'mass_loss_percent', str([np.random.uniform(30, 90)])) #deg
    # else:
    Config.set('physical_properties', 'mass_loss_height', str([])) #m
    Config.set('physical_properties', 'mass_loss_percent', str([])) #deg

    # if np.random.uniform(0,10) < 1: # One-in-ten chance of 3 station event
    #     n = 3
    # else:
    n = 2

    Config.add_section('observatory_properties')
    Config.set('observatory_properties', 'measurement_uncertainty', str(0.04)) #deg
    Config.set('observatory_properties', 'picking_uncertainty', str(0.04)) #deg
    Config.set('observatory_properties', 'obs_name', str(['observatory_'+str(i) for i in range(n)]))
    Config.set('observatory_properties', 'obs_location', str(['location_'+str(i) for i in range(n)]))
    Config.set('observatory_properties', 'obs_latitude', str([0]*n)) #deg <----- to be determined later < 300 km from center of trajectory
    Config.set('observatory_properties', 'obs_longitude', str([0]*n)) #deg <---- to be determined later < 300 km from center of trajectory
    Config.set('observatory_properties', 'obs_height', str([0]*n)) #m
    Config.set('observatory_properties', 'obs_timing_offset', str([0]*n)) #sec
    Config.set('observatory_properties', 'obs_calibration_offset', str([0]*n)) #sec

    # Write config to file
    with open(cfg_filename, 'w') as cfg_file:
        Config.write(cfg_file)

    return cfg_filename

def read_config(ifile):

    Config = configparser.RawConfigParser()
    Config.read(ifile); traj_dict = {}; obs_dict = {}
    extract = lambda header, name: ast.literal_eval(Config.get(header, name))

    # Event identity
    event_name = Config.get('identity', 'name')

    # trajectory conditions
    t0_isot = Config.get('dynamic_properties', 'initial_datetime')
    traj_dict['t0'] = Time(t0_isot, format='isot', scale='utc').jd # time

    lat0 = np.deg2rad(Config.getfloat('dynamic_properties', 'initial_latitude'))
    lon0 = np.deg2rad(Config.getfloat('dynamic_properties', 'initial_longitude'))
    hei0 = Config.getfloat('dynamic_properties', 'initial_height')
    pos_ecef = LLH2ECEF(np.vstack((lat0, lon0 , hei0)))

    speed0 = Config.getfloat('dynamic_properties', 'initial_velocity')
    slope0 = np.deg2rad(Config.getfloat('dynamic_properties', 'initial_slope'))
    bearing0 = np.deg2rad(Config.getfloat('dynamic_properties', 'initial_bearing'))
    vel_enu = -speed0 * np.vstack((np.sin(bearing0) * np.cos(slope0), 
        np.cos(bearing0) * np.cos(slope0), np.sin(slope0)))
    vel_ecef = ENU2ECEF(lon0, lat0).dot(vel_enu)
    
    [pos_eci, vel_eci] = ECEF2ECI(pos_ecef, vel_ecef, traj_dict['t0'])
    traj_dict['pos_eci'] = pos_eci.flatten() # position
    traj_dict['vel_eci'] = vel_eci.flatten() # velocity

    # Physical conditions
    traj_dict['m0'] = Config.getfloat('physical_properties', 'initial_mass') # mass
    traj_dict['rho'] = Config.getfloat('physical_properties', 'density') # density
    traj_dict['A'] = Config.getfloat('physical_properties', 'shape') # shape
    traj_dict['c_ml'] = Config.getfloat('physical_properties', 'ablation_coeff') # ablation coefficient
    traj_dict['mu'] = Config.getfloat('physical_properties', 'spin_state') # spin state
    traj_dict['tau'] = Config.getfloat('physical_properties', 'luminous_efficiency') # luminous efficiency
    traj_dict['dm_height'] = extract('physical_properties', 'mass_loss_height') # mass loss height [n]
    traj_dict['dm_percent'] = extract('physical_properties', 'mass_loss_percent') # mass loss percent [n]

    # Observatory conditions
    obs_dict['obs_name'] = extract('observatory_properties', 'obs_name') # observatory names [m]
    obs_dict['obs_location'] = extract('observatory_properties', 'obs_location') # observatory locations [m]
    obs_lat = np.deg2rad(extract('observatory_properties', 'obs_latitude'))
    obs_lon = np.deg2rad(extract('observatory_properties', 'obs_longitude'))
    obs_hei = np.array(extract('observatory_properties', 'obs_height'))
    obs_dict['obs_llh'] = np.vstack((obs_lat, obs_lon, obs_hei)).T # observatory locations [m,3]
    obs_dict['obs_dt'] = np.array(extract('observatory_properties', 'obs_timing_offset')) # timing offsets [m]
    obs_dict['obs_dang'] = np.array(extract('observatory_properties', 'obs_calibration_offset')) # calibration offsets [m]
    obs_dict['measurement_err'] = Config.getfloat('observatory_properties', 'measurement_uncertainty') # Measurement uncertainty for simulation
    obs_dict['picking_err'] = Config.getfloat('observatory_properties', 'picking_uncertainty') # Picking uncertainty from point-picker

    return event_name, traj_dict, obs_dict

def continue_condition(state_i):

    # Define the rough ground level
    lat = ECEF2LLH(np.vstack((state_i[:3])))[0]
    R_sealevel = EarthRadius(lat)
    r_end = R_sealevel #+ h_ground
    r_beg = R_sealevel + 120e3

    # brightness = fn()
    # if brightness < threshold: # Not luminous anymore
    if norm(state_i[3:6]) < 2e3: # Get a better measure
        print('Successfully found the end of the trajectory: v < 2km/s.')
        return False # Ends the integration
    elif norm(state_i[:3]) > r_beg: # Escaped the atmosphere
        print('Successfully found the start of the trajectory: h > 120km.')
        return False # Ends the integration
    elif norm(state_i[:3]) < r_end: # Reached ground
        print('Your meteoroid became a meteorite!!')
        return False # Ends the integration
    elif state_i[6] < 1e-3: # Lost all mass [<1g]
        print('Your meteoroid is dust!')
        return False # Ends the integration
    else:
        return True # Continues integration

def integrate(t_rel0, state0, dt, args):
    [WindData, t0, dm_height, dm_percent] = args

    # Setup integrator
    solver = ode(EarthDynamics).set_integrator('dopri5', \
        first_step=np.sign(dt)*0.01, max_step=np.sign(dt)*1, rtol=1e-4) #'dop853', 
    solver.set_f_params(WindData, t0)
    solver.set_initial_value(state0, t_rel0)

    # Integrate with RK4 until its high enough
    state = [state0]; t_rel = [t_rel0]
    hei_old = ECEF2LLH(np.vstack((state0[:3])))[2]
    while continue_condition(state[-1]):

        # Check for fragmentation event
        hei_new = ECEF2LLH(np.vstack((state[-1][:3])))[2]
        frag_condition = (dm_height > min(hei_old, hei_new)) \
            & (dm_height < max(hei_old, hei_new))

        if np.any(frag_condition): # We got fragmentation!
            heights = dm_height[frag_condition]
            percentages = dm_percent[frag_condition]

            # Determine the rough times of fragmentation (first order)
            ratio = (heights - hei_old) / (hei_new - hei_old)
            rough_times = t_rel[-2] + ratio * (t_rel[-1] - t_rel[-2])

            # Remove the last state
            t_last = t_rel[-1]
            [t_rel, state] = [t_rel[:-1], state[:-1]]

            # Integrate through the fragmentation events
            solver.set_initial_value(state[-1], t_rel[-1])
            for t, p in zip(rough_times, percentages):

                t_rel.append( t )
                state.append( solver.integrate(t) )

                # Fragment!
                state_after = state[-1].copy()
                state_after[6] *= 1 - np.sign(dt) * p / 100
                t_rel.append( t ); state.append( state_after )
                solver.set_initial_value(state[-1], t_rel[-1])

            # Integrate back to last state
            t_rel.append( t_last )
            state.append( solver.integrate(t_rel[-1]) )

        # Integrate to the next time
        t_rel.append( t_rel[-1] + dt ); hei_old = hei_new
        state.append( solver.integrate(t_rel[-1]) )

    return t_rel, state

def generate_trajectory(traj_dict):

    WindData = Table(); dt = 0.1

    # Calculate the meteor's initial state parameters
    t0 = traj_dict['t0']; t_rel0 = 0
    state0 = np.hstack((traj_dict['pos_eci'], traj_dict['vel_eci'], 
        traj_dict['m0'], traj_dict['rho'], traj_dict['A'], 
        traj_dict['c_ml'], traj_dict['tau'])) #[11]

    # Define the fragmentation heights/percentages
    dm_height = np.array(traj_dict['dm_height'])
    dm_percent = np.array(traj_dict['dm_percent'])

    args = [WindData, t0, dm_height, dm_percent]

    # Firstly integrate forwards
    [t_rel_f, state_f] = integrate(t_rel0, state0, +dt, args)

    # Secondly integrate backward
    [t_rel_b, state_b] = integrate(t_rel0, state0, -dt, args)

    # Package up the variables
    t_rel = np.array(t_rel_b[::-1] + t_rel_f[1:])
    t_jd = t_rel / (24*60*60) + t0 #[n]
    state = np.array(state_b[::-1] + state_f[1:]) #[n,10]

    # Calculate the luminosity
    abs_mag = np.zeros(len(t_jd))
    for i, (t, x) in enumerate(zip(t_rel, state)):
        abs_mag[i] = EarthDynamics(t, x, WindData, t0, True)

    print('relative_time,   height, velocity,   mass, absolute_mag')
    for t,h,v,m,ab in zip(t_rel, ECEF2LLH(state[:,:3].T)[2]/1000, \
        norm(state[:,3:6], axis=1)/1000, state[:,6], abs_mag):
        print('{0:13.2f},{1:9.2f},{2:9.1f},{3:7.2f},{4:13.3f}'.format(t,h,v,m,ab))

    return t_jd, state, abs_mag

# def debruin_filter(t_jd):
#     # Find the observations that match the debruin sequence

#     debruin_str = ('000000000111111111011111110011111101011111100011111011011111010011111'
#         +'001011111000011110111011110110011110101011110100011110011011110010011110001011'
#         +'110000011101110011101101011101100011101011011101010011101001011101000011100110'
#         +'011100101011100100011100011011100010011100001011100000011011011010011011001011'
#         +'011000011010110011010101011010100011010010011010001011010000011001100011001010'
#         +'011001001011001000011000101011000100011000010011000001011000000010101010010101'
#         +'00001010010001010001001010000001001001000001000100001')
#     debruin_seq = np.array([int(i) for i in debruin_str])

#     debruin_spacing = np.array([])
#     debruin_time = 

#     return cond1, db_index

def random_observatory(t_jd, state, viewing_limit):

    traj_center_eci = np.vstack(np.median(state[:,:3], axis=0))
    traj_center_ecef = ECI2ECEF_pos(traj_center_eci, np.median(t_jd))
    traj_center_llh = ECEF2LLH(traj_center_ecef)

    # Ground distance limit from center of trajectory
    R_e = EarthRadius(traj_center_llh[0])
    obs_range = R_e * (np.pi/2 - viewing_limit 
        - np.arcsin(R_e / (R_e + traj_center_llh[2]) * np.cos(viewing_limit)))

    # Generate random observer location
    r = np.sqrt(np.random.uniform(0,obs_range**2))
    az = np.random.uniform(0, 2*np.pi)
    obs_enu = np.vstack((r*np.sin(az), r*np.cos(az), 0))

    # Transform to llh coords
    obs_center_llh = traj_center_llh; obs_center_llh[2] = 0
    obs_center_ecef = LLH2ECEF(obs_center_llh)
    obs_ecef = ENU2ECEF(traj_center_llh[1], traj_center_llh[0]).dot(obs_enu) + obs_center_ecef
    obs_llh = ECEF2LLH(obs_ecef).flatten(); obs_llh[2] = 0

    return obs_llh

def generate_observations(t_jd, state, abs_mag, obs_llh, obs_dict, app_threshold, alt_threshold, t0):

    if np.all(obs_llh == 0): # Generate random observation locations - not saved in the cfg_file at the moment
        viewing_limit = np.deg2rad(20) # Rough observatory LOS elevation limit
        obs_llh = random_observatory(t_jd, state, viewing_limit)

    # Filter out any additional fragmentation data
    [t_unique, counts] = np.unique(t_jd, return_counts=True)
    cond1 = [True if t not in t_unique[counts>1] else False for t in t_jd]
    t_jd = t_jd[cond1]; state = state[cond1]
    abs_mag = abs_mag[cond1]

    # Filter out based on debruin viewability
    # [cond1, db_index] = debruin_filter(t_jd)
    db_index = np.arange(len(t_jd))*2 + 1

    # Convert the remaining positions into local ENU coords
    pos_ecef = ECI2ECEF_pos(state[:,:3].T, t_jd) #[3,n]
    obs_ecef = LLH2ECEF(np.vstack((obs_llh))) #[3,1]
    pos_enu = ECEF2ENU(obs_llh[1], obs_llh[0]).dot(pos_ecef - obs_ecef) #[3,n]

    # Filter out based on apparent magnitude
    d = norm(pos_ecef - obs_ecef, axis=0) #[n]
    app_mag = abs_mag - 5 * np.log10(d / 100e3) #[n]
    cond2 = (app_mag < app_threshold)
    pos_enu = pos_enu[:,cond2] #[3,n-]
    t_jd = t_jd[cond2]; db_index = db_index[cond2] #[n-]

    # Convert to spherical coords
    altaz = np.vstack((np.arcsin(pos_enu[2] / norm(pos_enu, axis=0)),
        np.arctan2(pos_enu[0], pos_enu[1]))).T #[n-,2]

    # Filter out based on angle to horizon
    cond3 = (altaz[:,0] > alt_threshold)
    altaz = altaz[cond3] #[n--,2]
    t_jd = t_jd[cond3]; db_index = db_index[cond3] #[n--]

    if not np.any(cond3):
        print('This observatory cannot see the meteoroid event.')
        raise NotVisibleError

    print('\nrelative_time, height, altitude, absolute_mag, apparent_mag')
    for t,h,alt,ab,ap in zip((t_jd - t0) *24*60*60, ECEF2LLH(pos_ecef[:,cond2][:,cond3])[2]/1000, \
        np.rad2deg(altaz[:,0]), abs_mag[cond2][cond3], app_mag[cond2][cond3]):
        print('{0:13.2f},{1:7.2f},{2:9.2f},{3:13.3f},{4:13.3f}'.format(t,h,alt,ab,ap))

    # # Give the measurements some uncertainty
    altaz += np.random.normal(0, np.deg2rad(obs_dict['measurement_err']), altaz.shape) #[n--,2]
    altaz_err = np.ones(altaz.shape) * np.deg2rad(obs_dict['picking_err']) #[n--,2]
    
    return t_jd, db_index, altaz, altaz_err, obs_llh #[n],[n],[n,2],[n,2],[3]

def write_trajectory_file(t_jd, state, ofile, kml): #[n],[n,10]

    traj_table = Table()
    traj_table['datetime'] = Time(t_jd, format='jd', scale='utc').isot

    [pos_ecef, vel_ecef] = ECI2ECEF(state[:,:3].T, state[:,3:6].T, t_jd)
    pos_llh = ECEF2LLH(pos_ecef)
    traj_table['latitude'] = np.rad2deg(pos_llh[0]) * u.deg
    traj_table['longitude'] = np.rad2deg(pos_llh[1]) * u.deg
    traj_table['height'] = pos_llh[2] * u.m
    
    traj_table['X_geo'] = pos_ecef[0] * u.m
    traj_table['Y_geo'] = pos_ecef[1] * u.m
    traj_table['Z_geo'] = pos_ecef[2] * u.m
    traj_table['DX_DT_geo'] = vel_ecef[0] * u.m / u.second
    traj_table['DY_DT_geo'] = vel_ecef[1] * u.m / u.second
    traj_table['DZ_DT_geo'] = vel_ecef[2] * u.m / u.second
    traj_table['D_DT_geo'] = norm(vel_ecef, axis=0) * u.m / u.second

    traj_table['mass'] = state[:,6] * u.kg
    traj_table['rho'] = state[:,7] * u.kg / u.m**3
    traj_table['shape'] = state[:,8]
    traj_table['c_ml'] = state[:,9]

    traj_table.write(ofile, format='ascii.csv', delimiter=',', fast_writer=False)

    # Make a KML of the trajectory
    if kml: 
        Path(ofile)
        Points(ofile, np.round(traj_table['mass'],3), colour='ff1400ff') # red points

def write_observation_file(t_jd, db_index, altaz, altaz_err, obs_llh, ofile, kml, event_name, obs_name, obs_loc): #[n],[n,2],[n,2],[3]

    obs_table = Table()
    obs_table.meta['event_codename'] = event_name
    obs_table.meta['telescope'] = obs_name
    obs_table.meta['location'] = obs_loc
    obs_table.meta['isodate_start_obs'] = Time(t_jd[0], format='jd', scale='utc').isot

    obs_table.meta['obs_latitude'] = np.rad2deg(obs_llh[0])
    obs_table.meta['obs_longitude'] = np.rad2deg(obs_llh[1])
    obs_table.meta['obs_elevation'] = obs_llh[2]

    obs_table['datetime'] = Time(t_jd, format='jd', scale='utc').isot
    obs_table['time_err_minus'] = 6.5e-4 * u.second
    obs_table['time_err_plus'] = 5e-5 * u.second
    obs_table['de_bruijn_sequence_element_index'] = db_index
    obs_table['dash_start_end'] = ['E' if i%2==0 else 'S' for i in db_index]

    obs_table['altitude'] = np.rad2deg(altaz[:,0]) * u.deg
    obs_table['err_minus_altitude'] = np.rad2deg(altaz_err[:,0]) * u.deg
    obs_table['err_plus_altitude'] = np.rad2deg(altaz_err[:,0]) * u.deg
    obs_table['azimuth'] = np.rad2deg(altaz[:,1]) * u.deg
    obs_table['err_minus_azimuth'] = np.rad2deg(altaz_err[:,1]) * u.deg
    obs_table['err_plus_azimuth'] = np.rad2deg(altaz_err[:,1]) * u.deg

    obs_table.write(ofile, format='ascii.ecsv', delimiter=',')

    # Make a KML of the trajectory
    if kml: Rays(ofile)


def main(directory, input_file=None, kml=True):

    if directory and os.path.isdir(directory): # Create a random trajectory
        input_file = generate_random_config(directory)

    elif input_file and os.path.isfile(input_file): # Determine the directory
        directory = os.path.dirname(input_file)
        
    else: # Chuck an error
        print("\nSorry, but this file/directory does not exist."); exit()

    # Read the config file into dictionaries
    [event_name, traj_dict, obs_dict] = read_config(input_file)

    # Use the dynamic equations to determine the ideal trajectory
    [t_jd, state, abs_mag] = generate_trajectory(traj_dict) #[n],[n,11]

    # Write the trajectory information to file (and produce ray kmls)
    ofile = os.path.join(directory, event_name + '_trajectory.csv')
    write_trajectory_file(t_jd, state, ofile, kml)

    # Write files
    app_threshold = 1.5; alt_threshold = np.deg2rad(5); t0 = traj_dict['t0']
    for i, obs_llh in enumerate(obs_dict['obs_llh']):

        # Include uncertainties to determine what might have been seen
        [t_obs, db_index, altaz, altaz_err, obs_llh] = generate_observations(t_jd, 
            state, abs_mag, obs_llh, obs_dict, app_threshold, alt_threshold, t0)

        # Write the pointing information to file (and produce ray kmls)
        obs_name = obs_dict['obs_name'][i]; obs_loc = obs_dict['obs_location'][i]
        ofile = os.path.join(directory, event_name + '_from_' + obs_name + '.ecsv')
        write_observation_file(t_obs, db_index, altaz, altaz_err, 
            obs_llh, ofile, kml, event_name, obs_name, obs_loc)

###################################################################################
if __name__ == '__main__':
    '''
    Code to simulate a meteor's observations.
    Inputs: One config file with the meteorite's properties
    Outputs: Multiple observation files (.ecsv)
    '''

    # Gather some user defined information
    parser = argparse.ArgumentParser(description='Meteoroid Simulator')
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("-f", "--configFile", type=str,
        help="The config file containing the meteoroid properties to simulate [.cfg]")
    input_group.add_argument("-d", "--outputDirectory", type=str,
        help="The output directory to write the randomised config file.")
    parser.add_argument("-k", "--kml", action="store_false", default=True,
        help="use this option if you don't want to generate KMLs")
    args = parser.parse_args()

    input_file = args.configFile
    directory = args.outputDirectory
    kml = args.kml

    main(directory, input_file, kml)

    
