import os
from numpy import radians


"""citation for much of the planetary (earth) infomation: [1] NASA, "Earth Fact Sheet," NASA Solar System Exploration, Apr. 22, 2024.
 [Online]. Available: https://nssdc.gsfc.nasa.gov/planetary/factsheet/earthfact.html """


image_path = os.path.expanduser(r"42.png")
BASIS = 'aug-cc-pVDZ'
UNITS = "Angstrom"
molar_mass = 28.97 #molar mass of the atmosphere
WIDTH, HEIGHT = 800, 600
caption = 'test'
alpha = 200 
detla= 1
P = 1014  #atmosphereic pressure in mb
L_star = 1
R_star = 1
R_exoplanet = 1
rotation_period = 24
exoplanet_R = 1
G = 9.82
axial_tilt_degrees = 23.44
orbital_period_days = 365.256
map_res = 200
eccentricity = 0.0167

#L_star = 0.24
# Lsun
#R_star = 0.752
# Rsun
#R_exoplanet = 0.67
# AU
#rotation_period = 20
# hours
#exoplanet_R = 1.355*6371
#Rearth
#G = 16.024
#axial_tilt_degrees = 10
#orbital_period_days = 238.410




#in this version of the code, all we need to do is add any gas that you want to be factored into calculations in the format below and the program will consider it. However, 
# consider that may not be able to handle some gases. We have tried to compensate for that, but accuracy can still be effected and glitches for nonstandard gases may still be present.
#O2 is an example. 

GAS_CONFIG= {
    'N2O' : {
        'atom' : 'N 0 0 0; N 0 0 1.128; O 0 0 2.313',  
        'basis' : BASIS,     
        'unit' : UNITS,
        'cas_number' : '10024-97-2',
        'I' : 4,  #HITRAN isotopologue number
        'P' : 0.000337  , # partial pressure in mb
        'weight' : 44.013,
    },

    'CH4' : {
        'atom' : 'C 0.000 0.000 0.000; H 0.629 0.629 0.629; H -0.629 -0.629 0.629; H -0.629 0.629 -0.629; H 0.629 -0.629 -0.629',  
        'basis' : BASIS,
        'unit' : UNITS,
        'cas_number' : '74-82-8',
        'I' : 6,
        'P' : 0.0020,
        'weight' : 16.04, 
    },

    'O3' :{
        'atom' : 'O 0.000 0.000 0.000; O 1.278 0.000 0.000; O -0.580 1.139 0.000',  
        'basis' : BASIS, 
        'unit' : UNITS,
        'cas_number': '10028-15-6',
        'I' : 1,
        'P' : 0.000005,
        'weight' : 48
    },

    'CO2' : {
        'atom' : 'O 0.000 0.000  1.160; C 0.000 0.000  0.000; O 0.000 0.000 -1.160',  
        'basis' : BASIS, 
        'unit' : UNITS,
        'cas_number': '124-38-9',
        'I' : 2,
        'P' : 0.43,
        'weight' : 44.009,
    },

    'H2O' : {
        'atom' : 'O 0.000 0.000 0.000; H 0.757 0.000 0.586; H -0.757 0.000 0.586',  
        'basis' : BASIS, 
        'unit' : UNITS,
        'cas_number': '7732-18-5',
        'I' : 1,
        'P' : 10.14, 
        'weight' : 18.01528
    },

    'N2' : {
        'atom': 'N -0.37 0.00 0.00; N 0.37 0.00 0.00',
        'basis' : BASIS, 
        'unit' : UNITS,
        'cas_number': '7727-37-9',
        'I' : 1,
        'P' : 792.7,
        'weight' :  28.0134
    },

    'O2' : {
        'atom' : ' O -0.605 0.00 0.00; O 0.605 0.00 0.00',  
        'basis' : BASIS, 
        'unit' : UNITS,
        'cas_number': '7782-44-7',
        'I' : 1,
        'P' : 212.4,
        'weight' : 31.9988 
    },
    
                }




'______UNEDITABLE______'

L_sun_watts = 3.828e26  
AU_to_meters = 1.496e11 
R_sun_meters = 6.963e8
initial_rotation_angle = 0
declination=0
R_earth_km = 6378 #[1] NASA, "Earth's Shape – Imagine the Universe!," NASA Goddard Space Flight Center. [Online]. Available: https://imagine.gsfc.nasa.gov/features/cosmic/earth_info.html. [Accessed: Apr. 24, 2025].


def convert():
    global rotation_period_sec, axial_tilt,L_star, L_sun_watts, R_exoplanet, AU_to_meters, R_star, R_sun_meters, axial_tilt_degrees, rotation_period, detla, orbital_period_days
    L_star = L_star * L_sun_watts
    R_exoplanet = R_exoplanet * AU_to_meters
    R_star = R_star * R_sun_meters
    rotation_period_sec = rotation_period*3600
    axial_tilt = radians(axial_tilt_degrees)
    rotation_period = rotation_period/detla
    orbital_period_days = orbital_period_days/detla
    orbital_period_sec = orbital_period_days*86400
    return L_star, R_exoplanet, R_star, rotation_period_sec, axial_tilt, orbital_period_sec  

