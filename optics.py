import numpy as np
import hapi as h
import pandas as pd
from config import *
import logging as l
from chemistry import begin
from chemicals import permittivity
import pygame

pygame.init()

l.basicConfig(

    filename='optics.log', 
    level=l.INFO,
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s' 
)

l.info('check')



isoto_numbers,  cas_numbers, permittivity_data, depolarization, polarizability_mol = begin()

Map_image = pygame.image.load(str(image_path))

def generate_albedo_map():
    scaled_image = pygame.transform.scale(Map_image, (200, 200))
    scaled_image = scaled_image.convert(24) 
    heightmap_array = pygame.surfarray.pixels3d(scaled_image)
    heightmap_array = np.dot(heightmap_array[..., :3], [0.299, 0.587, 0.114]) / 255.0  
    H,W = heightmap_array.shape
    latitudes = np.linspace(-np.pi/2, np.pi/2, W)
    lat_grid = np.repeat(latitudes[:, np.newaxis], H, axis=1)
    albedo_map = np.zeros((W, H), dtype=np.float32)


    water_mask = heightmap_array < 0.3
    land_mask = ~water_mask


    base_water = 0.07
    sea_ice = (1 - np.cos(lat_grid[water_mask])) * 0.65
    albedo_map[water_mask] = base_water + sea_ice
    base_land = 0.25
    polar_effect = (1 - np.cos(lat_grid[land_mask])) * 0.65
    height_effect = (heightmap_array[land_mask] - 0.4) * 0.1
    albedo_map[land_mask] = base_land + polar_effect + height_effect
    albedo_map = np.clip(albedo_map, 0, 0.8)
    return albedo_map

L_star, R_exoplanet, R_star, rotation_period_sec, axial_tilt, orbital_period_sec  = convert()

albedo_map = generate_albedo_map()
def calculate_temperature(aldebo, greenhouse):
    'only use bond aldebo for realistic results'
    return np.power(((1-aldebo)*L_star)/(16*np.pi*(R_exoplanet**2)*5.67e-8), 0.25)*greenhouse

t = calculate_temperature(albedo_map, 1)





def mole_fraction(PP):
    return PP/P

def partial_density(weight, mole):
    return (mole*weight)/(8.314*t)

def number_density(PP):
    return PP/(1.380649e-23*t)

def CM(n, polarizability, epsilon ):
    return (n*polarizability)/(3*epsilon)

def planckWNCM(v, T):
    v = v *100
    c = 3.0e8
    h = 6.62607015e-34
    kB = 1.380649e-23  
    return 100.0*np.pi*2.0*h*c*c*v*v*v/(np.exp(h*c*v/(kB*T))-1)

def get_abscoef(
    gas_name: str,
    wavenumbers: np.ndarray,
    temperature: float = 288.0,
    total_pressure: float = 1.0,
    self_pressure: float = 0.0,
    wavenumber_step: float = 0.01,
    gammaL: str = 'gamma_air',
    intensity_cutoff: float = 1e-19
    ):

    Cond = ('AND', ('BETWEEN', 'nu', min(wavenumbers), max(wavenumbers)),
                  ('>=', 'sw', intensity_cutoff))    
    try:
        h.select(gas_name, Conditions=Cond, DestinationTableName='filtered')
    except Exception:
        try:
            h.fetch(gas_name,isoto_numbers[gas_name],1,min(wavenumbers),max(wavenumbers), Parameters=['nu', 'Sw'])
        except Exception as e:
            l.exception(f"Failed to fetch data for {gas_name} (isotope {isoto_numbers[gas_name]}): {e}")
            h.fetch(gas_name,isoto_numbers[gas_name],1,min(wavenumbers),max(wavenumbers))
            Cond = ('AND', ('BETWEEN', 'nu', min(wavenumbers), max(wavenumbers)))
            h.select(gas_name, Conditions=Cond, DestinationTableName='filtered')
             

    alpha, nu = h.absorptionCoefficient_Voigt(
        SourceTables='filtered',
        OmegaGrid=wavenumbers,
        Environment={'T': temperature, 'p': total_pressure, 'p_self': self_pressure},
        WavenumberStep=wavenumber_step,
        GammaL=gammaL,
        HITRAN_units=False
    )
    return alpha, nu

def start():
    PP = {gas: data['P'] for gas, data in GAS_CONFIG.items()}
    mole_f = {}

    for compound, Pi in PP.items():
        mole_f[compound] = mole_fraction(Pi)
    PD = {}
    molar_masses = {gas: data['weight'] for gas, data in GAS_CONFIG.items()}
    for compound, mole in mole_f.items():
        PD[compound] = partial_density(molar_masses[compound],mole)
    ND = {}
    for compound, mole in mole_f.items():
        ND[compound] = number_density(PD[compound])
    N_total = sum(ND.values())
    epsilon = {}
    for compound, cas in cas_numbers.items():
        try:
            row = permittivity_data.loc[cas]
        
            A = float(row.at['A'])
            B = float(row.at['B'])
            C = float(row.at['C'])
            D_raw = row.get('D', np.nan)
            D = float(D_raw) if not pd.isna(D_raw) else 1
            # the constant D is not availble in this database. As such, We will put a placeholder here until I can empircally tune it.

            epsilon[cas] = permittivity.permittivity_CRC(t, A, B, C, D)
        except KeyError:
            default_value = 1.0005
            epsilon[cas] = default_value
            l.warning(f"No permittivity data found for {compound} (CAS {cas}). Please check if your cas number is valid. Using default value: {default_value}")
    LD = {}
    for compound, n in ND.items():
        LD[compound] = CM(n, polarizability_mol ,epsilon[cas_numbers[compound]] )

    common_keys = mole_f.keys() & LD.keys()
    L_total = {key: mole_f[key] * LD[key] for key in common_keys}

    L_total = sum(L_total.values()) 
    with np.errstate(invalid='raise', divide='raise'): 
        try:
            n = np.sqrt((1 + 2 * L_total) / (1 - L_total))
        except FloatingPointError:
            l.warning("Invalid value encountered in sqrt calculation for n. Attempting to Convert to real number.")
            n = (1 + 2 * L_total) / (1 - L_total)
            if np.any(np.isnan(n)) or np.any(np.isinf(n)):
                l.warning("n is NaN or Inf. Returning None.")
                n = None
            else:
                n = np.real(n)
    return N_total, L_total, PP, ND

def initialize_wavenumbers():
    wavenumbers = pd.DataFrame({"nu": np.linspace(50, 3500, 350000)})
    wavenumbers["inpower"]= wavenumbers.nu.apply(lambda x: planckWNCM(x, np.mean(t)))  
    return wavenumbers
