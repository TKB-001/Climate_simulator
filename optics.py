import numpy as np
import hapi as h
import pandas as pd
from config import *
import logging as l
l.getLogger('numba').setLevel(l.WARNING)
from chemistry import begin
from chemicals import permittivity
import pygame
from numpy.polynomial.legendre import leggauss
from scipy.optimize import minimize
import numba


pygame.init()

l.basicConfig(filename='optics.log', level=l.DEBUG, force=True)

l.info('check')



isoto_numbers,  cas_numbers, permittivity_data, depolarization, polarizability_mol = begin()

try:
    Map_image = pygame.image.load(str(image_path))
except NameError:
    raise RuntimeError("image_path is not defined in the config module. Please define 'image_path' in config.py.")

global_context = {}

def init_worker(context):
    global global_context
    global_context = context

@numba.njit(parallel=True, fastmath=True)
def integrate_map(
    s_nodes, w_s,  phi_nodes, v_phi,
    slope, aspect,
    f_iso, f_vol, f_geo
):
    H, W = slope.shape
    out = np.zeros((H, W), dtype=np.float32)

    pi_o_4 = np.pi/4.0

    for i in numba.prange(H):
        for j in range(W):

            pix_slope  = slope[i, j]
            pix_aspect = aspect[i, j]

            acc = 0.0
            for si in range(s_nodes.shape[0]):
                θ1 = s_nodes[si]
                ws = w_s[si]
                cos_ts = np.cos(θ1)  
                sin_ts = np.sin(θ1)

                for k in range(phi_nodes.shape[0]):
                    Δφ = phi_nodes[k]  
                    vφ = v_phi[k]

                    cos_i = (np.cos(pix_slope)*cos_ts
                           + np.sin(pix_slope)*sin_ts*np.cos(np.pi - pix_aspect))
                    if cos_i < 0.0:
                        continue
                    sin_i = np.sqrt(1.0 - cos_i**2)

                    cos_r = cos_ts    
                    sin_r = sin_ts

                    cos_xi = cos_i*cos_r + sin_i*sin_r*np.cos(Δφ)
                    if cos_xi > 1.0:
                        cos_xi = 1.0
                    elif cos_xi < -1.0:
                        cos_xi = -1.0
                    xi = np.arccos(cos_xi)

                    Kvol = ((np.pi/2 - xi)*cos_xi + np.sin(xi)) / (cos_i + cos_r + eps) - pi_o_4

                    sec_i = 1.0/cos_i
                    sec_r = 1.0/cos_r

                    cos_xip = cos_i*cos_r - sin_i*sin_r*np.cos(Δφ)
                    if cos_xip > 1.0:
                        cos_xip = 1.0
                    elif cos_xip < -1.0:
                        cos_xip = -1.0
                    ξp = np.arccos(cos_xip)

                    O   = ξp/np.pi
                    geo_n = O - sec_i - sec_r + (1.0 + cos_xip)/(2.0*cos_i*cos_r)
                    kgeo = geo_n / (cos_i + cos_r + eps)

                    r = f_iso + f_vol*Kvol + f_geo*kgeo
                    if r < 0.0:
                        r = 0.0

                    acc += r * ws * vφ * cos_ts * sin_ts

            out[i, j] = acc / np.pi

    return out



def zenith(vars):
    days_since_start = vars
    return np.arccos(np.sin(global_context["latitudes"][:, None]) * np.sin(axial_tilt * np.sin(2 * np.pi * days_since_start / orbital_period_days))+ np.cos(global_context["latitudes"][:, None])* np.cos(axial_tilt * np.sin(2 * np.pi * days_since_start / orbital_period_days))*
                    np.cos(2 * np.pi * ((days_since_start*3600 % rotation_period_sec) / rotation_period_sec) + global_context["longitudes"][None, :] ))-(np.pi/2) #compute theta

def objective(days_since_start):
    return np.sum(zenith(days_since_start)**2)

def albedo_function(theta1, delta_phi):

    cos_ts = theta1

    sin_ts = np.sin(theta1)
    cos_i = np.cos(global_context["slope"])*cos_ts + np.sin(global_context["slope"])*sin_ts*np.cos(np.pi - global_context["aspect"])
    cos_i = np.maximum(cos_i, 0.0)
    sin_i = np.sqrt(np.clip(1 - cos_i**2, 0.0, 1.0))
    sin_r = np.sqrt(np.clip(1 - cos_ts**2, 0.0, 1.0))
    cos_xi = cos_i * cos_ts + sin_i * sin_r * np.cos(delta_phi)
    cos_xi = np.clip(cos_xi, -1.0, 1.0) 
    xi = np.arccos(cos_xi)
    K_vol = ((np.pi/2 - xi)*cos_xi + np.sin(xi)) / (cos_i + cos_ts + eps) - (np.pi/4)
    tan_i = sin_i/cos_i
    sec_i = 1/cos_i
    sec_r = 1/cos_ts
    O = (np.arccos(tan_i*sin_r*np.cos(delta_phi)))/np.pi 
    k_geo = O - sec_i - sec_r + (1+np.arccos(cos_i * cos_ts - sin_i * sin_r))/(2*cos_i * cos_ts)
    k_geo = k_geo / (cos_i + cos_ts + eps)

    f_iso_map = np.full((200,200), global_context["f_iso"], dtype=np.float32)
    f_vol_map = np.full((200,200), global_context["f_vol"], dtype=np.float32)
    f_geo_map = np.full((200,200), global_context["f_geo"], dtype=np.float32)

    r_BRDF = f_iso_map + f_vol_map * K_vol + f_geo_map * k_geo
    return np.clip(r_BRDF, 0.0, None)*np.cos(theta1)*np.sin(theta1)


def Prepare_albedo_map(a, f_iso = 0.05 , f_vol = 0 , f_geo = 0):
    
    """
    Prepare the variables for an albedo map in bond albedo.

    Parameters:
    a :  an empirical tuning parameter for mountain-snow cover as a function of latitude.
    f_iso : float, optional
        Baseline (isotropic) reflectance.
    f_vol : float, optional
        Volumetric scattering weight. 
    f_geo : float, optional
        Geometric shadowing weight. 
    """
    sum_f = f_iso + f_vol + f_geo + eps
    f_iso /= sum_f
    f_vol /= sum_f
    f_geo /= sum_f
    scaled_image = pygame.transform.scale(Map_image, (200, 200)).convert()
    scaled_image = scaled_image.convert(24) 
    heightmap_array = pygame.surfarray.pixels3d(scaled_image)
    heightmap_array = np.dot(heightmap_array[..., :3], [0.299, 0.587, 0.114]) / 255.0
    heightmap_array = np.swapaxes(heightmap_array, 0, 1)
    H, W = heightmap_array.shape
    latitudes = np.linspace(-np.pi/2, np.pi/2, H)
    longitudes = np.linspace(0, 2*np.pi, W)
    lat_grid = np.repeat(latitudes[:, None], W, axis=1)
    lon_grid = np.repeat(longitudes[None, :], H, axis=0)
    l.info("lat_grid: %s lon_grid: %s", lat_grid.shape, lon_grid.shape)
    base_land = 0.25
    snow = 1/(1+np.exp(-a*(heightmap_array-(np.max(heightmap_array)*np.cos(lat_grid)))))

    water_mask = heightmap_array < 0.3


    fresnel_factor = 0.02005056 #average 
    cos_z = np.sin(lat_grid) * np.sin(lon_grid) #average over time


    fresnel = fresnel_factor * (1 - cos_z)**2

    dzdx = np.zeros_like(heightmap_array)
    dzdy = np.zeros_like(heightmap_array)
    dzdx[:, 1:-1] = (heightmap_array[:, 2:] - heightmap_array[:, :-2]) * 0.5
    dzdy[1:-1, :] = (heightmap_array[2:, :] - heightmap_array[:-2, :]) * 0.5
    dzdx[:, 0]   = heightmap_array[:, 1] - heightmap_array[:, 0]
    dzdx[:, -1]  = heightmap_array[:, -1] - heightmap_array[:, -2]
    dzdy[0, :]   = heightmap_array[1, :] - heightmap_array[0, :]
    dzdy[-1, :]  = heightmap_array[-1, :] - heightmap_array[-2, :]
    slope = np.arctan(np.sqrt(dzdx**2 + dzdy**2)) # s = arctan(|∇z|).

    aspect = np.arctan2(dzdx, dzdy) 
    aspect = np.mod(aspect, 2*np.pi)

    land_mask = ~water_mask

    Bound_Upper_Outer = 2*np.pi
    Bound_Lower = 0
    err = 0.0
    Ns, Nφ = 8, 16          
    x_s, w_s = leggauss(Ns)
    phi_nodes = np.linspace(0, 2*np.pi, Nφ, endpoint=False)
    v_phi = np.full(Nφ, 2*np.pi/Nφ)
    context = {
    "albedo_function": albedo_function, "Bound_Lower": Bound_Lower,
    "Bound_Upper_Outer": Bound_Upper_Outer,
    "latitudes": latitudes, "longitudes": longitudes, "w_s": w_s, "phi_nodes": phi_nodes,
    "v_phi": v_phi, "f_iso": f_iso, "slope": slope, "aspect": aspect,
    "f_vol": f_vol, "f_geo": f_geo, "lat_grid": lat_grid,
    "water_mask": water_mask, "land_mask": land_mask, "fresnel": fresnel, "err": err,
    "base_land": base_land, "snow": snow, "heightmap_array": heightmap_array, "W": W, "H": H
    }

    init_worker(context)
    Bound_Upper_Inner = minimize(objective, x0=[0.5], method='Nelder-Mead').x[0] 
    S = float(np.mean(Bound_Upper_Inner))
    s_nodes = 0.5*(x_s + 1)*S
    w_s = w_s * 0.5*S
    context.update({
        "s_nodes": s_nodes,
        "w_s": w_s,
        "Bound_Upper_Inner": Bound_Upper_Inner
    })
    init_worker(context)



def generate_albedo_map():

    albedo_pixel = integrate_map(
        global_context['s_nodes'], global_context['w_s'], global_context['phi_nodes'], global_context['v_phi'],
        global_context['slope'], global_context['aspect'],
        global_context['f_iso'], global_context['f_vol'], global_context['f_geo']
    )
      
    albedo_pixel = np.array(albedo_pixel).reshape((200, 200))

    l.info(f"Albedo pixel value: {np.mean(albedo_pixel)}, Total error estimate: {global_context['err']}")
    base_water = 0.07
    wm = global_context["water_mask"]
    lm = global_context["land_mask"]
    sea_ice = 1 - np.cos(global_context["lat_grid"])

    albedo_map = np.zeros_like(global_context["heightmap_array"], dtype=np.float64)
    fresnel = global_context["fresnel"]

    l.info(f"fresnel: shape={fresnel.shape}, min/max={fresnel.min():.3e}/{fresnel.max():.3e}")
    l.info(f"water_mask: shape={wm.shape}, true_count={wm.sum()}")
    l.info(f"snow: shape={global_context['snow'].shape}, min={np.nanmin(global_context['snow'])}, max={np.nanmax(global_context['snow'])}")
    albedo_map[wm] = (
        base_water
        + sea_ice[wm]
        + fresnel[wm]
    )
    albedo_map[lm] = (
        global_context["base_land"]
        + global_context["snow"][lm]
        + albedo_pixel[lm]
    )
    albedo_map = np.clip(albedo_map, 0.0, 1.0)

    return albedo_map

def get_size():
    return global_context["W"], global_context["H"]

def calculate_temperature(albedo, greenhouse):
    'only use bond albedo for realistic results'
    return (np.power(((1-albedo)*L_star)/(16*np.pi*(R_exoplanet**2)*5.67e-8), 0.25)*greenhouse)+eps






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
    return 2.0*h*c*c*v*v*v/(np.exp(h*c*v/(kB*T))-1)


screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
L_star, R_exoplanet, R_star, rotation_period_sec, axial_tilt, orbital_period_sec  = convert()
Map_image = pygame.transform.scale(Map_image, (200, 200)).convert()
Prepare_albedo_map(1.5)
albedo_map = generate_albedo_map()
t = calculate_temperature(albedo_map, 1)

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
            Lc = np.minimum(L_total, 1.0 - eps)
            radicand = (1 + 2*Lc) / (1 - Lc)
            radicand = np.clip(radicand, 0.0, None)   
            n = np.sqrt(radicand)
        except FloatingPointError:
            l.warning("Invalid value encountered in sqrt calculation for n. Attempting to Convert to real number.")
            n = (1 + 2 * L_total) / (1 - L_total)
            if np.any(np.isnan(n)) or np.any(np.isinf(n)):
                l.warning("n is NaN or Inf. Returning None.")
                n = None
            else:
                n = np.real(n)
    return N_total, L_total, PP, ND

def get_abscoef(
    self_pressure: float,
    gas_name: str,
    wavenumbers: np.ndarray,
    temperature: float = np.mean(t),
    total_pressure: float = P,
    wavenumber_step: float = 0.1,
    gammaL: str = 'gamma_air',
    intensity_cutoff: float = 1e-17
    ): 
    Cond = ('AND', ('BETWEEN', 'nu', min(wavenumbers), max(wavenumbers)),
                  ('>=', 'Sw', intensity_cutoff))    
    try:
        h.select(gas_name, Conditions=Cond, DestinationTableName=f"{gas_name}_filtered")
    except Exception:
        try:
            h.fetch(gas_name,isoto_numbers[gas_name],1,min(wavenumbers),max(wavenumbers), Parameters=['nu', 'Sw'])
        except Exception as e:
            l.exception(f"Failed to fetch data for {gas_name} (isotope {isoto_numbers[gas_name]}): {e}")
            h.fetch(gas_name,isoto_numbers[gas_name],1,min(wavenumbers),max(wavenumbers))
            Cond = ('AND', ('BETWEEN', 'nu', min(wavenumbers), max(wavenumbers)))
            h.select(gas_name, Conditions=Cond, DestinationTableName=f"{gas_name}_filtered")
    nu_vals = h.getColumn(f"{gas_name}_filtered", 'nu')
    if len(nu_vals) == 0:
        l.info(f"No data above intensity_cutoff for {gas_name}. Fetching full spectra.")
        h.fetch(gas_name, isoto_numbers[gas_name], 1, min(wavenumbers), max(wavenumbers))
        Cond = ('AND', ('BETWEEN', 'nu', min(wavenumbers), max(wavenumbers)))
        h.select(gas_name, Conditions=Cond, DestinationTableName=f"{gas_name}_filtered")

    alpha, nu = h.absorptionCoefficient_Voigt(
        SourceTables=[f"{gas_name}_filtered"],
        OmegaGrid=wavenumbers,
        Environment={'T': temperature, 'p': total_pressure, 'p_self': self_pressure},
        WavenumberStep=wavenumber_step,
        GammaL=gammaL,
        HITRAN_units=False
    )
    return alpha, nu

def initialize_wavenumbers(step, nu_min, nu_max): #https://iopscience.iop.org/article/10.1088/0004-6256/149/4/131 
    v_cm = np.arange(nu_min, nu_max + step, step)
    I0 = planckWNCM(v_cm, (5777*np.sqrt(np.sqrt(L_star/(R_star/R_sun_meters)**2))))
    v_m = v_cm * 100.0 
    return pd.DataFrame({
      "nu_cm":    v_cm,
      "nu_m":     v_m,
      "inpower":  I0,       # W/m²/m⁻¹
      "dv_m":     np.gradient(v_m)
    })



