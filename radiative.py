import numpy as np
import pygame
from concurrent.futures import ProcessPoolExecutor, as_completed
from numba import cuda
from optics import start, initialize_wavenumbers, get_abscoef, polarizability_mol, t, get_size
import math
from config import *
import logging as l
from chemistry import depolarization
from interplanetary import orbital_distance
from numba.cuda import occupancy


l.basicConfig(filename='radiative.log', level=l.DEBUG, force=True)


l.info('check')


N_total, L_total, PP, ND  = start()
wn = initialize_wavenumbers(0.1, nu_min, nu_max)
I_star = wn["inpower"].to_numpy()   # W/m² per m⁻¹
dv      = wn["dv_m"].to_numpy()     # m⁻¹ spacing
v_m = wn["nu_m"].to_numpy()  
L_star, R_exoplanet, R_star, rotation_period_sec, axial_tilt, orbital_period_sec = convert()

H, W = get_size()


def fill_water(surface, level, albedo_map):
    surface = surface.convert(24)
    surface.lock()
    array = pygame.surfarray.pixels3d(surface)
    surface.unlock()
    array = array.astype(np.uint8)
    array = np.transpose(array, (1, 0, 2))

    grayscale = np.dot(array[..., :3], [0.299, 0.587, 0.114])

    mask = grayscale < level

    albedo_255 = (albedo_map * 255)
    if albedo_255.ndim == 2:
        albedo_255 = np.stack((albedo_255,)*3, axis=-1)
    albedo_surface = pygame.surfarray.make_surface(albedo_255)
    resized_albedo_surface = pygame.transform.smoothscale(albedo_surface, (200, 200))
    resized_albedo = pygame.surfarray.array3d(resized_albedo_surface).astype(np.float32) / 255.0
    resized_albedo_gray = np.dot(resized_albedo[..., :3], [0.299, 0.587, 0.114])
    

    water_condition = mask
    ice_condition = (resized_albedo_gray >= 0.5)  &  (resized_albedo_gray <= 0.7)
    array[water_condition] = [0, 0, 255]  
    array[ice_condition] = [255, 255, 255]  



    pygame.surfarray.blit_array(surface, np.transpose(array, (1, 0, 2)))  
    return surface



@cuda.jit(fastmath=True)
def step_kernel( latitudes,longitudes, temp, v, s_tot, R_mean, axial_tilt, orbital_period_days
                , rotation_period_sec, radiance, G,molar_mass, exoplanet_R, N_total, I_star, I_star_loc, time_steps, R_star,
                orbital_period_sec, eccentricity, R_exoplanet):
    ''' the code for calculating flux_diffuse above is a theoretical model based on standard Rayleigh scattering principles,
      and the code for direct flux is derived from two-stream radiative transfer approximations.'''
    
    '''References: [1] J. A. Sutton and J. F. Driscoll, "Rayleigh scattering cross sections of combustion species at 266, 355, and 532 nm for thermometry applications,"
    Optics Letters, vol. 29, no. 22, pp. 2620-2622, Nov. 2004.
    [2] Q. Wang, L. Jiang, W. Cai, and Y. Wu, "Study of UV Rayleigh scattering thermometry for flame temperature field measurement," J. Opt. Soc. Am. B, vol. 36,
    no. 10, pp. 2843-2852, Oct. 2019.
    [3] P. J. Webster and R. Lukas, “Tropical ocean-atmosphere interaction:
    The role of clouds in the radiative feedback,” J. Atmos. Sci., vol. 37, no. 3, pp. 630-
    643, Mar. 1980. [Online]. Available: https://journals.ametsoc.org/view/journals/atsc/37/3/1520-0469_1980_037_0630_tsatrt_2_0_co_2.xml'''
    h, w = cuda.grid(ndim=2)
    H, W = radiance.shape
    if h >= H or w >= W:
        return
    for idx in range(time_steps.size):
        current_time = time_steps[idx]
        r_t = orbital_distance(current_time, R_exoplanet, eccentricity, orbital_period_sec) #device function
        scale = (R_star / r_t)**2
        days_since_start = current_time / (24 * 3600)
        eps = 1e-8
        declination = axial_tilt * math.sin(2 * math.pi * days_since_start / orbital_period_days) 

        lat = latitudes[h]
        lon = longitudes[w]
        hour_angle = 2 * math.pi * ((current_time % rotation_period_sec) / rotation_period_sec) + lon
        cos_zenith = (
            math.sin(lat) * math.sin(declination)
            + math.cos(lat) * math.cos(declination) * math.cos(hour_angle)
        )
        if cos_zenith < 0:
            cos_zenith= 0
        zenith = math.acos(cos_zenith)

        H_atmosphere = (8.314*temp[h, w])/(G*molar_mass)

        air_mass_temp = (math.sqrt((exoplanet_R + H_atmosphere)**2 - (exoplanet_R * math.sin(zenith))**2) - exoplanet_R * math.cos(zenith)) * 100.0        
        g = 0.0 
        a_ij = N_total[h, w] * air_mass_temp
        cz = cos_zenith

        local_flux = 0.0
        sum_tau    = 0.0

        for k in range(v.size - 1):
            I_star_loc= I_star[k] * scale
            dv = v[k+1] - v[k]

            τ   = a_ij * s_tot[k] #[3]

            ω0  = (a_ij * R_mean) / (τ + eps) 
            denom = max(2.0 * (1.0 - ω0 * g), eps) 
            x     = τ / (cz + eps) 
            two_term = (ω0 / denom) * (1.0 - math.exp(-x)) 
            local_flux += I_star_loc * cz * two_term * dv 

            τk   = a_ij * s_tot[k] #Beer–Lambert law ([1] & [2])
            τk1  = a_ij * s_tot[k+1] 
            Ik   = I_star[k] * math.exp(-τk / cz)
            Ik1  = I_star[k+1] * math.exp(-τk1 / cz)
            sum_tau += 0.5 * (Ik + Ik1) * dv * cz

        total_width = v[-1] - v[0] 
        radiance[h, w] = (local_flux+sum_tau * cz)/ total_width



def aboscf():
    cs = {}
    nu = wn["dv_m"].to_numpy()
    args = [
        (compound, nu, np.mean(t), P, Pi)
        for compound, Pi in PP.items()
    ]
    with ProcessPoolExecutor(max_workers=4) as exe:
        futures = {
            exe.submit(get_abscoef, t,  *arg): arg[0]
            for arg in args
        }
        for fut in as_completed(futures):
            gas = futures[fut]
            try:
                alpha, nu_out = fut.result()
            except Exception as e:
                l.error(f"get_abscoef failed for {gas}: {e}")
                alpha = np.zeros_like(nu)
                nu_out = nu
            cs[gas] = (alpha, nu_out)
    return cs



class Irradiance:
    def __init__(self):
        self.array = np.zeros((200, 200, 3), dtype=np.uint8)
        self.d_radiance = cuda.device_array((H,W), dtype=np.float32)
        self.d_I_star_loc = cuda.device_array((H,W), dtype=np.float32)

    def compute_flux(self,H, W,s_tot,R_mean,v, latitudes, longitudes, temp
        ,d_N_total, d_I_star, time_steps, R_star, orbital_period_sec, eccentricity, R_exoplanet, bmax):
        
        threadsperblock = (bmax, bmax)
   
        blockspergrid = (
            (H + threadsperblock[0] - 1) // threadsperblock[0],
            (W + threadsperblock[1] - 1) // threadsperblock[1]
        )


        step_kernel[blockspergrid, threadsperblock]( 
            latitudes, longitudes, temp, v, s_tot, R_mean, np.float32(axial_tilt), np.float32(orbital_period_days)
            , np.float32(rotation_period_sec), self.d_radiance, np.float32(G),np.float32(molar_mass), np.float32(exoplanet_R), d_N_total, d_I_star, self.d_I_star_loc,
            time_steps, R_star, orbital_period_sec, eccentricity, R_exoplanet

        )

        flux = self.d_radiance.copy_to_host()
        return flux
    def step(self, longitudes,d_longitudes,
              latitudes,d_latitudes, temp,d_temp, s_tot,d_s_tot, v,
                d_v, R_mean, d_R_mean, I_star, d_I_star,N_total, d_N_total, time_steps, R_star,
                orbital_period_sec, eccentricity, R_exoplanet, bmax):

        cuda.select_device(0)
        irradiance_array = self.compute_flux(H, W, d_s_tot, d_R_mean,d_v, d_latitudes, d_longitudes, d_temp, d_N_total, I_star, time_steps, R_star,
                                              orbital_period_sec, eccentricity, R_exoplanet, bmax)  
        # H, W,s_tot,R_mean,v, current_time, latitudes, longitudes, temp

    
        l.info("mean total_irradiance: %f", np.mean(irradiance_array))
        if np.isnan(irradiance_array).any():
            l.warning("Irradiance array contains NaN values")
        if np.isinf(irradiance_array).any():
            l.warning("Irradiance array contains inf values")
        return irradiance_array


    def calculate_irradiance_time(self, time, time_range, samples, cs):
        global v_m, I_star, N_total, axial_tilt, orbital_period_days, rotation_period_sec, G, molar_mass, exoplanet_R, R_exoplanet, eccentricity, orbital_period_sec
        start_time = time - time_range
        end_time = time + time_range

        time_steps = np.linspace(start_time, end_time, samples)

        latitudes = np.linspace(-np.pi/2, np.pi/2, 200)
        longitudes = np.radians(np.linspace(-180, 180, 200))
        total_irradiance = np.zeros((200, 200))

        temp = t
        temp = np.clip(temp, 50.0, 350.0) 
        alpha_arrays = [(alpha) for alpha, nu in cs.values()] #cm^-1
        s_abs = np.sum(alpha_arrays, axis=0).astype(np.float32) #cm^-1
        lambda_m = 1.0/v_m
        α_vol = np.mean(polarizability_mol) / (4*np.pi*8.8541878128e-12)

        depol_val = np.asarray(depolarization, dtype=np.float32)
        depol_safe = np.clip(depol_val, 0, 6.0/7.0 - 1e-6)
        r_scatter = (24*np.pi**3 * α_vol**2 * (6+3*depol_safe)/(6-7*depol_safe)
                    / lambda_m**4)
        s_tot = (s_abs + r_scatter)
        R_mean = np.mean(r_scatter)
        R_mean /= 1e-4
        results= []



        R_mean = np.float32(R_mean)
        axial_tilt = np.float32(axial_tilt)
        orbital_period_days = np.float32(orbital_period_days)
        rotation_period_sec = np.float32(rotation_period_sec)
        G = np.float32(G)
        molar_mass = np.float32(molar_mass)
        exoplanet_R= np.float32(exoplanet_R)
        d_R_mean= np.float32(R_mean)  
        R_exoplanet = np.float32(R_exoplanet) 
        eccentricity, orbital_period_sec = np.float32(eccentricity), np.float32(orbital_period_sec)

        d_N_total=cuda.to_device(N_total.astype(np.float32))

        d_s_tot= cuda.to_device(s_tot.astype(np.float32))
        d_v= cuda.to_device(v_m.astype(np.float32))
        d_temp = cuda.to_device(temp.astype(np.float32))
        d_longitudes, d_latitudes = cuda.to_device(longitudes.astype(np.float32)), cuda.to_device(latitudes.astype(np.float32))    
        d_I_star = cuda.to_device(I_star.astype(np.float32))
        time_steps = cuda.to_device(time_steps.astype(np.float32))
        bmin, bmax = occupancy_max_potential_block_size(step_kernel, [H, W, d_s_tot, d_R_mean,d_v, d_latitudes, d_longitudes, d_temp, d_N_total, I_star, time_steps, R_star,
                                              orbital_period_sec, eccentricity, R_exoplanet])
        total_irradiance = self.step(longitudes,d_longitudes, latitudes,d_latitudes,
                        temp, d_temp,s_tot,  d_s_tot,v_m, d_v, R_mean,
                        d_R_mean, I_star, d_I_star, N_total, d_N_total, time_steps, R_star, orbital_period_sec, eccentricity, R_exoplanet, bmax
                        )


            

        averaged_irradiance = total_irradiance / samples

        max_irradiance = np.max(averaged_irradiance)
        l.info("Max averaged irradiance: %f", max_irradiance)
        l.info("Min averaged irradiance: %f", np.min(averaged_irradiance))

        normalized_irradiance = averaged_irradiance / (max_irradiance if max_irradiance > 0 else 1)
        r = np.clip(255 * (normalized_irradiance ** 1.0), 0, 255)
        green = r
        b = r

        self.array[:] = np.stack([r, green, b], axis=-1).astype(np.uint8)
        return self.array[:]



