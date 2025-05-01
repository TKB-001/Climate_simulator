import numpy as np
import pygame
from numba import prange, cuda, njit
from optics import albedo_map, start, initialize_wavenumbers, get_abscoef, calculate_temperature
import math
from config import *
import logging as l
from chemistry import depolarization 
from concurrent.futures import ThreadPoolExecutor
from interplanetary import orbital_distance

t = calculate_temperature(albedo_map, 1)
l.basicConfig(

    filename='radiative.log', 
    level=l.INFO,
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s' 
)

l.info('check')



N_total, L_total, PP = start()
wavenumbers = initialize_wavenumbers()
I_star = wavenumbers["inpower"].to_numpy()
L_star, R_exoplanet, R_star, rotation_period_sec, axial_tilt, orbital_period_sec = convert()


def fill_water(surface, level, albedo_map):
    surface = surface.convert(24)
    surface.lock()
    array = pygame.surfarray.pixels3d(surface)
    surface.unlock()
    array = array.astype(np.uint8)
    array = np.transpose(array, (1, 0, 2))

    grayscale = np.dot(array[..., :3], [0.299, 0.587, 0.114])

    mask = grayscale < level
    new_size = (grayscale.shape[0], grayscale.shape[1]) 

    albedo_255 = (albedo_map * 255).astype(np.uint8)
    if albedo_255.ndim == 2:
        albedo_255 = np.stack((albedo_255,)*3, axis=-1)
    albedo_surface = pygame.surfarray.make_surface(albedo_255)
    resized_albedo_surface = pygame.transform.smoothscale(albedo_surface, new_size)
    resized_albedo = pygame.surfarray.array3d(resized_albedo_surface).astype(np.float32) / 255.0
    resized_albedo_gray = np.dot(resized_albedo[..., :3], [0.299, 0.587, 0.114])
    

    water_condition = mask
    ice_condition = resized_albedo_gray >= 0.6
    array[water_condition] = [0, 0, 255]  
    array[ice_condition] = [255, 255, 255]  



    pygame.surfarray.blit_array(surface, np.transpose(array, (1, 0, 2)))  
    return surface





@cuda.jit
def compute_tau_cuda_kernel( N_total, air_mass, s_tot, I_star, v, cos_zenith, tau_out, chunk_size):
                i, j = cuda.grid(2)
                lat, lon = N_total.shape
                n = v.shape[0]

                if i >= lat or j >= lon:
                    return

                a_ij = N_total[i, j] * air_mass[i, j]
                cz = cos_zenith[i, j]
                sum_ij = 0.0

                for start in range(0, n - 1, chunk_size):
                    end = min(start + chunk_size + 1, n)
                    for k in range(start, end - 1):
                        τk = a_ij * s_tot[k]
                        τk1 = a_ij * s_tot[k + 1]
                        Ik = I_star[k] * math.exp(-τk)
                        Ik1 = I_star[k + 1] * math.exp(-τk1)
                        sum_ij += 0.5 * (Ik + Ik1) * (v[k + 1] - v[k])

                tau_out[i, j] = sum_ij * cz

@njit(parallel=True, fastmath=True)
def compute_tau( N_total, air_mass, s_tot, I_star, v, cos_zenith, chunk_size): 
        lat, lon = N_total.shape
        A = (N_total * air_mass).astype(np.float32)
        n = v.shape[0]
        τ = np.zeros((lat, lon), dtype=np.float32)
        for i in prange(lat):
            for j in range(lon):
                a_ij = A[i, j]
                cz = cos_zenith[i, j]
                sum_ij = 0.0
                for start in range(0, n - 1, chunk_size):
                    end = min(start + chunk_size + 1, n)
                    τ_vec = a_ij * s_tot[start:end]
                    for k in range(start, end - 1):
                        τk = τ_vec[k - start]
                        τk1 = τ_vec[k - start + 1]
                        Ik = I_star[k] * np.exp(-τk)
                        Ik1 = I_star[k + 1] * np.exp(-τk1)
                        sum_ij += 0.5 * (Ik + Ik1) * (v[k + 1] - v[k])
                τ[i, j] = sum_ij * cz

        return τ

def aboscf():
        cs = {}
        for compound, Pi in PP.items():
            cs[compound] = get_abscoef(compound, wavenumbers["nu"].to_numpy(), np.mean(t) , P, Pi)
        return cs

class Irradiance:
    def __init__(self):
        self.array = np.zeros((200, 200, 3), dtype=np.uint8)

    def compute_tau_cuda(self, N_total, air_mass, s_tot, I_star, v, cos_zenith, chunk_size):
                lat, lon = N_total.shape

                tau_out = np.zeros((lat, lon), dtype=np.float32)

                d_N_total = cuda.to_device(N_total.astype(np.float32))
                d_air_mass = cuda.to_device(air_mass.astype(np.float32))
                d_s_tot = cuda.to_device(s_tot.astype(np.float32))
                d_I_star = cuda.to_device(I_star.astype(np.float32))
                d_v = cuda.to_device(v.astype(np.float32))
                d_cos_zenith = cuda.to_device(cos_zenith.astype(np.float32))
                d_tau_out = cuda.device_array_like(tau_out)

                threadsperblock = (16, 16)
                blockspergrid_x = math.ceil(lat / threadsperblock[0])
                blockspergrid_y = math.ceil(lon / threadsperblock[1])

                compute_tau_cuda_kernel[(blockspergrid_x, blockspergrid_y), threadsperblock](
                    d_N_total, d_air_mass, d_s_tot, d_I_star, d_v, d_cos_zenith, d_tau_out, chunk_size
                )

                return d_tau_out.copy_to_host()
    def calculate_irradiance_time(self, time, time_range, samples, cs):
        start_time = time - time_range
        end_time = time + time_range

        time_steps = np.linspace(start_time, end_time, samples)

        I0 = L_star / (4 * np.pi * R_exoplanet**2)
        latitudes = np.linspace(-np.pi/2, np.pi/2, 200)
        longitudes = np.radians(np.linspace(-180, 180, 200))
        total_irradiance = np.zeros((200, 200))

        temp = t
        scalar_temp = np.mean(temp).item()
        v = wavenumbers["nu"].to_numpy()   
        alpha_arrays = [alpha for alpha, nu in cs.values()]
        s_tot = np.sum(alpha_arrays, axis=0).astype(np.float32)
        l.info("Mean temp: %f",scalar_temp )

        def step(current_time):
            days_since_start = current_time / (24 * 3600)

            declination = axial_tilt * np.sin(2 * np.pi * days_since_start / orbital_period_days)

            hour_angles = 2 * np.pi * ((current_time % rotation_period_sec) / rotation_period_sec) + longitudes[None, :]
            cos_zenith = (
                np.sin(latitudes[:, None]) * np.sin(declination)
                + np.cos(latitudes[:, None])
                * np.cos(declination)
                * np.cos(hour_angles)
            )
            cos_zenith[cos_zenith < 0] = 0
            zenith = np.acos(cos_zenith)

            H_atmosphere = (8.314*temp)/(G*molar_mass)

            air_mass_temp = (np.sqrt((exoplanet_R + H_atmosphere)**2 - (exoplanet_R * np.sin(zenith))**2) - exoplanet_R * np.cos(zenith)) * 100.0
            r_t = orbital_distance(current_time, R_exoplanet, eccentricity, orbital_period_sec)
            scale = (R_star / r_t)**2 
            I_star = I_star * scale
            try:
                cuda.select_device(0)
                τ = self.compute_tau_cuda(N_total, air_mass_temp, s_tot, I_star, v, cos_zenith, 5000)
            except Exception:    
                τ = compute_tau(N_total, air_mass_temp, s_tot, I_star, v, cos_zenith, 1000)
            l.info("mean τ: %f", np.mean(τ))
            λ = 500e-9
            r_scatter = ((24*np.pow(np.pi, 3))/(np.pow(λ ,4))*N_total**2)*(L_total**2)*((6+3*depolarization)/(6-7*depolarization))
             # the equation above is a theoretical model based on standard Rayleigh scattering principles.
            '''References: [1] J. A. Sutton and J. F. Driscoll, "Rayleigh scattering cross sections of combustion species at 266, 355, and 532 nm for thermometry applications," Optics Letters, vol. 29, no. 22, pp. 2620–2622, Nov. 2004.
            [2] Q. Wang, L. Jiang, W. Cai, and Y. Wu, "Study of UV Rayleigh scattering thermometry for flame temperature field measurement," J. Opt. Soc. Am. B, vol. 36, no. 10, pp. 2843–2852, Oct. 2019.
            '''
            l.info("mean r_scatter: %f", np.mean(r_scatter))  
            SA = r_scatter/(r_scatter+(τ))
            l.info("mean SA: %f", np.mean(SA))
            flux_diffuse = ((I0*albedo_map*SA)/(1-SA*albedo_map))+((τ*I0)/(1-SA*albedo_map)) 
            '''formulation derived from two-stream radiative transfer approximations.
            citations: N. D. Shine, "Parametrization of the shortwave flux over high albedo surfaces as a function of cloud thickness and surface albedo," 
            Quarterly Journal of the Royal Meteorological Society, vol. 110, no. 465, pp. 747-764, 1984, doi: 10.1002/qj.49711046511. G. E. Thomas and K. Stamnes,
            Radiative Transfer in the Atmosphere and Ocean. Cambridge, U.K.: Cambridge University Press, 1999. '''
            l.info("mean flux_diffuse: %f", np.mean(flux_diffuse))
            irradiance_array = flux_diffuse * cos_zenith 
            if np.isnan(irradiance_array).any():
                l.warning("Irradiance array contains NaN values")
            if np.isinf(irradiance_array).any():
                l.warning("Irradiance array contains inf values")

            return irradiance_array
        results= []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(lambda current_time: step(current_time), current_time) for current_time in time_steps]
            for f in futures:
                result = f.result()
                results.append(result)
        total_irradiance =  sum(results)

        averaged_irradiance = total_irradiance / samples

        print(np.max(averaged_irradiance) if np.max(averaged_irradiance) > 0 else 1)
        print(np.min(averaged_irradiance) if np.min(averaged_irradiance) > 0 else 1)

        normalized_irradiance = averaged_irradiance / (np.max(averaged_irradiance) if np.max(averaged_irradiance) > 0 else 1)
        r = np.clip(255 * (normalized_irradiance ** 1.0), 0, 255)
        g = r
        b = r

        self.array[:] = np.stack([r, g, b], axis=-1).astype(np.uint8)
        return self.array[:]
