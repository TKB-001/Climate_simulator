import pygame
from config import *
from radiative import Irradiance, fill_water, aboscf
from optics import albedo_map
import time
import logging as l
import numpy as np
import pygame.surfarray as sa
from optics import Map_image
import faulthandler
faulthandler.enable(all_threads=True)

clock = pygame.time.Clock()
pygame.init()

screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
irradiance_map = pygame.Surface((200, 200))
irradiance_map.set_alpha(200)

Map_image = fill_water(Map_image, 143, albedo_map)
cs = aboscf()

l.basicConfig(filename='executor.log', level=l.DEBUG, force=True)

              
l.info('check')
pixel_maps = []
def generate(t, time_range, samples, cs):
    global pixel_maps
    t *= 3600 * 24
    time_range *= 3600 * 24

    irr = Irradiance()
    start_time = time.perf_counter()
    arr = irr.calculate_irradiance_time(t, time_range, samples, cs)
    end_time = time.perf_counter()

    l.info(f"Total time for {samples} samples: {end_time - start_time:.2f} seconds")

    frame = np.transpose(arr, (1, 0, 2))
    pixel_maps.append(frame)

orbital_period_parts = (orbital_period_days*detla)/4
#generate(orbital_period_parts, 20, 250, cs)
#generate(orbital_period_parts*2, 20, 250, cs)
#generate(orbital_period_parts*1.5, 20, 250, cs)
generate(orbital_period_parts*2, 0, 1, cs)
current_map = 0
start_time = pygame.time.get_ticks()
running = True 
while running:
    elapsed_time = (pygame.time.get_ticks() - start_time) / 1000
    screen_width, screen_height = screen.get_size()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_1, pygame.K_2, pygame.K_3):
                current_map = event.key - pygame.K_1
            elif event.key == pygame.K_q:
                running = False





    sa.blit_array(irradiance_map, pixel_maps[current_map])



    resized_image = pygame.transform.scale(Map_image, (screen_width, screen_height))
    resized_irradiance = pygame.transform.scale(  irradiance_map, (screen_width, screen_height))


    screen.fill((0, 0, 0))  
    screen.blit(resized_image, (0, 0)) 
    screen.blit(resized_irradiance, (0, 0))

    pygame.display.flip()  
    clock.tick(60)
pygame.quit()
