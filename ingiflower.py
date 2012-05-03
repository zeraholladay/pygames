import cv, pygame, pygame.camera, numpy
from pygame.locals import *
from itertools import product
from scipy import zeros
from random import uniform

def infinite_01():
    while True:
        yield 0
        yield 1

size = (640, 480)

pygame.init()
pygame.camera.init()

camera = pygame.camera.Camera("/dev/video0", size, "RGB") # hsv, yuv
camera.start()

display = pygame.display.set_mode(size, 0)

cvimages = [ cv.CreateImageHeader(size, cv.IPL_DEPTH_8U, 3)
             for n in range(3) ]
for cvimage in cvimages:
    pyg_image = camera.get_image()
    cv.SetData(cvimage, pyg_image.get_buffer())

flowers = zeros((size[1], size[0]),
                int)

for i in infinite_01():
    for e in pygame.event.get():
        if e.type == QUIT or (e.type == KEYDOWN and e.key == K_ESCAPE):
            exit(0)

    pyg_image = camera.get_image()
    cv.SetData(cvimages[i], pyg_image.get_buffer())

    if not i:
        cv.AbsDiff(*cvimages)
    else:
        cv.AbsDiff(cvimages[1],
                   cvimages[0],
                   cvimages[2])

    cv.Flip(cvimages[2], cvimages[2], 1)

    pyg_surface = pygame.Surface(size)
    #    pyg_surface.fill((250, 250, 250))

    for x, y in product(range(0, cvimage.height, 15), range(0, cvimage.width, 15)):
        color = cvimages[2][x, y]
        luminance = 0.0722 * color[0] + 0.7152 * color[1] + 0.2126 * color[2]
        flowers[x][y] /= 1.5
        flowers[x][y] = (luminance + flowers[x][y]) if luminance > 5 else 0
        if flowers[x][y]:
            color = flowers[x][y]
            color = 255 if color > 255 else color
            i = (x + y + int(uniform(0, 3))) % 3
            rx = uniform(-50, 50)
            ry = uniform(-50, 50)
            coord = (int(y + ry), int(x + rx))
            if 0 == i:
                pygame.draw.circle(pyg_surface, (color, 0, 0), coord, color / 3)
            elif  1 == i:
                pygame.draw.circle(pyg_surface, (0, color, 0), coord, color / 3)
            else:
                pygame.draw.circle(pyg_surface, (0, 0, color), coord, color / 3)
        
    display.blit(pyg_surface, (0,0))             
    pygame.display.flip()
