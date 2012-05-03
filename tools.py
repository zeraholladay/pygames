import cv, pygame, pygame.camera, numpy
from pygame.locals import *
from itertools import product
from scipy import zeros

def canny(cvimage, threshold=(0, 255)):
    size = cv.GetSize(cvimage)
    gray = cv.CreateImage(size, cv.CV_8UC2, 1)
    edges = cv.CreateImage(size, cv.CV_8UC2, 1)    
    cv.CvtColor(cvimage, gray, cv.CV_BGR2GRAY)
    cv.Canny(gray, edges, *threshold)
    cv.CvtColor(edges, cvimage, cv.CV_GRAY2BGR)    
    return cvimage

def featurefinder(cvimage, n=150):
    cvimage_bnw = cv.CreateImage(cv.GetSize(cvimage), cvimage.depth, 1)
    cv.CvtColor(cvimage, cvimage_bnw, cv.CV_BGR2GRAY)
    eig_image = cv.CreateImage(cv.GetSize(cvimage_bnw), cvimage.depth, 1)
    temp_image = cv.CreateImage(cv.GetSize(cvimage_bnw), cvimage.depth, 1)
    features = cv.GoodFeaturesToTrack(cvimage_bnw, eig_image, temp_image, n, 0.004, 1.0, useHarris = True)
    return features # [ (x0, y0), (x1, y1), ... ]

def smooth(cvimage, parms=(5,5)):
    cv.Smooth(cvimage, cvimage, cv.CV_GAUSSIAN, *parms)
    return cvimage

def sobel(cvimage, xyorders=(0, 1)):
    gray = cv.CreateImage(cv.GetSize(cvimage), cv.CV_8UC2, 1)    
    sobel = cv.CreateMat(gray.height, gray.width, cv.CV_16S)
    cv.CvtColor(cvimage, gray, cv.CV_BGR2GRAY)    
    cv.Sobel(gray, sobel, *xyorders)
    cv.ConvertScaleAbs(sobel, gray, 1, 0)
    cv.CvtColor(gray, cvimage, cv.CV_GRAY2BGR)
    return cvimage

def infinite_01():
    while True:
        yield 0
        yield 1

size = (640, 480)
flowers = scipy.zeros(size, int)

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

    cv.Flip(cvimage[2], cvimage[2], 1)

    for x, y in product(range(20, cvimage.height, 20), range(20, cvimage.width, 20)):
        color = cvimage[2][x, y]
        luminance = 0.0722 * color[0] + 0.7152 * color[1] + 0.2126 * color[2]
        flowers[x][y] /= 3
        flowers[x][y] += luminance

        #    cv.CvtColor(cvflowers, cvflowers, cv.CV_BGR2RGB)
        #    pyg_image = pygame.image.frombuffer(cvflowers.tostring(),
        #                                        cv.GetSize(cvflowers), "RGB")
        
    display.blit(pyg_flowers, (0,0))             
    pygame.display.flip()
