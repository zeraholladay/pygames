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

size = (620, 480)

pygame.init()
pygame.camera.init()

camera = pygame.camera.Camera("/dev/video0", size, "RGB") # hsv, yuv
camera.start()

display = pygame.display.set_mode(size, 0)
cvimage = cv.CreateImageHeader(size, cv.IPL_DEPTH_8U, 3)

while True:
    for e in pygame.event.get():
        if e.type == QUIT or (e.type == KEYDOWN and e.key == K_ESCAPE):
            exit(0)

    pyg_image = camera.get_image()
    mask = pygame.mask.from_threshold(pyg_image, (0x80, 0x1a,0x2b), (0x40, 10, 10))
    connected = mask.connected_component()
    
    if mask.count() > 20:
        coord = mask.centroid()
        pygame.draw.circle(pyg_image, (0,255,0), coord, max(min(50,mask.count() / 400),5))

    display.blit(pyg_image, (0,0))    
    pygame.display.flip()

