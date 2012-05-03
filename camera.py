import pygame
from pygame.locals import *
from pygame.mask import *

def makeImageFeeder(url):
    from urllib import urlopen
    import ImageFile

    def _f():
        f = urlopen(url)
        p = ImageFile.Parser()
        while True:
            buf = f.read(1024)
            if not buf: break
            else: p.feed(buf)
        image = p.close()
        ibuffer = image.tostring()
        return ibuffer, image.size

    return _f

url = "http://192.168.1.118:8080/shot.jpg"
feeder = makeImageFeeder(url)

pygame.init()
(image, size) = feeder()
display = pygame.display.set_mode(size, 0)

while True:
    events = pygame.event.get()
    for e in events:
        if e.type == QUIT or (e.type == KEYDOWN and e.key == K_ESCAPE):
             exit(0)
    buf, size = feeder()
    image = pygame.image.fromstring(buf, size, 'RGB')
    ########################################
    # mask color:                          #
    ########################################
    color = (100, 0, 0 ) # the color we're interested in
    mask = pygame.mask.from_threshold(image, color, (30, 30, 30))
    if mask.count() > 5:
        coord = mask.centroid()
        print coord
        pygame.draw.circle(image, (0,255,0), coord, max(min(50,mask.count()/400),5) )
    ########################################
    # mask color:                          #
    ########################################
    display.blit(image, (0,0))             
    pygame.display.flip()
