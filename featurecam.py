import cv, pygame
from pygame.locals import *
from math import hypot
from itertools import product, izip_longest
from scipy.cluster.vq import kmeans2
from numpy import array as np_array
from couchdb import *

########################################
# analyze logic:                       #
########################################

def distance(p1, p2):
    x = [ x - y for x, y in zip(p1, p2) ]
    return hypot(*x)

def groupby(n, iterable, fillvalue=None):
    args = [iter(iterable)] * n
    return izip_longest(fillvalue=fillvalue, *args)

def match(matrix1, matrix2, ratio=.9):
    l1, l2 = len(matrix1), len(matrix2)
    small, big = (l1, l2) if l1 < l2 else (l2, l1)
    distances = [ distance(*x) for x in product(matrix1, matrix2) ]
    for sorted_distances in [ sorted(x) for x in groupby(big, distances) ]:
        d1, d2 = sorted_distances[:2]
        try:
            if d1 / d2 > ratio:
                pass
            else:
                yield d1
        except ZeroDivisionError:
            pass

def record_rank(features, docs):
    for doc in docs:
        url = "http://localhost:5984/dud/" + doc["_id"] + "/" + doc["_attachments"].keys()[0]
        feeder = makeImageFeeder(url)
        normalized_cvimage, size = feeder()
        normalized_features = featurefinder(normalized_cvimage)
        matches = list(match(features, normalized_features))
        length = len(matches)
        if 0 == length:
            continue
        score = sum(matches) / length # XXX adjust scoring here
        try:
            if score < best_score:
                best_score, _id, name = score, doc["_id"], doc["Name"]
        except NameError:
            best_score, _id, name = score, doc["_id"], doc["Name"]
    print best_score, name            
    return best_score, _id, name

def server_rank(features, docs):
    for doc in docs:
        matches = list(match(features, doc["features"]))
        length = len(matches)
        if 0 == length:
            continue
        score = sum(matches) / length # XXX adjust scoring here
        try:
            if score < best_score:
                best_score, _id, name = score, doc["_id"], doc["Name"]
        except NameError:
            best_score, _id, name = score, doc["_id"], doc["Name"]
    return best_score, _id, name

########################################
# makeImageFeeder: returns function    #
########################################

def makeImageFeeder(url):
    from urllib import urlopen
    import ImageFile # PIL image loader    
    ########################################
    # Returns (B&W cv object, size)        #
    ########################################
    def _f():
        try:
            f = urlopen(url)
            p = ImageFile.Parser()
            while True:
                buf = f.read(1024)
                if not buf: break
                else: p.feed(buf)
            pil = p.close() # i.e. PIL image
            pilbuf = pil.tostring()
            cvimage = cv.CreateImageHeader(pil.size, cv.IPL_DEPTH_8U, 3)
            cv.SetData(cvimage, pilbuf)
            return cvimage, cv.GetSize(cvimage)
        except:
            print "Failed to connect to %s.  Is IP Webcam started on this host?" % url
            exit(1)
    return _f

########################################
# Image transformations:               #
########################################

def centroidfinder(cvimage, color, threshold):
    lo =  [ c - t for c, t in zip(color, threshold) ]
    hi =  [ c + t for c, t in zip(color, threshold) ]
    mat = cv.CreateMat(cvimage.height, cvimage.width, cv.CV_8U)
    cv.InRangeS(cvimage, lo, hi, mat)
    data = [ [x, y] for x, y in product(range(mat.height), range(mat.width))
             if int(mat[x, y]) ]
    np_data = np_array(data)
    np_centroids = np_array( [ [0, 0], [0, mat.width],
                            [mat.height, 0], [mat.height, mat.width] ])
    centroids, labels = kmeans2(np_data, np_centroids)
    return [ (x, y) for x, y in centroids.tolist() ]

def featurefinder(cvimage, n=150):
    cvimage_bnw = cv.CreateImage(cv.GetSize(cvimage), cvimage.depth, 1)
    cv.CvtColor(cvimage, cvimage_bnw, cv.CV_BGR2GRAY)
    eig_image = cv.CreateImage(cv.GetSize(cvimage_bnw), cvimage.depth, 1)
    temp_image = cv.CreateImage(cv.GetSize(cvimage_bnw), cvimage.depth, 1)
    features = cv.GoodFeaturesToTrack(cvimage_bnw, eig_image, temp_image, n, 0.004, 1.0, useHarris = True) 
    return features # [ (x0, y0), (x1, y1), ... ]

def warp(cvimage, src):
    """
src is the contains 4xy points from centroidfinder()
    """
    dst = [ (0, 0), (0, Config.width),
            (Config.height, 0), (Config.height, Config.width) ]
    distance_groups = groupby(4,
                              [ (distance(*prod), prod) for prod in product(dst, src) ])
    distance_least = [ sorted(x, cmp=lambda x, y: cmp(x[0], y[0]))[0] for x in distance_groups ]
    dst = [ tuple(reversed(m[0])) for d, m in distance_least ]
    src = [ tuple(reversed(m[1])) for d, m in distance_least ]
    mat =  cv.CreateMat(3, 3, cv.CV_32FC1)
    out = cv.CreateImage((Config.width, Config.height),
                         cvimage.depth, cvimage.nChannels)
    cv.GetPerspectiveTransform(src, dst, mat)
    cv.WarpPerspective(cvimage, out, mat)
    return out

########################################
# main:                                #
########################################

if "__main__" == __name__:    
    from sys import argv
    from yaml import load
    from getopt import getopt, GetoptError
    from posixpath import basename, splitext

    prog = basename(argv[0])
    base, extension = splitext(prog)
    
    try:
        flags =  [ "mode=", "device=" ]
        opts, args = getopt(argv[1:], "", flags)
    except GetoptError:
        pretty_flags = " ".join([ "--%s [val]" % x.rstrip("=") for x in flags ])
        print "usage: %s" % pretty_flags
        exit(1)

    options = dict((x.lstrip("-"), y) for x, y in opts)

    with open(base + ".yaml") as f:
        options.update(load(f))
        Config = type("Config", (dict,), options)

    device_name = Config.device
    all_devices = Config.devices[device_name]

    for attr in all_devices.items():
        setattr(Config, *attr)

        #    server = Server(Config.server)
        #    Config.couchdb = server[Config.dbname]
    
    feeder = makeImageFeeder(Config.url)
    cvimage, size = feeder()

    pygame.init()
    display = pygame.display.set_mode(size, 0)


    while True:
        for e in pygame.event.get():
            if e.type == QUIT or (e.type == KEYDOWN and e.key == K_ESCAPE):
                exit(0)

        cvimage, cvimage_size = feeder()
        #
        cv.Smooth(cvimage, cvimage, cv.CV_MEDIAN, 3, 3)
        gray = cv.CreateImage(cv.GetSize(cvimage), 8, 1)
        edges = cv.CreateImage(cv.GetSize(cvimage), 8, 1)
        cv.CvtColor(cvimage, gray, cv.CV_BGR2GRAY)
        #cv.EqualizeHist(gray, gray)
        #        cv.Threshold(gray, gray, 200, 255, cv.CV_THRESH_BINARY)
        cv.Canny(gray, edges, *Config.canny)
        #
        gray = cv.CreateImage(cv.GetSize(cvimage), 8, 1)
        #        cv.Sobel(gray, gray, 5, 5)        
        cv.CvtColor(edges, cvimage, cv.CV_GRAY2BGR)

        pyg_image = pygame.image.frombuffer(cvimage.tostring(),
                                            cv.GetSize(cvimage), "RGB")
        display.blit(pyg_image, (0,0))             
        pygame.display.flip()

        continue

        docs = map(lambda x: couchdb.get(x),
                   [ row["id"] for row in couchdb.view("core/screens") ])
        
        device_docs = [ doc for doc in docs if doc["device"] == device ]

        if mode == "record":
            device_docs = [ doc for doc in device_docs if not "features" in doc ]
            try:
                #doc = device_docs.pop()
                print "Click to capture '%s' on device '%s'" % (doc["Name"], doc["device"])
            except IndexError:
                print "No devices to capture"
                capture_features = False
            ########################################
            # perspective option:                  #
            ########################################                
            if perspective:
                n = [ int(i) for i in perspective.split(",") ]
                color, threshold = n[:3], n[3:]
                centroids = centroid_ranges(cvimage, color, threshold)
                if len(centroids) == 4:
                    out = cv.CreateImage(size, cvimage.depth, cvimage.nChannels)
                    cvimage = transform(cvimage, centroids, out)
                    image = cv2pygame(cvimage)
                    for x, y in centroids:
                        coords = (int(y), int(x))
                        pygame.draw.circle(image, (255, 0, 0), coords, 7)
                else:
                    continue
            else:
                image = cv2pygame(cvimage)                    
            ########################################
            # end perspective options.             #
            ########################################
            cvimage = smooth(cvimage)  
            features = featurefinder(cvimage)
            record_rank(features, device_docs)
            if capture_features:
                doc["features"] = features
                couchdb.save(doc)
        ########################################
        # Server mode:                         #
        ########################################                    
        if mode == "server":
        ########################################
        # perspective option:                  #
        ########################################                
            if perspective:
                n = [ int(i) for i in perspective.split(",") ]
                color, threshold = n[:3], n[3]
                centroids = centroid_ranges(cvimage, color, threshold)
                if len(centroids) == 4:
                    cvimage = transform(cvimage, centroids)
                    image = cv2pygame(cvimage)                        
                    for x, y in centroids:
                        coords = (int(y), int(x))
                        pygame.draw.circle(image, (0, 0, 0), coords, 10)
                else:
                    print "Failed to find four centroids required for perspective mode"
                    continue
            else:
                image = cv2pygame(cvimage)                    
            ########################################
            # end perspective options.             #
            ########################################
            image = cv2pygame(cvimage)                
            cvimage = smooth(cvimage)
            features = featurefinder(cvimage)
            best_score, _id, name = server_rank(features,
                                                [ d for d in device_docs if "features" in d ])
            print "Screen is '%s' by score '%s'" % (name, best_score)
            session_docs = map(lambda x: couchdb.get(x),
                               [ row["id"] for row in couchdb.view("core/sessions") ])
            for doc in [ doc for doc in session_docs if doc["device"] == device ]:
                doc["screen_to_display"] = _id
                couchdb.save(doc)
                
        for x, y in features:
            coords = (int(x), int(y))
            pygame.draw.circle(pyg_image, (0,255,0), coords, 5)
        display.blit(pyg_image, (0,0))             
        pygame.display.flip()

