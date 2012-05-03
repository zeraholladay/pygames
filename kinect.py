import cv, pygame, numpy
from pygame.locals import *
from freenect import sync_get_depth as get_depth, sync_get_video as get_video

def cv_duplicate(cv_src):
    size = cv.GetSize(cv_src)
    depth = cv_src.depth
    nChannels = cv_src.nChannels
    cv_dst = cv.CreateImage(size, depth, nChannels)
    cv.Copy(cv_src, cv_dst);
    return cv_dst

def cv_convert(cv_src, to="grey:bgr"):
    p_dict = { "gray:bgr":(cv.IPL_DEPTH_8U, cv.CV_16S, cv.CV_GRAY2RGB),
               "grey:bgr":(cv.IPL_DEPTH_8U, cv.CV_16S, cv.CV_GRAY2RGB),
               "bgr:gray":(cv.IPL_DEPTH_8U, cv.CV_8S,  cv.CV_BGR2GRAY),
               "bgr:hsv": (cv.IPL_DEPTH_8U, cv.CV_16S, cv.CV_BGR2HSV),               
               }
    depth, n_channels, flags = p_dict[to]
    size = cv.GetSize(cv_src)
    cv_dst = cv.CreateImage(size, depth, n_channels)
    cv.CvtColor(cv_src, cv_dst, flags)
    return cv_dst

def cv2pygame(cv_image, fmt="RGB"):
    pyg_image = pygame.image.frombuffer(cv_image.tostring(),
                                        cv.GetSize(cv_image), fmt)
    return cv_image

def numpy2pygame(np_array, fmt="RGB"):
    size = (np_array.shape[1], np_array.shape[0])
    pyg_image = pygame.image.fromstring(np_array.tostring(),
                                        size, "RGB")
    return pyg_image

def numpy2cv(np_array, depth=cv.IPL_DEPTH_8U, n_channels=cv.CV_16S,):
    size = (np_array.shape[1], np_array.shape[0])
    cv_image = cv.CreateImageHeader(size, depth, n_channels)
    cv.SetData(cv_image, np_array.tostring())
    return cv_image

def featurefinder(cvimage, n=150):
    cvimage_bnw = cv.CreateImage(cv.GetSize(cvimage), cvimage.depth, 1)
    cv.CvtColor(cvimage, cvimage_bnw, cv.CV_BGR2GRAY)
    eig_image = cv.CreateImage(cv.GetSize(cvimage_bnw), cvimage.depth, 1)
    temp_image = cv.CreateImage(cv.GetSize(cvimage_bnw), cvimage.depth, 1)
    features = cv.GoodFeaturesToTrack(cvimage_bnw, eig_image, temp_image, n, 0.004, 1.0, useHarris = True) 
    return features # [ (x0, y0), (x1, y1), ... ]

def in_field(cv_video, cv_depth):
    cv.Dilate(cv_depth, cv_depth, None, 10)
    cv.Erode(cv_depth, cv_depth, None, 10)
    cv.And(cv_depth, cv_rgb, cv_depth)

    cv_bnw = cv_convert(cv_depth, "bgr:gray")
    storage = cv.CreateMemStorage(0)

    for contour in cv.FindContours(cv_bnw, storage,
                                   cv.CV_RETR_CCOMP, cv.CV_CHAIN_APPROX_SIMPLE):
        bound_rect = cv.BoundingRect(list(contour))
        yield bounding_rect # (x, y, w, h)

size = (640, 480)
pygame.init()
display = pygame.display.set_mode(size, 0)

cv_video, cv_depth = (cv.CreateImageHeader(size, cv.IPL_DEPTH_8U, cv.CV_16S),
                      cv.CreateImageHeader(size, cv.IPL_DEPTH_8U, cv.CV_16S))

cv_hist = cv.CreateHist([180], cv.CV_HIST_ARRAY, [(0,180)], 1 )

track_window = None

while True:
    depth=500, (near, far)=(500, 500)
    (np_video, _), (np_depth) = get_video(), get_depth()

    np_depth = 255 * numpy.logical_and(np_depth >= depth - near,
                                       np_depth <= depth + far)
    np_depth = numpy.dstack((np_depth, np_depth, np_depth)).astype(numpy.uint8).reshape(size[1], size[0], 3)

    cv.SetData(cv_depth, np_depth.tostring())
    cv.SetData(cv_video, np_video.tostring())

    rectangles = in_field(cv_video, cv_depth)

    for x,y,w,h in rectangles:
        pt1 = (x, y)
        pt2 = (x + w, y + h)
        cv.Rectangle(cv_rgb, pt1, pt2, cv.CV_RGB(0,0,255), 1)

    cv.Dilate(cv_depth, cv_depth, None, 10)
    cv.Erode(cv_depth, cv_depth, None, 10)
    cv.And(cv_depth, cv_rgb, cv_depth)

    cv_bnw = cv_convert(cv_depth, "bgr:gray")

    storage = cv.CreateMemStorage(0)
    contour = cv.FindContours(cv_bnw, storage, cv.CV_RETR_CCOMP, cv.CV_CHAIN_APPROX_SIMPLE)

    while contour:
        bound_rect = cv.BoundingRect(list(contour))
        contour = contour.h_next()
        pt1 = (bound_rect[0], bound_rect[1])
        pt2 = (bound_rect[0] + bound_rect[2], bound_rect[1] + bound_rect[3])
        cv.Rectangle(cv_rgb, pt1, pt2, cv.CV_RGB(0,0,255), 1)
        
        # cv.SetImageROI(cv_rgb, bound_rect)
        # features = featurefinder(cv_rgb)
        # for x, y in features:
        #     coord = (int(x), int(y))
        #     cv.Circle(cv_rgb, coord, 2, cv.CV_RGB(255, 0, 0), 3)
        # cv.ResetImageROI(cv_rgb)

        width, height =  bound_rect[2],  bound_rect[3]

        if not track_window and width > 30 and height > 30:
            track_window = bound_rect
        if track_window:
            cv_hsv = cv_convert(cv_rgb, "bgr:hsv")
            hue = cv.CreateImage(cv.GetSize(cv_hsv), 8, 1)
            cv.Split(cv_hsv, hue, None, None, None)
            cv_backproject = cv.CreateImage(cv.GetSize(cv_rgb), 8, 1)
            cv.CalcArrBackProject( [hue], cv_backproject, hist )
            crit = ( cv.CV_TERMCRIT_EPS | cv.CV_TERMCRIT_ITER, 10, 1)
            (iters, (area, value, rect), track_box) = cv.CamShift(cv_backproject, track_window, crit)
            pt1 = (rect[0], rect[1])
            pt2 = (rect[0] + rect[2],
                   rect[1] + rect[3])
            cv.Rectangle(cv_rgb, pt1, pt2, cv.CV_RGB(244,255,0), 1)

        
    # if len(points):
    #     center_point = reduce(lambda a, b: ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2), points)
    #     cv.Circle(cv_rgb, center_point, 40, cv.CV_RGB(255, 255, 255), 1)
    #     cv.Circle(cv_rgb, center_point, 30, cv.CV_RGB(255, 100, 0), 1)
    #     cv.Circle(cv_rgb, center_point, 20, cv.CV_RGB(255, 255, 255), 1)
    #     cv.Circle(cv_rgb, center_point, 10, cv.CV_RGB(255, 100, 0), 1)

    pyg_image = pygame.image.frombuffer(cv_rgb.tostring(),
                                        cv.GetSize(cv_rgb), "RGB")

        #    display.fill([0x00, 0x00, 0xff])

    display.blit(pyg_image, (0,0))
    pygame.display.flip()
    
