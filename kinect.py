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

def cv_convert(cv_src, to="grey:rgb"):
    p_dict = { "gray:rgb":(cv.IPL_DEPTH_8U, cv.CV_16S, cv.CV_GRAY2RGB),
               "grey:rgb":(cv.IPL_DEPTH_8U, cv.CV_16S, cv.CV_GRAY2RGB),
               "rgb:gray":(cv.IPL_DEPTH_8U, cv.CV_8S,  cv.CV_RGB2GRAY),
               "rgb:hsv" :(cv.IPL_DEPTH_8U, cv.CV_16S, cv.CV_RGB2HSV),
               "hsv:rgb" :(cv.IPL_DEPTH_8U, cv.CV_16S, cv.CV_HSV2RGB),               
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
    cv.CvtColor(cvimage, cvimage_bnw, cv.CV_RGB2GRAY)
    eig_image = cv.CreateImage(cv.GetSize(cvimage_bnw), cvimage.depth, 1)
    temp_image = cv.CreateImage(cv.GetSize(cvimage_bnw), cvimage.depth, 1)
    features = cv.GoodFeaturesToTrack(cvimage_bnw, eig_image, temp_image, n, 0.004, 1.0, useHarris = True) 
    return features # [ (x0, y0), (x1, y1), ... ]

def in_field(cv_video, cv_depth):
    cv_tmp = cv_duplicate(cv_depth)
    cv.Dilate(cv_tmp, cv_tmp, None, 1)
    cv.Erode(cv_tmp, cv_tmp, None, 1)
    cv.And(cv_tmp, cv_video, cv_tmp)
    cv_bnw = cv_convert(cv_tmp, "rgb:gray")
    storage = cv.CreateMemStorage(0)
    contour = cv.FindContours(cv_bnw, storage, cv.CV_RETR_CCOMP, cv.CV_CHAIN_APPROX_SIMPLE)
    while contour:
        bounding_rect = cv.BoundingRect(list(contour))
        cv.DrawContours(cv_video, contour,
                         cv.CV_RGB(255,0,0), cv.CV_RGB(0,0,255), 1000)
        contour = contour.h_next()
        (x, y, w, h) = bounding_rect
        yield bounding_rect

size = (640, 480)
pygame.init()
display = pygame.display.set_mode(size, 0)

cv_video, cv_depth = (cv.CreateImageHeader(size, cv.IPL_DEPTH_8U, cv.CV_16S),
                      cv.CreateImageHeader(size, cv.IPL_DEPTH_8U, cv.CV_16S))

while True:
    depth=600
    near, far=(200, 200)
    (np_video, _), (np_depth, _) = get_video(), get_depth()

    np_depth = 255 * numpy.logical_and(np_depth >= depth - near,
                                       np_depth <= depth + far)
    np_depth = numpy.dstack((np_depth, np_depth, np_depth)).astype(numpy.uint8).reshape(size[1], size[0], 3)

    cv.SetData(cv_depth, np_depth.tostring())
    cv.SetData(cv_video, np_video.tostring())

    # cv_hsv = cv_convert(cv_video, "rgb:hsv")
    # hue = cv.CreateImage(cv.GetSize(cv_hsv), 8, 1)
    # cv.Split(cv_hsv, hue, None, None, None)
    # cv_backproject = cv.CreateImage(cv.GetSize(cv_video), 8, 1)
    # cv.CalcArrBackProject( [hue], cv_backproject, cv_hist )

    kinect_rects = in_field(cv_video, cv_depth)

    for x,y,w,h in kinect_rects:
        p1 = (x, y)
        p2 = (x + w, y + h)
        cv.Rectangle(cv_video, p1, p2, cv.CV_RGB(0,0,255), 1)
        # cv.SetImageROI(cv_video, (x,y,w,h))
        # features = featurefinder(cv_video)
        # for x, y in features:
        #     coord = (int(x), int(y))
        #     cv.Circle(cv_video, coord, 2, cv.CV_RGB(255, 0, 0), 3)
        # cv.ResetImageROI(cv_video)
        

    #     if w * h > 10000 and not track_window:
    #         track_window = (x,y,w,h)            
    #         sub = cv.GetSubRect(cv_video, (x,y,w,h))
    #         save = cv.CloneMat(sub)
    #         cv.ConvertScale(cv_video, cv_video, 0.5)
    #         cv.Copy(save, sub)
    #         cv.Rectangle(cv_video, (x,y), (x+w,y+h), (255,255,255))
    #         sel = cv.GetSubRect(hue, track_window )
    #         cv.CalcArrHist( [sel], cv_hist, 0)
    #         (_, max_val, _, _) = cv.GetMinMaxHistValue(cv_hist)
    #         if max_val != 0:
    #             cv.ConvertScale(cv_hist.bins, cv_hist.bins, 255. / max_val)

    # if track_window and track_window[2] > 0 and track_window[3] > 0:
    #     (iters, (area, value, rect), track_box) = cv.CamShift(cv_backproject, track_window,
    #                                                           ( cv.CV_TERMCRIT_EPS | cv.CV_TERMCRIT_ITER, 10, 1))
    #     (x,y,w,h) = track_window
    #     track_window = rect
    #     sub = cv.GetSubRect(cv_video, (x,y,w,h))
    #     save = cv.CloneMat(sub)
    #     cv.ConvertScale(cv_video, cv_video, 0.5)
    #     cv.Copy(save, sub)
    #     cv.Rectangle(cv_video, (x,y), (x+w,y+h), (0,0,255))
    #     sel = cv.GetSubRect(hue, track_window )
    #     cv.CalcArrHist( [sel], cv_hist, 0)
    #     (_, max_val, _, _) = cv.GetMinMaxHistValue(cv_hist)
    #     if max_val != 0:
    #         cv.ConvertScale(cv_hist.bins, cv_hist.bins, 255. / max_val)
    #         #        cv.EllipseBox(cv_video, track_window, cv.CV_RGB(0,255,255), 1, cv.CV_AA, 0 )


    pyg_image = pygame.image.frombuffer(cv_video.tostring(),
                                        cv.GetSize(cv_video), "RGB")
    display.blit(pyg_image, (0,0))
    pygame.display.flip()


    #         track_window = bound_rect
    #     if track_window:
    #         cv_hsv = cv_convert(cv_rgb, "bgr:hsv")
    #         hue = cv.CreateImage(cv.GetSize(cv_hsv), 8, 1)
    #         cv.Split(cv_hsv, hue, None, None, None)
    #         cv_backproject = cv.CreateImage(cv.GetSize(cv_rgb), 8, 1)
    #         cv.CalcArrBackProject( [hue], cv_backproject, hist )
    #         crit = ( cv.CV_TERMCRIT_EPS | cv.CV_TERMCRIT_ITER, 10, 1)
    #         (iters, (area, value, rect), track_box) = cv.CamShift(cv_backproject, track_window, crit)
    #         p1 = (rect[0], rect[1])
    #         p2 = (rect[0] + rect[2],
    #                rect[1] + rect[3])
    #         cv.Rectangle(cv_rgb, p1, p2, cv.CV_RGB(244,255,0), 1)

        
    # # if len(points):
    # #     center_point = reduce(lambda a, b: ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2), points)
    # #     cv.Circle(cv_rgb, center_point, 40, cv.CV_RGB(255, 255, 255), 1)
    # #     cv.Circle(cv_rgb, center_point, 30, cv.CV_RGB(255, 100, 0), 1)
    # #     cv.Circle(cv_rgb, center_point, 20, cv.CV_RGB(255, 255, 255), 1)
    # #     cv.Circle(cv_rgb, center_point, 10, cv.CV_RGB(255, 100, 0), 1)

    # pyg_image = pygame.image.frombuffer(cv_rgb.tostring(),
    #                                     cv.GetSize(cv_rgb), "RGB")

    #     #    display.fill([0x00, 0x00, 0xff])

    # display.blit(pyg_image, (0,0))
    # pygame.display.flip()
    
