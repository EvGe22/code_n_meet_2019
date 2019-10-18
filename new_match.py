# import the necessary packages
import numpy as np
import argparse
import imutils
import glob
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--template", required=True, help="Path to template image")
ap.add_argument("-i", "--images", required=True,
                help="Path to images where template will be matched")
ap.add_argument("-v", "--visualize",
                help="Flag indicating whether or not to visualize each iteration")
args = vars(ap.parse_args())

# load the image image, convert it to grayscale, and detect edges
template = cv2.imread(args["template"])
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
# template = 255 - template
template = cv2.Canny(template, 50, 200)
(tH, tW) = template.shape[:2]

# loop over the images to find the template in
#imagePath = list(glob.glob(args["images"] + "/*.jpg"))[0]

# imagePath = '/home/ilya/PycharmProjects/meet_code/new_images/IMG_20191016_223907.jpg'
# imagePath = '/home/ilya/PycharmProjects/meet_code/new_images/IMG_20191016_224015.jpg'
# imagePath = '/home/ilya/PycharmProjects/meet_code/new_images/IMG_20191016_224151.jpg'
# imagePath = '/home/ilya/PycharmProjects/meet_code/new_images/IMG_20191016_224248_001_COVER.jpg'

# tidyman
# imagePath = '/home/ilya/PycharmProjects/meet_code/new_images/IMG_20191016_224248_001_COVER.jpg'
imagePath = '/home/ilya/PycharmProjects/meet_code/new_images/IMG_20191016_224337.jpg'

# load the image, convert it to grayscale, and initialize the
# bookkeeping variable to keep track of the matched region
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
found = None
found_min = None

# loop over the scales of the image
for t_scale in np.linspace(0.5, 1.0, 20)[::-1]:
    r_template = imutils.resize(template, width=int(template.shape[1] * t_scale)) #, inter=cv2.INTER_BILLINEAR)
    r_t = gray.shape[1] / float(r_template.shape[1])
    # cv2.imshow("Image", r_template)
    # cv2.waitKey(500)
    for scale in np.linspace(0.2, 1.0, 20)[::-1]:
        # resize the image according to the scale, and keep track
        # of the ratio of the resizing
        resized = imutils.resize(gray, width=int(gray.shape[1] * scale))
        r = gray.shape[1] / float(resized.shape[1])

        # if the resized image is smaller than the template, then break
        # from the loop
        if resized.shape[0] < tH or resized.shape[1] < tW:
            break
        # detect edges in the resized, grayscale image and apply template
        # matching to find the template in the image
        edged = cv2.Canny(resized, 50, 200)
        # cv2.TM_CCOEFF_NORMED
        # cv2.TM_CCORR_NORMED
        # cv2.TM_SQDIFF_NORMED
        result = cv2.matchTemplate(edged, r_template, cv2.TM_CCORR_NORMED)
        new_image = np.zeros((edged.shape[0], edged.shape[1]*2))
        new_image[:, :edged.shape[1]] = edged
        result_reshaped = cv2.resize(result, edged.shape[::-1])
        new_image[:, edged.shape[1]:] = result_reshaped
        # cv2.imshow("Image", new_image)
        # cv2.waitKey(20)

        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(result)

        # check to see if the iteration should be visualized
        if args.get('visualize') is not None:
            # draw a bounding box around the detected region
            clone = np.dstack([edged, edged, edged])
            cv2.rectangle(clone, (maxLoc[0], maxLoc[1]),
                          (maxLoc[0] + tW, maxLoc[1] + tH), (0, 0, 255), 2)
            cv2.imshow("Visualize", clone)

        # if we have found a new maximum correlation value, then update
        # the bookkeeping variable
        if found is None or maxVal > found[0]:
            found = (maxVal, maxLoc, r)
        if found_min is None or minVal < found[0]:
            found_min = (minVal, minLoc, r)

    # unpack the bookkeeping variable and compute the (x, y) coordinates
    # of the bounding box based on the resized ratio
if found is not None:
    (_, maxLoc, r) = found
    (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
    (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))

    # draw a bounding box around the detected result and display the image
    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
    cv2.imshow("Image", image)
    cv2.waitKey(1000)
