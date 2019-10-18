from collections import namedtuple
import numpy as np
import argparse
import cv2


SavedResult = namedtuple('SavedResult', ['confidence', 'detection_bbox'])

font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
fontColor = (255, 255, 255)
lineType = 2


def resize(image, width=None, height=None):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        ratio = height / float(h)
        dim = (int(w * ratio), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        ratio = width / float(w)
        dim = (width, int(h * ratio))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    # return the resized image
    return resized, ratio


def prepare_image(image_):
    image_ = cv2.cvtColor(image_, cv2.COLOR_BGR2GRAY)
    return image_


def prepare_template(template_):
    template_ = cv2.cvtColor(template_, cv2.COLOR_BGR2GRAY)
    template_ = cv2.Canny(template_, 50, 200)
    return template_


def find_template(image, template):
    found = None

    for t_scale in np.linspace(0.5, 1.0, 10)[::-1]:
        r_template, _ = resize(template, width=int(template.shape[1] * t_scale))
        (t_height, t_width) = r_template.shape[:2]

        for scale in np.linspace(0.2, 1.0, 20)[::-1]:
            resized, resize_ratio = resize(image, width=int(image.shape[1] * scale))

            if resized.shape[0] < t_height or resized.shape[1] < t_width:
                break

            edged = cv2.Canny(resized, 50, 200)
            match_result = cv2.matchTemplate(edged, r_template, cv2.TM_CCORR_NORMED)
            (_, maxVal, _, maxLoc) = cv2.minMaxLoc(match_result)

            if found is None or found.confidence < maxVal:
                (startX, startY) = (int(maxLoc[0] * resize_ratio), int(maxLoc[1] * resize_ratio))
                (endX, endY) = (int((maxLoc[0] + t_width) * resize_ratio), int((maxLoc[1] + t_height) * resize_ratio))
                bbox = ((startX, startY), (endX, endY))
                found = SavedResult(confidence=maxVal,
                                    detection_bbox=bbox)
    return found


def draw_match(image, bbox, name=None):
    # bbox == (startX, startY), (endX, endY)
    cv2.rectangle(image, *bbox, (0, 0, 255), 2)
    if name is not None:
        cv2.putText(image, name,
                    bbox[0],
                    font,
                    fontScale,
                    fontColor,
                    lineType)


if __name__ == '__main__':
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--template", default='./eco_icons/tidyman.jpg', help="Path to template image")
    ap.add_argument("-i", "--image", default='./new_images/IMG_20191016_224337.jpg',
                    help="Path to image where template will be matched")
    args = vars(ap.parse_args())
    template = cv2.imread(args["template"])
    image = cv2.imread(args["image"])
    prep_image, prep_template = prepare_image(image), prepare_template(template)
    result = find_template(prep_image, prep_template)
    print(result)
    if result is not None:
        draw_match(image, result.detection_bbox)
        cv2.imshow("Image", image)
        cv2.waitKey(2000)
