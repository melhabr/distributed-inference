import numpy as np
import cv2

# Based on JK Jung's visualization function
# https://github.com/jkjung-avt/tensorrt_demos/blob/master/utils/visualization.py

def draw_label(img, bbox, id, conf):
    img_h, img_w, _ = img.shape

    if bbox[0] >= img_w or bbox[1] >= img_h:
        return

    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (232, 35, 244), 2)

    text = "{} {}".format(id, round(conf, 2))

    margin = 3
    size = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 1.0, 1)
    w = size[0][0] + margin * 2
    h = size[0][1] + margin * 2

    patch = np.zeros((h, w, 3), dtype=img.dtype)
    patch[...] = (232, 35, 244)
    cv2.putText(patch, text, (margin + 1, h - margin - 2), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255),
                lineType=cv2.LINE_8)
    cv2.rectangle(patch, (0, 0), (w - 1, h - 1), (0, 0, 0), thickness=1)

    w = min(w, img_w - bbox[0])
    h = min(h, img_h - bbox[1])
    roi = img[bbox[1]:bbox[1] + h, bbox[0]:bbox[0] + w, :]
    cv2.addWeighted(patch[0:h, 0:w, :], 0.5, roi, 0.5, 0, roi)
