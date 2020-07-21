import numpy as np
import cv2


def read_labels(path):
    labels = {}
    f = open(path, "r")
    for line in f:
        s = line.split()
        labels.update({int(s[0]): " ".join(s[1:])})
    f.close()
    return labels


# Based on JK Jung's visualization function
# https://github.com/jkjung-avt/tensorrt_demos/blob/master/utils/visualization.py

def draw_labels(img, proposals, labels=None):
    img_h, img_w, _ = img.shape

    for prop in proposals:
        if prop[0] >= img_w or prop[1] >= img_h:
            continue

        cv2.rectangle(img, (prop[0], prop[1]), (prop[2], prop[3]), (232, 35, 244), 2)

        text = "{} {}".format(labels[prop[4]] if labels else prop[4], round(prop[5], 2))

        margin = 3
        size = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 1.0, 1)
        w = size[0][0] + margin * 2
        h = size[0][1] + margin * 2

        patch = np.zeros((h, w, 3), dtype=img.dtype)
        patch[...] = (232, 35, 244)
        cv2.putText(patch, text, (margin + 1, h - margin - 2), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255),
                    lineType=cv2.LINE_8)
        cv2.rectangle(patch, (0, 0), (w - 1, h - 1), (0, 0, 0), thickness=1)

        w = min(w, img_w - prop[0])
        h = min(h, img_h - prop[1])
        roi = img[prop[1]:prop[1] + h, prop[0]:prop[0] + w, :]
        cv2.addWeighted(patch[0:h, 0:w, :], 0.5, roi, 0.5, 0, roi)
