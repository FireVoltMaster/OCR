import cv2
import keras_ocr
import numpy as np


def resize_img(src, tgt, times):
    # Load the image
    image = cv2.imread(src)

    # Get the current width and height of the image
    height, width = image.shape[:2]

    # Double the width and height
    desired_width = int(width * times)
    desired_height = int(height * times)

    # Resize the image
    resized_image = cv2.resize(image, (desired_width, desired_height))
    cv2.imwrite(tgt, resized_image)


def detect_non_gray(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]

    if (np.where(hue > 150, hue, 0).sum() > 0):
        return True
    else:
        return False


def OCR(image, opt):
    temp_img_path = "temp.jpg"
    resize_img(image, temp_img_path, 2)

    pipeline = keras_ocr.pipeline.Pipeline()
    images = [
        keras_ocr.tools.read(temp_img_path)
    ]
    result = pipeline.recognize(images)[0]
    print(result)

    temp_img = cv2.imread(temp_img_path)
    res = ""

    if opt == 1:
        C_flag, T_flag = False, False
        for characs in result:
            if characs[0] == 't':
                T_flag = True
                st_point = np.int32(characs[1][0])
                ed_point = np.int32(characs[1][2])
                width = ed_point[0] - st_point[0]
                tgt_img = temp_img[st_point[1]: ed_point[1], st_point[0] - 3 * width: st_point[0] - 1 * width]
                if detect_non_gray(tgt_img):
                    res = res + "T:Positive    "
                else:
                    res = res + "T:Negative    "
            elif characs[0] == 'c':
                C_flag = True
                st_point = np.int32(characs[1][0])
                ed_point = np.int32(characs[1][2])
                width = ed_point[0] - st_point[0]
                tgt_img = temp_img[st_point[1]: ed_point[1], st_point[0] - 3 * width: st_point[0] - 1 * width]
                if detect_non_gray(tgt_img):
                    res = res + "C:Positive    "
                else:
                    res = res + "C:Negative    "

        if not C_flag:
            res = res + "C:Positive    "
        if not T_flag:
            res = res + "T:Positive    "

    if opt == 2:
        C_flag, Pv_flag, Pf_flag = False, False, False
        for characs in result:
            if characs[0] == 'pv':
                Pv_flag = True
                st_point = np.int32(characs[1][0])
                ed_point = np.int32(characs[1][2])
                width = ed_point[0] - st_point[0]
                tgt_img = temp_img[st_point[1]: ed_point[1], st_point[0] - 3 * width: st_point[0] - 1 * width]
                if detect_non_gray(tgt_img):
                    res = res + "Pv:Positive    "
                else:
                    res = res + "Pv:Negative    "
            elif characs[0] == 'pf':
                Pf_flag = True
                st_point = np.int32(characs[1][0])
                ed_point = np.int32(characs[1][2])
                width = ed_point[0] - st_point[0]
                tgt_img = temp_img[st_point[1]: ed_point[1], st_point[0] - 3 * width: st_point[0] - 1 * width]
                if detect_non_gray(tgt_img):
                    res = res + "Pf:Positive    "
                else:
                    res = res + "Pf:Negative    "
            elif characs[0] == 'c':
                C_flag = True
                st_point = np.int32(characs[1][0])
                ed_point = np.int32(characs[1][2])
                width = ed_point[0] - st_point[0]
                tgt_img = temp_img[st_point[1]: ed_point[1], st_point[0] - 3 * width: st_point[0] - 1 * width]
                if detect_non_gray(tgt_img):
                    res = res + "C:Positive    "
                else:
                    res = res + "C:Negative    "

        if not C_flag:
            res = res + "C:Positive    "
        if not Pv_flag:
            res = res + "Pv:Positive    "
        if not Pf_flag:
            res = res + "Pf:Positive    "

    return res


if __name__ == '__main__':
    image_name = "3.jpg"
    image_path = 'image/' + image_name
    detect_result = OCR(image_path, 1)
    print(detect_result)
