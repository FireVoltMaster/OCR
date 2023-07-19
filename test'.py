import cv2
import os
import sys
import easyocr
import numpy as np


# Function to find nearest point
def find_nearest_point(list1, list2):
    list1 = np.array(list1)
    list2 = np.array(list2)

    distances = np.sqrt(np.sum((list2[:, np.newaxis] - list1) ** 2, axis=2))
    nearest_points = list2[np.argmin(distances, axis=0)]

    return nearest_points.tolist()


def get_key_from_value(dictionary, value):
    for key, val in dictionary.items():
        if val == value:
            return key
    # If the value is not found, you can handle it based on your requirement
    raise ValueError("Value not found in the dictionary")


def find_blue_regions(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to the HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for the blue color (in HSV)
    lower_blue = np.array([110, 50, 50])
    upper_blue = np.array([130, 255, 255])

    # Threshold the image to get only the blue regions
    blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

    # Bitwise-AND the original image with the blue mask to extract the blue regions
    blue_regions = cv2.bitwise_and(image, image, mask=blue_mask)

    return blue_regions


# Function to classify the number of red lines in an image
def classify_red_lines(image_path, opt):
    temp_img = "temp.jpg"

    # Load the image
    image = cv2.imread(image_path)

    # Get the current width and height of the image
    height, width = image.shape[:2]

    # Double the width and height
    desired_width = int(width * 1.5)
    desired_height = int(height * 1.5)

    # Resize the image
    resized_image = cv2.resize(image, (desired_width, desired_height))
    cv2.imwrite(temp_img, resized_image)

    # Convert the image to grayscale
    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    # Apply color thresholding to detect red regions
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Count the number of red lines
    red_line_count = 0
    red_line = []
    for contour in contours:
        # Calculate the area of each contour
        area = cv2.contourArea(contour)

        # If the area is above a certain threshold, consider it as a red line
        if area > 100:  # Adjust the threshold value as per your requirements
            print(contour)
            contour_np = np.squeeze(contour, 1)
            red_line.append(np.average(contour_np, 0))
            red_line_count += 1
    red_line = np.array(red_line)

    if opt == 1:
        # Recognize characters
        reader = easyocr.Reader(['en'], gpu=True)  # this needs to run only once to load the model into memory
        result = reader.readtext(temp_img)
        print(result)
        core_pos = []
        data = {}
        for characs in result:
            if characs[1] == 'C':
                core_pos.append(np.average(characs[0], 0))
                data['C'] = np.average(characs[0], 0).tolist()
            elif characs[1] == 'T':
                core_pos.append(np.average(characs[0], 0))
                data['T'] = np.average(characs[0], 0).tolist()

    else:
        reader = easyocr.Reader(['en'], gpu=True)  # this needs to run only once to load the model into memory
        cv2.imwrite(temp_img, find_blue_regions(temp_img))
        result = reader.readtext(temp_img)
        print(result)
        core_pos = []
        data = {}
        for characs in result:
            if characs[1] == 'C':
                core_pos.append(np.average(characs[0], 0))
                data['C'] = np.average(characs[0], 0).tolist()
            elif characs[1] == 'Pv':
                core_pos.append(np.average(characs[0], 0))
                data['Pv'] = np.average(characs[0], 0).tolist()
            elif characs[1] == 'Pf':
                core_pos.append(np.average(characs[0], 0))
                data['Pf'] = np.average(characs[0], 0).tolist()

    core_pos = np.array(core_pos)

    # Match characters and red-lines
    result = find_nearest_point(red_line, core_pos)
    result_ch = []
    for i in result:
        result_ch.append(get_key_from_value(data, i))
    print(result_ch)

    # Return the predicted red line count
    # print(red_line_count)
    if red_line_count > 1:
        str = "positive: "
        for i in result_ch:
            str = str + i + " "
        return str
    else:
        str = "negative: "
        for i in result_ch:
            str = str + i + " "
        return str


if __name__ == '__main__':
    image_name = "2.jpg"
    image_path = 'image/' + image_name
    image_redline = classify_red_lines(image_path, 1)
    print(image_redline)

