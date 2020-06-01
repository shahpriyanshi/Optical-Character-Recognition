from __future__ import division
from __future__ import print_function

import os
import cv2
import numpy as np
from Models import word_classification_model
import pytesseract
from ParseDocument import Document
from pytesseract import Output


class FilePaths:
    "filenames and paths to data"
    fnCharList = '../model/charList.txt'
    fnAccuracy = '../model/accuracy.txt'
    fnTrain = '../data/'

    # corpus is a word dictionary
    fnCorpus = '../data/corpus.txt'


def processedImage(image):

    # Pre-processing the image

    img = cv2.GaussianBlur(image, (7, 7), 0)
    ret, thresh = cv2.threshold(img.copy(), 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    erode = cv2.erode(thresh, np.ones((3, 3), np.uint8), iterations=1)
    dilate = cv2.dilate(erode, np.ones((3, 3), np.uint8), iterations=1)
    # autoCanny = imutils.auto_canny(dilate.copy(), sigma=0)

    return dilate


def dilateImage(image, number):

    # number : represents wideness of the dilation. i.e for line level or word level or character level.
    kernel = np.ones((9, number),
                     np.uint8)
    img_dilation = cv2.dilate(image, kernel, iterations=3)
    return img_dilation


def getCountours(input_Image):

    a, contours, u = cv2.findContours(input_Image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours


def sortCountours(cnts, method="left-to-right"):

    # initializing the reverse flag and sorting index
    reverse = False

    i = 0
    # handling the flag if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))

    # return the list of sorted contours and bounding boxes
    return (cnts)


def getWordFromImage(image):

    im = image.copy()
    image = processedImage(image)
    kernel = np.ones((9, 13), np.uint8)
    Word_Dilated_Image = cv2.dilate(image, kernel, iterations=3)

    # cv2.imshow("word_dil", Word_Dilated_Image)

    Word_Contours = getCountours(Word_Dilated_Image)
    Word_Contours = sortCountours(Word_Contours, "left-to-right")

    return Word_Contours, im


classification_model = word_classification_model.Hand_and_Printed_Classifier.build(width=64, height=48, depth=3)
classification_model.load_weights("../src/Models/weights1.h5")
obj = Document()


def wordSegmentation(original_img):
    # the main function for segmenting words from pages and lines of an image and predicting them.

    # A copy for backup
    image_ori = original_img.copy()

    image = processedImage(original_img)

    image_with_lines = dilateImage(image.copy(), 150)

    contours = getCountours(image_with_lines.copy())

    contours = sortCountours(contours, "top-to-bottom")
    for cont in contours:

        prediction = []

        # extracting line from a page
        x, y, w, h = cv2.boundingRect(cont)

        word_cnt, line_image = getWordFromImage(image_ori[y:y + h, x:x + w])
        for c1 in word_cnt:

            # extracting the words from the lines
            x1, y1, w1, h1 = cv2.boundingRect(c1)
            word_image = line_image[y1:y1 + h1, x1:x1 + w1]
            word_image_original = word_image.copy()

            word_image = cv2.cvtColor(word_image, cv2.COLOR_GRAY2RGB)
            word_gray = word_image.copy()
            word_image = cv2.resize(word_image, (64, 48))

            ex_img = np.expand_dims(word_image, axis=0)

            # Model to classify handwritten and printed words/characters
            # (if prediction>0.5 Printed else handwritten)
            preds = classification_model.predict(ex_img)

            if preds[0] > 0.5:
                d = pytesseract.image_to_data(word_image_original, output_type=Output.DICT)

                for i in d['text']:

                    # Rejecting Empty spaces
                    list1 = ['', ' ', '  ']
                    if i not in list1:
                        prediction.append(i)
            else:

                # Zip Code, Birthday, Phone Number contain only digits
                list1 = ['z', 'Z', 'B', 'b', 'p', 'P']

                # First Name, Last Name and City in forms only contain Alphabets
                list2 = ['F', 'f', 'l', 'L', 'C', 'c']

                if not prediction:
                    flag_internal = 'x'
                else:
                    first_char = list(prediction[0])
                    if first_char[0] in list1:
                        flag_internal = 'digit'

                    elif first_char[0] in list2:
                        flag_internal = 'alphabet'

                    elif first_char[0] in ['E', 'e']:
                        flag_internal = 'email'
                    else:
                        flag_internal = 'alphanumeric'

                text = obj.getTextFromImage(flag_internal, word_gray)
                prediction.append(text)

        print("prediction=", prediction)


def gray_and_resize(img, height):
    """convert given image to grayscale image (if needed) and resize to desired height"""

    h = img.shape[0]
    factor = height / h

    return cv2.resize(img, dsize=None, fx=factor, fy=factor)


def main():

    # path to the test directory(images could contain handwritten or printed characters)
    for i in os.listdir("../data/testing_data"):
        print(i)
        fnInfer = "../data/testing_data/" + i
        img = cv2.imread(fnInfer, 0)

        wordSegmentation(img)


if __name__ == '__main__':
    main()
