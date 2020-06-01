# main Script for loading the Text Images and fetching the text out of the scanned document

import cv2, numpy as np, os
import pickle
from keras.models import load_model


class Document():
    def __init__(self):
        self.dfNames = ['FirstName', 'LastName', 'Email', 'Street', 'City', 'State', 'ZipCode', 'Phone', 'BirthDay']
        # self.modelPath = modelPath
        self.path = os.path.dirname(os.path.abspath('__file__')) + '/'
        self.path = self.path.replace('\\', '/')
        alphaNumericModel_path = self.path + 'Models/AlphabetNumeric_v3.h5'
        alphabetModel_path = self.path + 'Models/Alphabets_v3.h5'
        digitModel_path = self.path + 'Models/Digit_v2.h5'
        atr_Model_path = self.path + 'Models/@.h5'
        _modelpath = self.path + 'Models/_model.h5'

        """#This is General model for everything(alpabets,digits and Special characters"""
        self.alphaNumericModel = load_model(alphaNumericModel_path)
        self.alphabetModel = load_model(alphabetModel_path)
        self.digitModel = load_model(digitModel_path)
        self.atrModel = load_model(atr_Model_path)
        self._model = load_model(_modelpath)

        self.model = None
        self.label = None
        self.checkForATR = False
        # getting the labels from pickled file #CNN_OCRModel_v1.h5
        with open(self.path + 'Models/AlphabetNumericLabels_v3.pkl', 'rb') as f:
            self.alphaNumericLabels = pickle.load(f)
        self.alphaNumericLabels = dict([(v, k) for k, v in self.alphaNumericLabels.items()])

        with open(self.path + 'Models/AlphabetsLabels_v3.pkl', 'rb') as f:
            self.alphabetLabel = pickle.load(f)
        self.alphabetLabel = dict([(v, k) for k, v in self.alphabetLabel.items()])

        with open(self.path + 'Models/Digit_v2.pkl', 'rb') as f:
            self.digitLabel = pickle.load(f)
        self.digitLabel = dict([(v, k) for k, v in self.digitLabel.items()])

        with open(self.path + 'Models/label2_1.pickle', 'rb') as f:
            self.atrLabel = pickle.load(f)
        self.atrLabel = dict([(v, k) for k, v in self.atrLabel.items()])

        # getting the labels from pickled file #CNN_OCRModel_v1.h5
        with open(self.path + 'Models/_Labels.pkl', 'rb') as f:
            self._label = pickle.load(f)
        self._label = dict([(v, k) for k, v in self._label.items()])

        self.list_Character_Positions = []
        self.count = 0

    def getCountours(self, input_Image):

        x, contours, hierarchy = cv2.findContours(input_Image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def sortCountours(self, cnts, method="left-to-right"):

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

    def getNewResizedImage(self, input_Image, image_Size):

        height, width = input_Image.shape
        # print (height, width)

        if width > height:
            aspect_Ratio = (float)(width / height)
            width = 22
            height = round(width / aspect_Ratio)
        else:
            aspect_Ratio = (float)(height / width)
            height = 22
            width = round(height / aspect_Ratio)

        input_Image = cv2.resize(input_Image, (width, height), interpolation=cv2.INTER_AREA)

        height, width = input_Image.shape

        number_Of_Column_To_Add = 30 - width
        temp_Column = np.zeros((height, int(number_Of_Column_To_Add / 2)), dtype=np.uint8)
        input_Image = np.append(temp_Column, input_Image, axis=1)
        input_Image = np.append(input_Image, temp_Column, axis=1)

        height, width = input_Image.shape
        number_Of_Row_To_Add = 30 - height
        temp_Row = np.zeros((int(number_Of_Row_To_Add / 2), width), dtype=np.uint8)
        input_Image = np.concatenate((temp_Row, input_Image))
        input_Image = np.concatenate((input_Image, temp_Row))

        return cv2.resize(input_Image, (image_Size, image_Size), interpolation=cv2.INTER_AREA)

    def getTextFromImage(self, flag, image):

        alphabetPrediction = ''
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, thres = cv2.threshold(img.copy(), 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

        # number : represents wideness of the dilation. i.e for line level or word level or character level.
        kernel = np.ones((3, 3), np.uint8)
        dil_image = cv2.dilate(thres, kernel, iterations=1)

        c = self.getCountours(dil_image)
        c = self.sortCountours(c, "left-to-right")
        for c1 in c:
            area = cv2.contourArea(c1)
            # print("area=", area)
            if area < 110:
                continue
            x, y, w, h = cv2.boundingRect(c1)
            # print(w, h)
            if 290 < area < 360 and w < 60 and h < 20:
                alphabetPrediction = alphabetPrediction + '_'
                continue
            elif 150 < area < 220 and w < 20 and h < 20:
                alphabetPrediction = alphabetPrediction + '.'
                continue

            else:
                if len(dil_image[y - 1:y + h + 1, x - 1:x + w + 1]) == 0:
                    continue
                resize_i = cv2.resize(dil_image[y - 1:y + h + 1, x - 1:x + w + 1], (32, 32))

                resize_img = cv2.resize(resize_i, (28, 28))
                resize_image = Document.getNewResizedImage(self, resize_i, 32)

                self.count += 1

                if flag == 'email':
                    # print('email Model Selected..!!')
                    self.model = load_model("Models/sp_char_model1.h5")
                    self.label = self.atrLabel

                    resize = np.expand_dims(resize_img, axis=0)
                    resize = np.expand_dims(resize, axis=-1)
                    charProb = self.model.predict(resize)[0]
                    index = np.argmax(charProb)
                    char = self.label[index]

                    if char == 'junk':
                        self.model = self.alphaNumericModel
                        self.label = self.alphaNumericLabels
                        flag = 'alphanumeric'
                    else:
                        alphabetPrediction = alphabetPrediction + char

                if flag == 'alphabet':
                    # print('Alphabet Model Selected..!')
                    self.model = self.alphabetModel
                    self.label = self.alphabetLabel

                elif flag == 'digit':
                    # print('DIGIT Model Selected..!!')
                    self.model = self.digitModel
                    self.label = self.digitLabel

                else:
                    # print('AlphaNumeric Model Selected..!!')
                    self.model = self.alphaNumericModel
                    self.label = self.alphaNumericLabels

                resize_image = resize_image.reshape(1, 32, 32, 1)
                charProb = self.model.predict(resize_image)
                index = int(np.argmax(charProb))
                char = self.label[index]
                # print("char=", char)
                alphabetPrediction = alphabetPrediction + char

        return alphabetPrediction
