from zipfile import ZipFile

from PIL import Image, ImageDraw
import pytesseract
import cv2 as cv
import numpy as np
from kraken import pageseg
import math

# loading the face detection classifier
face_cascade = cv.CascadeClassifier('readonly/haarcascade_frontalface_default.xml')

keyword_1 = "Christopher"
file_1 = "readonly/small_img.zip"

keyword_2 = "Mark"
file_2 = "readonly/images.zip"

# the rest is up to you!
''' Class which stores an image and knows how to scan itself and save scanned results internally'''
class MyImageData:
    def __init__(self, file_name, pil_img):
        self.file_name = file_name
        self.pil_img = pil_img
        self.face_boxes = None
        self.text_boxes = None
        self.text = None

    def ocr_me(self):
        self.text = pytesseract.image_to_string(self.pil_img.convert("L"))
    
    # Face recognition can be made better and better
    def face_box_me(self, scale=1.0):
        cv_img= cv.cvtColor(np.array(self.pil_img), cv.COLOR_RGB2BGR)
#         cv_img = cv.cvtColor(cv_img, cv.COLOR_RGB2GRAY)
#         self.face_boxes = face_cascade.detectMultiScale(cv_img, scale)
        self.face_boxes = face_cascade.detectMultiScale(cv_img)

    def show(self):
        print(self.file_name)
        display(self.pil_img)
        
    def show_with_face_box(self):
        print(self.file_name)
        drawing=ImageDraw.Draw(self.pil_img)
        # And iterate through the faces sequence, tuple unpacking as we go
        for x,y,w,h in self.face_boxes:
        # And remember this is width and height so we have to add those appropriately.
            drawing.rectangle((x,y,x+w,y+h), outline="white")
        display(self.pil_img)

''' Loads the images from the zip file and makes a dictionary of images'''
def build_zip_file_images(z_file_name):
    ret_dict = {}
    with ZipFile(z_file_name, 'r') as z_file:
        for afile in z_file.infolist():
            ret_dict[afile.filename] = MyImageData(afile.filename, Image.open(z_file.open(afile)))
            
    return ret_dict

''' Scans the images for text and faces and stores them in the dictionary along with the image'''
def scan_images(img_dict):
    for anImage in img_dict.values():
        anImage.ocr_me()
        anImage.face_box_me()

''' Resizes the faces in an image to fit in a 100x100 box and displays it blindly 
    OR
    prints a message if there are no faces
    This assumes that the image has already been scanned
'''
def display_faces_strip(my_img_data):
    if my_img_data.face_boxes is None or len(my_img_data.face_boxes) == 0:
        print("But there were no faces in file {} !".format(my_img_data.file_name))
    else:
        print("Faces in file {} below !".format(my_img_data.file_name))
        new_size = 100
        pics_in_row = 5
        curr_idx = 0
        strip_image = Image.new("RGB", (new_size*pics_in_row, new_size*math.floor(len(my_img_data.face_boxes) / pics_in_row) + 1))
        for aBox in my_img_data.face_boxes:
            x, y, w, z = aBox
            new_x, new_y = (100*(curr_idx % pics_in_row), 100*math.floor(curr_idx / pics_in_row))
            strip_image.paste(my_img_data.pil_img.crop((x, y, x+w, y+w)).resize((new_size,new_size)), box=(new_x, new_y))
            curr_idx += 1
    
        display(strip_image)

''' Loops through all the processed images in a dictionary if keyword is there then displays faces strip '''
def display_face_if_keyword(img_dict, keyword):
    for anImage in img_dict.values():
        if keyword in anImage.text:
            print("Keyword '{}' found in file {}".format(keyword, anImage.file_name))
            display_faces_strip(anImage)
        else:
            print("Keyword '{}' not found in file {}".format(keyword, anImage.file_name))

''' the main calling function - Takes a keyword and a file name and calls all functions in order'''
def display_faces_with_keyword_in_zipfile(in_keyword, z_file_name):
    img_dict = build_zip_file_images(z_file_name)
    scan_images(img_dict)
    display_face_if_keyword(img_dict, in_keyword)
    return # ret_images_dict
    
display_faces_with_keyword_in_zipfile(keyword_1, file_1)

display_faces_with_keyword_in_zipfile(keyword_2, file_2)
