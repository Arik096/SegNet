import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import itertools
import operator
import os, csv
import tensorflow as tf
import time, datetime
Sky = [128,128,128]
Building = [128,0,0]
Pole = [192,192,128]
Road_marking = [255,69,0]
Road = [128,64,128]
Pavement = [60,40,222]
Tree = [128,128,0]
SignSymbol = [192,128,128]
Fence = [64,64,128]
Car = [64,0,128]
Pedestrian = [64,64,0]
Bicyclist = [0,128,192]
Unlabelled = [0,0,0]
label_values = [Sky,Building,Pole,Road,Pavement,Tree,
                SignSymbol,Fence,Car,Pedestrian,Bicyclist,Unlabelled]
import cv2                  
from keras.preprocessing.image import *
from tqdm import tqdm_notebook, tnrange
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
def load_images(img_len,img_wid, img_path):
    imgs = []
    for img in tqdm(os.listdir(img_path)):
        path = os.path.join(img_path,img)
        img = cv2.imread(path,cv2.IMREAD_COLOR)
        img = cv2.resize(img, (img_wid,img_len))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        imgs.append(np.array(img))
    return(imgs)
def image_preprocessing(X_train,X_train_masks,X_val,X_val_masks,img_size,
                       img_path,mask_path,val_path,val_mask_path):
    data_gen_args = dict(rescale=1./255,
                    width_shift_range=0.25,
                    height_shift_range=0.25,
                    zoom_range=0.15,
                    horizontal_flip=True)
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    val_datagen = ImageDataGenerator(**data_gen_args)
    val_masks =  ImageDataGenerator(**data_gen_args)
    seed = 1
    image_datagen.fit(X_train, augment=True, seed=seed)
    mask_datagen.fit(X_train_masks, augment=True, seed=seed)
    seed_val = 7
    val_datagen.fit(X_train, augment=True, seed=seed_val)
    val_masks.fit(X_train_masks, augment=True, seed=seed_val)
    image_generator = image_datagen.flow_from_directory(img_path,
    batch_size = 1,
    class_mode=None,
    target_size=img_size,
    seed=seed)
    mask_generator = mask_datagen.flow_from_directory(mask_path,
    batch_size =1,
    class_mode=None,
    target_size=img_size,
    seed=seed)
    val_generator = val_datagen.flow_from_directory(val_path,
    batch_size = 1,
    class_mode=None,
    target_size=img_size,
    seed=seed_val)
    val_mask_generator = val_masks.flow_from_directory(val_mask_path,
    batch_size = 1,
    class_mode=None,
    target_size=img_size,
    seed=seed_val)
    train_generator = zip(image_generator, mask_generator)
    val_generator =zip(val_generator,val_mask_generator) 
    return (train_generator,val_generator)
def plot_predictions(X_test,preds,path):
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(12,5))
    ax[0].imshow(X_test,aspect="auto")
    ax[0].set_title("Input")
    ax[1].imshow(preds, aspect="auto")
    ax[1].set_title("Prediction")
    fig.tight_layout()    
def equalize_hist(img):
    img[:,:,0] = cv2.equalizeHist(img[:,:,0])
    img[:,:,1] = cv2.equalizeHist(img[:,:,1])
    img[:,:,2] = cv2.equalizeHist(img[:,:,2])
    return img
def get_label_info(csv_path):
    filename, file_extension = os.path.splitext(csv_path)
    if not file_extension == ".csv":
        return ValueError("File is not a CSV!")
    class_names = []
    label_values = []
    with open(csv_path, 'r') as csvfile:
        file_reader = csv.reader(csvfile, delimiter=',')
        header = next(file_reader)
        for row in file_reader:
            class_names.append(row[0])
            label_values.append([int(row[1]), int(row[2]), int(row[3])])
    return class_names, label_values
def one_hot_it(label, label_values):
    semantic_map = []
    for colour in label_values:
        equality = np.equal(label, colour)
        class_map = np.all(equality, axis = -1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)
    return semantic_map  
def reverse_one_hot(image):
    x = np.argmax(image, axis = -1)
    return x
def colour_code_segmentation(image, label_values):
    colour_codes = np.array(label_values)
    x = colour_codes[image.astype(int)]
    return x
def median_frequency_balancing(image_files, num_classes=len(label_values)):
    label_to_frequency_dict = {}
    for i in range(num_classes):
        label_to_frequency_dict[i] = []
    for n in range(image_files.shape[0]):
        image = image_files[n]
        for i in range(num_classes):
            class_mask = np.equal(image, i)
            class_mask = class_mask.astype(np.float32)
            class_frequency = np.sum(class_mask)

            if class_frequency != 0.0:
                label_to_frequency_dict[i].append(class_frequency)
    class_weights = []
    total_pixels = 0
    for frequencies in label_to_frequency_dict.values():
        total_pixels += sum(frequencies)
    for i, j in label_to_frequency_dict.items():
        j = sorted(j) 
        median_frequency = np.median(j) / sum(j)
        total_frequency = sum(j) / total_pixels
        median_frequency_balanced = median_frequency / total_frequency
        class_weights.append(median_frequency_balanced)
    return class_weights