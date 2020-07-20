from Net import *
from utils import *
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
from keras.utils import plot_model
import IPython
%matplotlib inline
%run ./Net.py

img_size = (360,480)

img_path = './CamVid/train'
mask_path = './CamVid/trainannot'
val_path = './CamVid/val'
val_masks = './CamVid/valannot'

X_train = load_images(img_size[0],img_size[1],img_path= img_path)
X_train_masks = load_images(img_size[0],img_size[1],img_path=mask_path)
X_val = load_images(img_size[0],img_size[1],val_path)
X_val_masks = load_images(img_size[0],img_size[1],val_masks)

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

# We dont want to classify unlabelled.it will only make our dataset more unbalanced than it already is
label_values = [Sky,Building,Pole,Road,Pavement,Tree,
                SignSymbol,Fence,Car,Pedestrian,Bicyclist,Unlabelled]

X_train_norm = [equalize_hist(X_train[i]) for i in range(len(X_train))]
X_val_norm = [equalize_hist(X_val[i]) for i in range(len(X_val))]
X_train_norm = np.array(X_train_norm)
X_val_norm = np.array(X_val_norm)
np.save("./Training/X_train.npy",X_train_norm)
np.save("./Training/X_val_norm.npy",X_val_norm)

y_train_norm = [one_hot_it(X_train_masks[i],label_values) for i in range(len(X_train_masks))]
y_train_val = [one_hot_it(X_val_masks[i],label_values) for i in range(len(X_val_masks))]
y_train_norm = np.array(y_train_norm)
y_train_val = np.array(y_train_val)
np.save("./Training/y_train.npy",y_train_norm)
np.save("./Training/y_val.npy",y_train_val)

X_train = np.load("./Training/X_train.npy")
y_train_norm = np.load("./Training/y_train.npy")
X_val = np.load("./Training/X_val_norm.npy")
y_val = np.load("./Training/y_val.npy")

X_oh = reverse_one_hot(y_train_norm)
weights = median_frequency_balancing(X_oh)

from keras.callbacks import ModelCheckpoint
from livelossplot.keras import PlotLossesCallback
fpathmodel="./logseg/seg1.hdf5"
cp = ModelCheckpoint(fpathmodel, monitor='acc', verbose=2, save_best_only=True, mode='min') 
callbacks_list = [cp,PlotLossesCallback()]

seg = arnet(img_size=(360,480,3),classes=12,optimizer='adam',loss_function='binary_crossentropy',metric=['acc'])

y_train_norm = y_train_norm.reshape((y_train_norm.shape[0],y_train_norm.shape[1]*y_train_norm.shape[2],
                                    y_train_norm.shape[3]))

y_val = y_val.reshape((y_val.shape[0],y_val.shape[1]*y_val.shape[2],
                                    y_val.shape[3]))

seg.fit(X_train,y_train_norm, 
                  validation_data=(X_val,y_val),batch_size=1,shuffle=True,
                  epochs=20, verbose=1, callbacks=callbacks_list,class_weight=weights)

loss = seg.history.history['loss']
val_loss = seg.history.history['val_loss']
acc = seg.history.history['acc']
val_acc = seg.history.history['val_acc']

epochs = range(1,len(loss)+1)

plt.plot(epochs,loss,'bo',label = 'Training Loss')
plt.plot(epochs,val_loss ,'b',label = 'Validation Loss' ,color = 'r')
plt.title('Training and validation Dice Coefficient Loss')
plt.legend()
plt.show()

plt.plot(epochs,acc,'bo',label = 'Training Accuracy')
plt.plot(epochs,val_acc ,'b',label = 'Validation Accuracy' ,color = 'r')
plt.title('Training and validation Accuracy')
plt.legend()


plt.figure()

