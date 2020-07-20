from keras.models import Sequential,Model
from keras.layers import Dense, Activation ,Conv2D,MaxPooling2D ,Flatten , Conv2DTranspose ,Input,Concatenate,Reshape,ZeroPadding2D,UpSampling2D, Dropout
from keras.layers.normalization import BatchNormalization
from keras import initializers
from keras.regularizers import l2
from keras.applications.vgg16 import VGG16
def arnet(img_size,classes,optimizer,loss_function,metric):
    inputs = Input(img_size)  
    encoder = VGG16(include_top=False, weights='imagenet',input_tensor=inputs,input_shape=img_size)
    for l in encoder.layers:
        l.trainable = False  
    layers = dict([(layer.name, layer) for layer in encoder.layers])  
    vgg_base = layers['block5_pool'].output
    
    up1 = UpSampling2D(size=(2,2))(vgg_base)
    conv1 = Conv2D(kernel_size=(3,3),filters=512,padding='same'
                   ,kernel_initializer='he_normal',kernel_regularizer=l2(0.0001))(up1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    
    conv2 = Conv2D(kernel_size=(3,3),filters=512,padding='same',kernel_initializer='he_normal',kernel_regularizer=l2(0.0001))(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)    
           
    conv3 = Conv2D(kernel_size=(3,3),filters=512,padding='same',kernel_initializer='he_normal',kernel_regularizer=l2(0.0001))(conv2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    
    up2 = UpSampling2D(size=(2,2))(conv3)
    pad2 = ZeroPadding2D(padding=((1, 0), (0, 0)))(up2)
    
    conv4 = Conv2D(kernel_size=(3,3),filters=512,padding='same'
                   ,kernel_initializer='he_normal',kernel_regularizer=l2(0.0001))(pad2)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    
    conv5 = Conv2D(kernel_size=(3,3),filters=512,padding='same',kernel_initializer='he_normal',kernel_regularizer=l2(0.0001))(conv4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)
    
    conv6 = Conv2D(kernel_size=(3,3),filters=512,padding='same',kernel_initializer='he_normal',kernel_regularizer=l2(0.0001))(conv5)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)  
                 
    up3 = UpSampling2D(size=(2,2))(conv6)
    
    conv7 = Conv2D(kernel_size=(3,3),filters=256,padding='same'
                   ,kernel_initializer='he_normal',kernel_regularizer=l2(0.0001))(up3)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation('relu')(conv7)
    
    conv8 = Conv2D(kernel_size=(3,3),filters=256,padding='same',kernel_initializer='he_normal')(conv7)
    conv8 = Dropout(0.5)(conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation('relu')(conv8)    
        
    conv9 = Conv2D(kernel_size=(3,3),filters=256,padding='same',kernel_initializer='he_normal')(conv8)
    conv9 = BatchNormalization()(conv9)
    conv9 = Activation('relu')(conv9)    
               
    up4 = UpSampling2D(size=(2,2))(conv9)
    
    conv10 = Conv2D(kernel_size=(3,3),filters=128,padding='same',kernel_initializer='he_normal')(up4)
    conv10 = BatchNormalization()(conv10)
    conv10 = Activation('relu')(conv10)    
         
    conv11 = Conv2D(kernel_size=(3,3),filters=128,padding='same',kernel_initializer='he_normal')(conv10)
    conv11 = BatchNormalization()(conv11)
    conv11 = Activation('relu')(conv11)   
                            
    up5 = UpSampling2D(size=(2,2))(conv11)  
                
    conv12 = Conv2D(kernel_size=(3,3),filters=64,padding='same',kernel_initializer='he_normal')(up5)
    conv12 = BatchNormalization()(conv12)
    conv12 = Activation('relu')(conv12)  
          
    conv13 = Conv2D(kernel_size=(3,3),filters=64,padding='same',kernel_initializer='he_normal')(conv12)
    conv13 = BatchNormalization()(conv13)
    conv13 = Activation('relu')(conv13)     
              
    output =  Conv2D(kernel_size=(1,1),filters=classes,padding="valid")(conv13)
    output =  Reshape((img_size[0]*img_size[1], classes))(output)
    output =  Activation("softmax")(output)              
    model  =  Model(inputs,output)
    model.compile(optimizer=optimizer,loss=loss_function,metrics=metric) 
    return model    