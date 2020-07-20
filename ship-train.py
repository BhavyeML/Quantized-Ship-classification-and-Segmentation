# %% [code]
import tensorflow as tf
import ship_dataloader
import matplotlib.pyplot as plt
import ship_callback

# %% [code]
class model_net(object):
    
    def __init__(self,file,channels,size,dropout=0.25):
        super(model_net,self).__init__()
        
        
        from ship_dataloader import dataloader
        from ship_callback import myCallback
        
        self.dropout=dropout
        self.loader=dataloader(file, channels,size)
        self.input_data,self.output_data=self.loader.vec_to_img()
        
        self.callbacks=myCallback()
        
        
        
    def model_arch(self):
        
        self.model= tf.keras.models.Sequential([
                                tf.keras.layers.Conv2D(32, (3, 3), padding='same', input_shape=(80, 80, 3), activation='relu'),
                                tf.keras.layers.MaxPooling2D(2,2),
                                tf.keras.layers.Dropout(self.dropout),
                                tf.keras.layers.Conv2D(32, (3, 3), padding='same',  activation='relu'),
                                tf.keras.layers.MaxPooling2D(2,2),
                                tf.keras.layers.Dropout(self.dropout),
                                tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
                                tf.keras.layers.MaxPooling2D(2,2),
                                tf.keras.layers.Dropout(self.dropout),
                                tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
                                tf.keras.layers.MaxPooling2D(2,2),
                                tf.keras.layers.Dropout(self.dropout),
                                tf.keras.layers.Flatten(),
                                tf.keras.layers.Dense(512,activation='relu'),
                                tf.keras.layers.Dropout(2*self.dropout),
                                tf.keras.layers.Dense(2,activation='softmax')
                                
            
        ])
