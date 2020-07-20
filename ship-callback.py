
import tensorflow as tf

class myCallback(tf.keras.callbacks.Callback):
    
    """
        Function: Used to Stop Training When Desired Accuracy is Achieved
        
    """
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('val_acc')>0.99):
            print("\nReached 99% accuracy so cancelling training!")
            self.model.stop_training = True