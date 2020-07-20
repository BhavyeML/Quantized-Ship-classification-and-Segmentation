# %% [code]
!pip install -r '../input/req-ship/rquirements.txt'

# %% [code]
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_model_optimization as tfmot
import ship_train
from ship_train import model_net

# %% [code]
class quantized_model_net(model_net):
    
    def __init__(self,file,channels,size,dropout=0.25):
        
        super().__init__(file,channels,size,dropout=0.25)
        self.quantize_model=tfmot.quantization.keras.quantize_model
    
    def model_arch(self):
        
        super().model_arch()
        self.q_aware_model=self.quantize_model(self.model)
        self.q_aware_model.summary()
    
     
    def trainer(self):
        
        from tensorflow.keras.optimizers import SGD
        sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)
        
        self.model_arch()
        
        self.q_aware_model.compile(optimizer=sgd,loss='categorical_crossentropy', metrics=['acc'])
    
        device_name = tf.test.gpu_device_name()
        if "GPU" not in device_name:
            print("GPU device not found")
            history = self.q_aware_model.fit(self.input_data,self.output_data,epochs=150,batch_size=32,validation_split=0.2,shuffle=True,verbose=2,callbacks=[self.callbacks])
        else:
            print('Found GPU at: {}'.format(device_name))
            config = tf.compat.v1.ConfigProto() 
            config.gpu_options.allow_growth = True
            with tf.device('/gpu:0'):
                history = self.q_aware_model.fit(self.input_data,self.output_data,epochs=150,batch_size=32,validation_split=0.2,shuffle=True,verbose=2,callbacks=[self.callbacks])
        
        self.visualize_train(history)
        self.save_model()
    
    def visualize_train(self,history):
        
        acc=history.history['acc']
        val_acc=history.history['val_acc']
        epochs=range(len(acc))
        plt.plot(epochs,acc)
        plt.plot(epochs,val_acc)
        plt.show()
        
    def save_model(self):
        
         self.model.save('final_model_train1.hdf5')

# %% [code]
if __name__=="__main__":
    tester= quantized_model_net("/kaggle/input/ships-in-satellite-imagery/shipsnet.json",3,80,0.25)
    tester.trainer()