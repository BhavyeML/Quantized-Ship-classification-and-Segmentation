import json,random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


class dataloader(object):
    
    def __init__(self,dataset_link,channels,size):
        super(dataloader,self).__init__()
        
        
        try:
            self.file=open(dataset_link)
            self.dataset=json.load(self.file)
            self.file.close()
        except:
            raise IOERROR(f"{dataset_link} does not exist")
        if channels is None:
            self.channels=3
        else:
            self.channels=channels
        if size is None:
            self.img_size=80
        else:
            self.img_size=size
        self.input_data = np.array(self.dataset['data']).astype('uint8')
        self.output_data = np.array(self.dataset['labels']).astype('uint8')

    def dataset_analysis(self,initial=False):
       
        
        print("shape of the input data array is {}".format(self.input_data.shape))
        print("shape of the output label array is {}".format(self.output_data.shape))
        if initial is True:
            print("distribution of the dataset as no-ships(0) and ships(1) is {}".format(np.bincount(self.output_data)))
        
    def vec_to_img(self):
        print("Orginal shape of dataset")
        
        self.dataset_analysis(initial=True)
        
        print("Reshaping vectors for model training")
        self.input_data=self.input_data.reshape([-1,self.channels,self.img_size,self.img_size])
        self.dataset_analysis()
        
        print("Visualizing image sample")
        self.visualize_img_sample()
        
        # shuffling indexes
        indexes = np.arange(4000)
        np.random.shuffle(indexes)
        self.input_data = self.input_data[indexes].transpose([0,2,3,1])
        self.output_data = self.output_data[indexes]
        
        self.normalize_data()
        self.tf_cast()
        
        return self.input_data,tf.one_hot(self.output_data,depth=2)
        
    def visualize_img_sample(self):
        
        
        sample = self.input_data[3]

        red_spectrum = sample[0]
        green_spectrum = sample[1]
        blue_spectrum = sample[2]
        
        plt.figure(2, figsize = (5*3, 5*1))
        plt.set_cmap('jet')
        plt.subplot(1, 3, 1)
        plt.imshow(red_spectrum)
        plt.subplot(1, 3, 2)
        plt.imshow(green_spectrum)
        plt.subplot(1, 3, 3)
        plt.imshow(blue_spectrum)

        plt.show()
    
    def normalize_data(self):
        
        self.input_data=self.input_data/255
    
    def tf_cast(self):
        
        self.input_data=tf.cast(self.input_data,tf.float32)
   
        