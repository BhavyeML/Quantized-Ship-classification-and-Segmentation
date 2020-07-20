# %% [code]
!pip install -r '../input/req-ship/requirements_infer.txt'

# %% [code]
class prediction_satellite(object):
    
    def __init__(self,image_link,model_file):
        
        try:
            self.image_link=image_link
        except:
            raise IOERROR("File does not exist")
            
        self.interpreter=tf.lite.Interpreter(model_path=model_file)
        
        
    def image_reader(self):
        
        self.image=cv.imread(self.image_link)
        self.image=cv.cvtColor(self.image,cv.COLOR_BGR2RGB)
        self.height,self.width=self.image.shape[0],self.image.shape[1]
        self.channels=self.image.shape[2]
        
        print("Preview of the image")
        self.image_viewer()
       
    def image_viewer(self):
        
        plt.figure(1, figsize = (15, 30))
        plt.subplot(3, 1, 1)
        plt.imshow(self.image)
        plt.show()
    
    def shore_removal(self,result_image,x,y,r_mean,g_mean,b_mean):

        channel_r,channel_g,channel_b= cv.split(result_image)



        area=np.zeros(3)
        for i in range(80):
            for j in range(80):
                area[0]+=channel_r[y+i][x+j]
                area[1]+=channel_g[y+i][x+j]
                area[2]+=channel_b[y+i][x+j]

        area=(area/(80*80))

        if area[0]>r_mean:
            channel_r=cv.rectangle(channel_r,(x+80,y+80),(x,y),(0,0,0),-1)


        if area[1]>g_mean:
            channel_g=cv.rectangle(channel_g,(x+80,y+80),(x,y),(0,0,0),-1)


        if area[2]>b_mean:
            channel_b=cv.rectangle(channel_b,(x+80,y+80),(x,y),(0,0,0),-1)

        result_image=cv.merge(( channel_r,channel_g,channel_b))
        return result_image
    
    def water_removal(self,result_image,img_mean):
    
        img=result_image.reshape(self.height*self.width*self.channels)

        for i in range(img.shape[0]):
                if img[i]<img_mean:
                    img[i]=0

        img=img.reshape(self.height,self.width,self.channels)

        return img
    
    def bbox(self,result_image,x,y):
        """ Produces less no. of boxes in segmentation but more accuracte, takes ~30 secs more"""
        flag=False
        for i in range(80):
            for j in range(80):
                if result_image[y+i][x+j][0]!=0:
                    result_image=cv.rectangle(result_image,(x+80,y+80),(x,y),(0,255,0),5)
                    flag=True
        return result_image,flag
        
    def segmentation_kmeans(self):
        
        
        K = 2
        attempts=10
        step=80
        ship_ct=0
        self.coordinates=[]
        vectorized = self.image.reshape((-1,3))
        vectorized = np.float32(vectorized)
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)

        ret,label,center=cv.kmeans(vectorized,K,None,criteria,attempts,cv.KMEANS_RANDOM_CENTERS)

        center = np.uint8(center)
        res = center[label.flatten()]
        result_image = res.reshape((self.image.shape))
        img_mean=int(result_image.mean())


        channel_r,channel_g,channel_b= cv.split(result_image)
        r_mean=channel_r.mean()
        g_mean=channel_g.mean()
        b_mean=channel_b.mean()

        for y in range(int((self.height-(80-step))/step)):
            for x in range(int((self.width-(80-step))/step)):
                result_image=self.shore_removal(result_image, x*step,y*step,r_mean,g_mean,b_mean)
        result_image=self.water_removal(result_image,img_mean)

        
        for y in range(0,self.height,10):
            for x in range(0,self.width,10):

                if result_image[y][x][0]!=0:
                    result_image=cv.rectangle(result_image,(x+40,y+40),(x-40,y-40),(0,0,255),5)
                    pts=(x-40,y-40)
                    self.coordinates.append(pts)
                    ship_ct+=1

        #for y in range(int((height-(80-step))/step)):
            #for x in range(int((width-(80-step))/step)):
                #result_image,flag=bbox(result_image,x*step,y*step)
                #if flag is True:

                    #c+=1

        print("No of segmentation masks",ship_ct)
        
        plt.imshow(result_image)
    
    def coordinates_correction(self):
        self.processed_coordinates=[]
        for i in range(len(self.coordinates)):
            c=list(self.coordinates[i])
            if c[0]<0:
                c[0]=0
            if c[1]<0:
                c[1]=0
            self.processed_coordinates.append(c)
                
        
    def interpreter_quant(self):
        
        self.interpreter.allocate_tensors()
        self.input_tensor_id=self.interpreter.get_input_details()[0]['index']
        self.output_tensor=self.interpreter.tensor(self.interpreter.get_output_details()[0]['index'])
        self.interpreter.get_tensor_details()
    

    
    def ship_detector(self):
        
        self.image_reader()
        start=time.time()
        step = 10

        self.interpreter_quant()
        self.segmentation_kmeans()
        self.coordinates_correction()
        
        for x,y in self.processed_coordinates:
            area1= self.ship_checker(x, y)
            area=tf.expand_dims(area1,axis=0)
            area=tf.cast(area,dtype=tf.float32)
            self.interpreter.set_tensor(self.input_tensor_id,area)
            self.interpreter.invoke()
            result=self.output_tensor()[0][1]
                
            if result > 0.90: #and self.not_near(x*step,y*step, 88, coordinates):
                #coordinates.append([[x*step, y*step], result])
                print(result)
                plt.imshow(area[0,:,:,:])
                plt.show()
                self.image=cv.rectangle(self.image,(x+80,y+80),(x,y),(0,0,255),4)
        
        print("time taken",time.time()-start)
        self.image_viewer()
        
    def ship_checker(self,x,y):
        
        channel_r,channel_g,channel_b= cv.split(self.image)
        
        area=np.ones(3*80*80).reshape(3,80,80)
        
       
        if x+80<self.width and y+80>self.height:
            for i in range(self.height-y):
                for j in range(80):
                    area[0][i][j]=channel_r[y+i][x+j]
                    area[1][i][j]=channel_g[y+i][x+j]
                    area[2][i][j]=channel_b[y+i][x+j]
        elif x+80>self.width and y+80 >self.height:
            for i in range(self.height-y):
                for j in range(self.width-x):
                    area[0][i][j]=channel_r[y+i][x+j]
                    area[1][i][j]=channel_g[y+i][x+j]
                    area[2][i][j]=channel_b[y+i][x+j]
        elif x+80>self.width and y+80 <self.height:
            for i in range(80):
                for j in range(self.width-x):
                    area[0][i][j]=channel_r[y+i][x+j]
                    area[1][i][j]=channel_g[y+i][x+j]
                    area[2][i][j]=channel_b[y+i][x+j]
        else:
            for i in range(80):
                for j in range(80):
                    area[0][i][j]=channel_r[y+i][x+j]
                    area[1][i][j]=channel_g[y+i][x+j]
                    area[2][i][j]=channel_b[y+i][x+j]
                
        area=area.transpose(2,1,0)
        area=area/255.0
        return area
    
    def not_near(self,x, y, s, coordinates):
        
        result_val = True
        for e in coordinates:
            if x+s > e[0] and x-s < e[0] and y+s > e[1] and y-s < e[1]:
                result_val = False
        return result_val
    
    
    
    
    
                

# %% [code]
if __name__=="__main__":
    pred=prediction_satellite('/kaggle/input/ships-in-satellite-imagery/scenes/scenes/sfbay_1.png','/kaggle/input/model-ship/model (2).tflite')
    pred.ship_detector()