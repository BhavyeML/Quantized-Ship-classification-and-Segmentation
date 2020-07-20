# Quantized-Ship-classification-and-Segmentation

This repository contains a segmentation and classification model for classifying ships in San Franciso Bay using Planet satellite imagery.

# Dataset

    Dataset contials 4000 image of size 80x80 with ground coverage of 3 metre per pixel. Dataset contains 1000 images of ship while remianing 3000 images with no-ship labe; are
    equally distributed within partial-ships, random sampling and images previously mis-labelled by ML Models

    Dataset also contains satellite imagery scenes covering a few os tens of sq. kms (~45 square km). These are multi-million pixels images with a ground resolution of 3m per
    pixel.
  
# Workkflow

    To beuild our solution, I have deployed a two step architecture:
    a. Image segmentation using K-means
    b. Image Classification using Quantization-aware model



    This two step architecture has been adapted because the satelltite scenes are too large and thus working on a single classification model was giving poor results of
    satisfactory metrics of Time-Bound prediction.Thus, using segmentation we are able to extract only a few coordinates where we need to use our classification model to detect 
    ships.
  
# Selective Segmentation Model

    Here, we have used  k-means clustering algorithm from OpenCV to segment our image into 2 cluster based on the 3-d pixel value as [R,G,B] for each pixels. Each pixel either
    gets categorized as "water-like" or "non-water-like"

    Basically, The k-means algorithm proceeds partitioning and distributing all those pixels in k different clusters, in our case based only on color proximity. This is a 
    difficult computational problem, but usually, and specifically here, the numerical algorithms converge rapidly into an optimal solution. Such optimization algorithms work 
    iteratively by assigning each pixel to the cluster whose mean vector is the nearest to the pixel (in the RGB space; the first assignment may be arbitrary), to then re-
    compute the new mean of each resulting cluster after the new pixel assignments, and iteratively repeat these two same actions again and again until the distribution of 
    pixels in clusters is stable. 

    This is a very fast an efficient way to remove pieces of water and land that need not be further processed by the deep learning algorithm coming next, enhancing efficiency 
    and speed. If we had proceeded without this segmentation stage, and instead gone for a brute force approach, fixed boxes would need to parse across the whole scene in the 
    next stage. Using 80 x 80 pixels boxes, as we describe below, that would have meant around 50 million windows to be analyzed by the coming algorithm, which would have led 
    to a prohibitive solution in terms of computing speed. Conversely, with the simple selective segmentation we have performed and described here, only a few hundred windows 
    need to be further analyzed.

# Classification Model

    For classification, we have performed quantization-aware training along with TfLite Framework, so as to to be able to deploy model on the edge or use it for the purpose of 
    streaming analytics. The input is given as cooridnates of a box os sie 80x80 from selective segmentation model. Such windows are suited for detecting ships of up to ~250–
    300 meters (like the largest cargo ships on the scene).

    Convolutional layers and Max-Pooling layers have been used to perform training. Dropout has been applied to regularize our model. The model has been trained for 50 epochs 
    with a callback function and Stochastic Gradient Optimizer with Categorical cross entropy loss. 

    The size of the model is very low ~ 400 kb and the acuuracy achievd is over 99%.The model weigts are also present in the repository

  # Metrics
  
    a. Optimal Metrics: High ship-detection accuracy ~ 99%
    b. Satisfactory Metrics: Speed of Detection~ 40 secs 
   
   
  
  
  
  
  
  
  
  
  

