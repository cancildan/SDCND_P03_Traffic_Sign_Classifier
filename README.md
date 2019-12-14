## Self-Driving Car Engineer Nanodegree Program

## Project: Build a Traffic Sign Recognition Classifier

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # "Image References"

[image1]: ./examples/visualization01.png "Visualization1"
[image2]: ./examples/visualization02.png "Visualization2"
[image3]: ./examples/preprocessed01.png "Grayscaling01"
[image3_2]: ./examples/preprocessed02.png "Grayscaling02"
[image4]: ./examples/accuracy.png "Accuracy"
[image5]: ./examples/test_images.png "Test Images"
[image6]: ./test_german_traffic_signs/caution.jpg "Traffic Sign 1"
[image7]: ./test_german_traffic_signs/slippery.jpg "Traffic Sign 2"
[image8]: ./test_german_traffic_signs/30km.jpg "Traffic Sign 3"
[image9]: ./test_german_traffic_signs/keep_right.jpg "Traffic Sign 4"
[image10]: ./test_german_traffic_signs/no_entry.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

The project implementation can be found [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

##### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

You can see some randomly selected images below:

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

The graph shows that distribution of traffic signs are similar between training, validation and testing set. And 5 most or least common traffic signs in training data are:

![alt text][image2]

With these data distribution, I can also say that there's a strong imbalance among the classes. However, I see that the data distribution is almost the same between training and testing set. I'm not sure training will end with high accuracy with these datasets but, in the starting point I decided to go with given data for training, validation and test. But If I don't satisfied with my final solution I would think to enlarge my dataset by using some augmentation by aiming to create balance between classes.

You can see most and least common labels for the training label.

```
Most common five:
  5.78%, 2010 images --- Speed limit (50km/h)
  5.69%, 1980 images --- Speed limit (30km/h)
  5.52%, 1920 images --- Yield
  5.43%, 1890 images --- Priority road
  5.34%, 1860 images --- Keep right

Least common five:
  0.60%, 210 images --- End of all speed and passing limits
  0.60%, 210 images --- Pedestrians
  0.52%, 180 images --- Go straight or left
  0.52%, 180 images --- Dangerous curve to the left
  0.52%, 180 images --- Speed limit (20km/h)
```

### Design and Test a Model Architecture

##### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because neural networks creates better solution    if the input have mean zero, therefore I tried different preprocessing methods but it end with grayscalling. Then I applied normalization as a last step, because its always better to have image data with zero mean and equal variance for optimizer to find solution.

There are a lot of different preprocessing I could do to improve both the image qualities and neural network performance, but I decided to go just for gray scaling the images.

Here is an example of a traffic sign image after grayscaling.

![alt text][image3]

You can also see randomly selected examples of a traffic sign images after grayscaling.

![alt text][image3_2]


##### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| **Layer**               | **Description**                              |
| ----------------------- | -------------------------------------------- |
| Input                   | 32x32x1 Grayscale image                      |
| Convolutional Layer 1   | 1x1 strides, valid padding, outputs 28x28x16 |
| RELU                    |                                              |
| Max Pool                | 2x2, outputs 14x14x16                        |
| Convolutional Layer 2   | 1x1 strides, valid padding, outputs 10x10x64 |
| RELU                    |                                              |
| Max Pool                | 2x2, outputs 5x5x64                          |
| Fatten                  | To connect to fully-connected layers         |
| Fully-connected Layer 1 | Outputs 1600                                 |
| RELU                    |                                              |
| Dropout                 | 0.7 keep probability                         |
| Fully-connected Layer 2 | Outputs 240                                  |
| RELU                    |                                              |
| Dropout                 | 0.7 keep probability                         |
| Fully-connected Layer 3 | Outputs 43                                   |



##### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I started with 15 epochs, and then increased it to 50. For the batch size I started with 128, and then decreased it to 64. Learning rate 0.001 and use [Adam](http://sebastianruder.com/optimizing-gradient-descent/index.html#adam) optimizer not needing to change the learning rate. Here is my network accuracy by epoch:

![alt text][image4]

##### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 1
* validation set accuracy of 0.975 
* test set accuracy of 0.958

The starting model was LeNet provided by Udacity while it was proved to work well in the recognition hand and print written character. After modifying the standard model work with color pictures, I could not have more than 85% accuracy with my current dataset and 15 epochs. Then I adjusted first two convolution layer deeper, and then increase the size of the fully-connected layers. With these modifications, I got just above 90% accuracy. To go further, I added two dropout layers with 0.7 keep probability and increased the training epochs to 50. And as a final adjustment I've changed the first input layer 32x32x3 RGB to 32x32x1 Grayscale image.


### Test a Model on New Images

##### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image5]

In the way to progression of neural network optimization, I sometimes failed to predict "slippery" sign but at the end of my progress all the traffic signs had been predicted correctly.

##### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| General caution | General caution |
|    Slippery road     | Slippery road |
| Speed limit (30km/h)	| Speed limit (30km/h)	|
| Keep right	| Keep right	|
| No entry	| No entry     |

The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%.

##### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

You can see all 5 images prediction details with softmax:

```
Test Image: ./test_german_traffic_signs/caution.jpg
Top 5 Probabilities:
   1.000000 : 18 - General caution
   0.000000 : 0 - Speed limit (20km/h)
   0.000000 : 1 - Speed limit (30km/h)
   0.000000 : 2 - Speed limit (50km/h)
   0.000000 : 3 - Speed limit (60km/h)

Test Image: ./test_german_traffic_signs/slippery.jpg
Top 5 Probabilities:
   1.000000 : 23 - Slippery road
   0.000000 : 0 - Speed limit (20km/h)
   0.000000 : 1 - Speed limit (30km/h)
   0.000000 : 2 - Speed limit (50km/h)
   0.000000 : 3 - Speed limit (60km/h)

Test Image: ./test_german_traffic_signs/30km.jpg
Top 5 Probabilities:
   1.000000 : 1 - Speed limit (30km/h)
   0.000000 : 0 - Speed limit (20km/h)
   0.000000 : 2 - Speed limit (50km/h)
   0.000000 : 3 - Speed limit (60km/h)
   0.000000 : 4 - Speed limit (70km/h)

Test Image: ./test_german_traffic_signs/keep_right.jpg
Top 5 Probabilities:
   1.000000 : 38 - Keep right
   0.000000 : 0 - Speed limit (20km/h)
   0.000000 : 1 - Speed limit (30km/h)
   0.000000 : 2 - Speed limit (50km/h)
   0.000000 : 3 - Speed limit (60km/h)

Test Image: ./test_german_traffic_signs/no_entry.jpg
Top 5 Probabilities:
   1.000000 : 17 - No entry
   0.000000 : 0 - Speed limit (20km/h)
   0.000000 : 1 - Speed limit (30km/h)
   0.000000 : 2 - Speed limit (50km/h)
   0.000000 : 3 - Speed limit (60km/h)
```


