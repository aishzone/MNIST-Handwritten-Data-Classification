### “When you’re fundraising, it’s AI. When you’re hiring, it’s ML. When you’re implementing, it’s logistic regression.”
## Introduction
Handwritten digit recognition is the ability of computers to recognize human handwritten digits.It is a tough task for machine because handwritten digits are not perfect.Today we will use the image of a digit and recognizes the digit present in the image.The MNIST data set contains 70000 images of handwritten digits (60,000 train_data  and 10,000 test_data).Each image has a resolution of 28x28, and the numerical pixel values are in greyscale. 

![img](https://miro.medium.com/max/700/1*1TkgO9Zz6rC3KpAYNl5KfA.png)

This is perfect for anyone who wants to get started with image classification using Scikit-Learn library. This is because, the set is neither too big to make beginners overwhelmed, nor too small so as to discard it altogether.It is a good database for people who want to try learning techniques and pattern recognition methods on real-world data while spending minimal efforts on preprocessing and formatting.

As you will be using the Scikit-Learn library, it is best to use its helper functions to download the data set.In here, we use __Logistic Regression__ to predict digit labels based on images. 

## CODING
### Fetching Dataset
Initially import some Standard Libraries

``` 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
```
Now we go ahead and fetch our MNIST dataset.Although there are many ways to do it we are goint to use **fetch_openml** function.

```
from sklearn.datasets import fetch_openml
mnist=fetch_openml('mnist_784')
mnist
```
The best part about downloading the data directly from Scikit-Learn is that it comes associated with a set of keys.
So basically, we get the `data` and `target` already separated. That makes our job much easier.

```
print(mnist.keys())
x, y = mnist['data'], mnist['target']
```
The data key contains 70000 rows (labels) and 784 columns (images). These columns all contain the pixel intensities of the handwritten numbers ranging from 0 to 255 which are of 28 x 28 (784) images.The target key contains all the labels from 0 to 9 corresponding to the data key pixels.

```
x.shape
y.shape
```

### Display
let us take a look at any one of the digits that are present in the data set. For this, you will be using the popular matplotlib library.

```
digit=x[36001]
digit_image=digit.reshape(28,28)
plt.imshow(digit_image,cmap=matplotlib.cm.binary,interpolation="nearest")
plt.axis("off")
y[36001]
```
First I have reshaped the images from 1-D arrays to __28 x 28__ matrices. Then, you will observe that I have used `plt.imshow()`. Actually, that takes an array image data and plots the pixels on the screen. (The pixel densities in this case).
Also the target label of `y[36001]` is __2__ as well, but with one caveat. The target label is a __string__. It is better to convert the labels to __integers__ as it will help further on in this guide.
### Separating the Training and Testing Set
Our next goal is to make a separate test set which the model will not see until the test phase is reached in the process.

```
x_train=x[:60000]
x_test=x[60000:]
y_train=y[:60000]
y_test=y[60000:]
```
The training set has 60,000 images and size of the test set is 10,000.
```
shuffle_index=np.random.permutation(60000)
x_train=x_train[shuffle_index]
y_train=y_train[shuffle_index]
```
It is possible that all the 1's are grouped together in the first 10,000 of the set and all 2's in the next 10,000 and so on. So it is necessary that we perform a __Shuffling__ or else our model may not be trained to recognize 8's and 9's.
```
y_train=y_train.astype(np.int8)
y_test=y_test.astype(np.int8)
```
We convert the type of our target data from `string` to `integer`.You can check your before and after by running `y_train`.
### Binary Classifier to detect all 2's
We make an attempt to create a Binary Classification (True/False) model that can detect whether the give input image is a 2 or not.
```
y_train_2=(y_train==2)
y_train_2
y_test_2=(y_test==2)
y_test_2
```
Now, we add a classifier to do our job.

__Step 1__ : Import the model you want to use

__Step 2__ : Make an instance of the Model

__Step 3__ : Training the model on the data, storing the information learned from the data

__Step 4__ : Predict the labels of new data (new images)
```
from sklearn.linear_model import LogisticRegression #Step 1
clf=LogisticRegression(tol=0.1,solver='lbfgs',max_iter=70000) #Step 2
```
We change from default solver to `lbfgs` to make our process faster.Also change max_itr from 100 (default) to 70,000.

```
clf.fit(x_train,y_train_2) #Step 3
```
So, we have replaced all labels from 0 till 9 with true and false.

```
clf.predict([digit]) #Step 4
```
Our Binary Classifier (clf) will return __true__ if 2 is present or else returns __false__. 

```
score = clf.score(x_test, y_test_2) #Model performance - Method 1
score
```
This is a simple way of checking the accuracy or model performance of our model.The other method is __Cross-Validation__ which comparitively takes longer time.
```
from sklearn.model_selection import cross_val_score
a=cross_val_score(clf,x_train,y_train_2,cv=5,scoring="accuracy",n_jobs=-1) #Model performance - Method 2
a.mean()
```
We now have a binary classifier with an accuracy of 97%.

### Step Modeling Pattern(MNIST) - Training,Fitting and Predicting
Training is possible with the help of __data__ and __target__.For sufficiently large datasets, it is best to implement SGD Classifier instead of Logistic Classifier to produce similar results in much less time.Moving to classify using the Logistic Regression you have to set loss to log.
```
from sklearn.linear_model import SGDClassifier #Step 1
clf = SGDClassifier(loss='log', random_state=42) #Step 2
clf.fit(x_train, y_train) #Step 3
clf.predict(x_test[0:10]) #Step 4
```
Now we go ahead and check our model performance

```
acc = clf.score(x_test,y_test)
acc
```
__Accuracy__ : 88%

You can go ahead and try to create a __Confusion matrix__ (Method 3 - To evaluate the accuracy of a classification) and __Seaborn__ (To make visualization a central part of exploring and understanding data)....etc... Also you can __display the Misclassified Images with predicted Labels__.

If you need help with the above mentioned then refer my `mnist_code.ipynb` attached in this repository.


## Thank You!!

