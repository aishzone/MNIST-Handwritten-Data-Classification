Handwritten digit recognition is the ability of computers to recognize human handwritten digits.It is a tough task for machine because handwritten digits are not perfect.Today we will use the image of a digit and recognizes the digit present in the image.
## CODING
### Fetching Dataset
Initially import some Standard Libraries

``` 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
```
Now we go ahead and fetch our MNIST dataset.Although there are many ways to do it we are goint to use **fetch_openml** function.The best part about downloading the data directly from Scikit-Learn is that it comes associated with a set of keys.

```
from sklearn.datasets import fetch_openml
mnist=fetch_openml('mnist_784')
mnist
```
So basically, we get the `data` and `target` already separated. That makes the job much easier.

```
mnist.keys()
x, y = mnist['data'], mnist['target']
```
The data key contains 70000 rows and 784 columns. These columns all contain the pixel intensities of the handwritten numbers ranging from 0 to 255 which are of 28 x 28 (784) images.The target key contains all the labels from 0 to 9 corresponding to the data key pixels.

```
x.shape
y.shape
```

### Display
let us take a look at the first few digits that are in the data set. For this, you will be using the popular matplotlib library.

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
### Training Data
Training is possible with the help of __data__ and __target__.For sufficiently large datasets, it is best to implement SGD Classifier instead of Logistic Classifier to produce similar results in much less time.Moving to classify using the Logistic Regression you have to set loss to log.
