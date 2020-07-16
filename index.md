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
Now we go ahead and fetch our MNIST dataset.Although there are many ways to do it we are goint to use **fetch_openml** function

```
from sklearn.datasets import fetch_openml
