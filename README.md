# Joint Pollen Grain Detection and Classification with Multiplicative Gating Architecture 

For papers, slides and posters, please refer to our [project page](http://http://www.ics.uci.edu/~skong2/pollen.html "modernPollen-detCls"). Please visit also visit the [webpage page of Pollen Morphology-based Classification and Analysis](http://www.ics.uci.edu/~fowlkes/bioshape/index.html) for more relevant projects.

![](https://github.com/aimerykong/modern_pollen_24wayCls/raw/master/figures/114_vir_10162_conf7_wid154.jpg)

![](https://github.com/aimerykong/modern_pollen_24wayCls/raw/master/figures/pixelSubtraction_big00448_y1x1.png)

![alt text](https://raw.githubusercontent.com/aimerykong/modern_pollen_24wayCls/master/figures/confusionMatrix24WayTesting_afterCalibration.jpg)


This is the first system for 24-way pollen grain classification. 
Please download the models and dataset from the [google drive](https://drive.google.com/drive/folders/0B6uW-Khc9uCDQ01SRlVuejlTemM?usp=sharing).
Note that download the folders and put them in the current directory so that the script can find the models.

Every pieces of the system are contained here.

1. Please simply run [part10_checkModel2confusiomMatrix_test.m](https://github.com/aimerykong/modern_pollen_24wayCls/blob/master/part10_checkModel2confusiomMatrix_test.m) to evaluate the model. 

2. Please run [part11_calibrationByLinearRegression.m](https://github.com/aimerykong/modern_pollen_24wayCls/blob/master/part11_calibrationByLinearRegression.m) to see how the calibration improves the result (this does not require Caffe toolbox).


The dataset is here in personal local machine (capricorn) -- 

~/LargeScalePollenProject/dataset

PS: 01253 does not exist!


caffe is used in our project, and some functions are changed/added. Please compile accordingly

```python
make all -j

```


last update: 11/01/2017

Shu Kong

aimerykong At g-m-a-i-l dot com




