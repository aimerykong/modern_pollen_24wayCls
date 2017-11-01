# Joint Pollen Grain Detection and Classification with Multiplicative Gating Architecture 

For papers, slides and posters, please refer to our [project page](http://http://www.ics.uci.edu/~skong2/pollen.html "modernPollen-detCls")

![alt text](https://drive.google.com/file/d/0B6uW-Khc9uCDVXNCMl96ZGJKTGM/view "visualization")

This is the first system for 24-way pollen grain classification. 
Please download the models and dataset from the [google drive](https://drive.google.com/drive/folders/0B6uW-Khc9uCDQ01SRlVuejlTemM?usp=sharing).
Note that download the folders and put them in the current directory so that the script can find the models.

While every pieces are contained here, please simply run "part10_checkModel2confusiomMatrix_test.m" to evaluate the model. 
Moreover, please run "part11_calibrationByLinearRegression.m" to see how the calibration improves the result (this does not require Caffe toolbox).


The dataset is here in personal local machine -- 

~/LargeScalePollenProject/dataset

PS: 01253 does not exist!


caffe is used in our project, and some functions are changed/added. Please compile accordingly

```python
make all -j

```


last update: 11/01/2017

Shu Kong

aimerykong At g-m-a-i-l dot com




