# tf ResNeSt and RegNet

## Introduction
 Currently support tensorflow in 
 - **ResNeSt** 
 - **RegNet**
 
model only, no pertrain model for download. easy to read and modified.   
welcome for using it, ask question, test it, find some bugs maybe.

ResNeSt based on [offical github](https://github.com/zhanghang1989/ResNeSt) .

## Update
**2020-5-27**: ResNeSt add [CB-Net](https://arxiv.org/pdf/1909.03625.pdf) style to enahce backbone. theoretically, it should improve the results. Wait for test.


## Usage
usage is simple:
```
from models.model_factory import get_model

input_shape = [224,244,3]
n_classes = 81
fc_activation = 'softmax'

model = get_model(model_name="ResNest50",input_shape=input_shape,n_classes=n_classes,
                verbose=False,fc_activation=fc_activation)
model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy())
```


if you want use `Mish` as activation (default is `relu`): 
```
#it imporve the results, but come with high memory usage
model = get_model(model_name="ResNest50",input_shape=input_shape,n_classes=n_classes,
                verbose=False,fc_activation=fc_activation,active='mish')
```

if you add CB_Net in ResNeSt: add `using_cb=True` like:
```
model = get_model(model_name="ResNest50",...,using_cb=True)
```


## Models 
models now support:
```
ResNest50
ResNest101
ResNest200
ResNest269

RegNetX400
RegNetX1.6
RegNetY400
RegNetY1.6
AnyOther RegNetX/Y
```
#### RegNet
for RegNet, cause there are various version, you can easily set it by `stage_depth`,`stage_width`,`stage_G`.

```
#RegNetY600
model = get_model(model_name="RegNet",input_shape=input_shape,n_classes=n_classes,
                verbose=True,fc_activation=fc_activation,stage_depth=[1,3,7,4],
                stage_width=[48,112,256,608],stage_G=16,SEstyle_atten="SE")

#RegNetX600
model = get_model(model_name="RegNet",input_shape=input_shape,n_classes=n_classes,
                verbose=True,fc_activation=fc_activation,stage_depth=[1,3,5,7],
                stage_width=[48,96,240,528],stage_G=24,SEstyle_atten="noSE")
```

details seting (from orginal paper ):
- [facebookresearch/pycls](https://github.com/facebookresearch/pycls)

![alt text](https://raw.githubusercontent.com/QiaoranC/tf_ResNeSt_RegNet_model/master/readme_img/regnet_setting.png)


- CB-Net, using this style to enhace ResNest
![alt text](https://raw.githubusercontent.com/QiaoranC/tf_ResNeSt_RegNet_model/master/readme_img/CBNet.png)

## Discussion
I compared **ResNeSt50** and some **RegNet**(below 4.0GF) in my own project, also compared to **EfficientNet b0/b1/b2**.
it seems **EfficientNet** is still good at balance in size/speed and accuracy, and **ResNeSt50** performe well at accuarcy also lower in size/speed, And **RegNet** not that fast and acuracy not that good, seems normal.
