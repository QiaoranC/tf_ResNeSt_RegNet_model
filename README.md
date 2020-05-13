# tf_ResNeSt_RegNet_model
 tensorflow in **ResNeSt** and **RegNet**, model only, no pertrain model for download.

Easily to read? and change?
welcome for using it, ask question, find some bugs maybe.

usage is simple: 
```
from models.model_factory import get_model

input_shape = [224,244,3]
n_classes=81
fc_activation='softmax' #softmax sigmoid

model = get_model(model_name="ResNest50",input_shape=input_shape,n_classes=n_classes,
                verbose=True,fc_activation=fc_activation)
```

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

```

for RegNet, cause there are various version, you can easily set it by `stage_depth`,`stage_width`,`stage_G`.
details seting from follwing pic from orginal paper https://arxiv.org/abs/2003.13678 .

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

![alt text](https://raw.githubusercontent.com/QiaoranC/tf_ResNeSt_RegNet_model/master/readme_img/regnet_setting.png)


I compared **ResNeSt50** and some **RegNet**(below 4.0GF) in my own project, also compared to **EfficientNet b0/b1/b2**.
it seems **EfficientNet** is still good at balance in size/speed and accuracy, and **ResNeSt50** performe well at accuarcy also lower in size/speed, And **RegNet** not that fast and acuracy not that good, seems normal.

