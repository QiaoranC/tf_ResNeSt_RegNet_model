from .ResNest import ResNest
from .RegNet import RegNet

def get_model(model_name='ResNest50',input_shape=(224,224,3),n_classes=81,
                verbose=False,dropout_rate=0,fc_activation=None,**kwargs):
    '''get_model
    input_shape: (h,w,c)
    fc_activation: sigmoid,softmax
    '''
    if model_name == 'ResNest50':
        model = ResNest(verbose=verbose, input_shape=input_shape,
        n_classes=n_classes, dropout_rate=dropout_rate, fc_activation=fc_activation,
        blocks_set=[3, 4, 6, 3], radix=2, groups=1, bottleneck_width=64, deep_stem=True,
        stem_width=32, avg_down=True, avd=True, avd_first=False,**kwargs).build()
    
    elif model_name == 'ResNest101':
        model = ResNest(verbose=verbose, input_shape=input_shape,
        n_classes=n_classes, dropout_rate=dropout_rate, fc_activation=fc_activation,
        blocks_set=[3, 4, 23, 3], radix=2, groups=1, bottleneck_width=64, deep_stem=True,
        stem_width=64, avg_down=True, avd=True, avd_first=False,**kwargs).build()
    
    elif model_name == 'ResNest200':
        model = ResNest(verbose=verbose, input_shape=input_shape,
        n_classes=n_classes, dropout_rate=dropout_rate, fc_activation=fc_activation,
        blocks_set=[3, 24, 36, 3], radix=2, groups=1, bottleneck_width=64, deep_stem=True,
        stem_width=64, avg_down=True, avd=True, avd_first=False,**kwargs).build()
    
    elif model_name == 'ResNest269':
        model = ResNest(verbose=verbose, input_shape=input_shape,
        n_classes=n_classes, dropout_rate=dropout_rate, fc_activation=fc_activation,
        blocks_set=[3, 30, 48, 8], radix=2, groups=1, bottleneck_width=64, deep_stem=True,
        stem_width=64, avg_down=True, avd=True, avd_first=False,**kwargs).build()
    
    elif model_name == 'RegNetX400':
        model = RegNet(verbose=verbose, input_shape=input_shape,
        n_classes=n_classes, dropout_rate=dropout_rate, fc_activation=fc_activation,
        stage_depth=[1,2,7,12],stage_width=[32,64,160,384],stage_G=16,SEstyle_atten="noSE",**kwargs).build()
    
    elif model_name == 'RegNetX1.6':
        model = RegNet(verbose=verbose, input_shape=input_shape,
        n_classes=n_classes, dropout_rate=dropout_rate, fc_activation=fc_activation,
        stage_depth=[2,4,10,2],stage_width=[72,168,408,912],stage_G=24,SEstyle_atten="noSE",**kwargs).build()
    
    elif model_name == 'RegNetY400':
        model = RegNet(verbose=verbose, input_shape=input_shape,
        n_classes=n_classes, dropout_rate=dropout_rate, fc_activation=fc_activation,
        stage_depth=[1,3,6,6],stage_width=[48,104,208,440],stage_G=16,SEstyle_atten="SE",**kwargs).build()
   
    
    elif model_name == 'RegNetY1.6':
        model = RegNet(verbose=verbose, input_shape=input_shape,
        n_classes=n_classes, dropout_rate=dropout_rate, fc_activation=fc_activation,
        stage_depth=[2,6,17,2],stage_width=[48,120,336,888],stage_G=24,SEstyle_atten="SE",**kwargs).build()
    
    elif model_name == 'RegNet':
        model = RegNet(verbose=verbose, input_shape=input_shape,
        n_classes=n_classes, dropout_rate=dropout_rate, fc_activation=fc_activation, **kwargs).build()
    
    else:
        raise ValueError('Unrecognize model name {}'.format(model_name))
    return model

if __name__ == "__main__":

    # model_names = ['ResNest50','ResNest101','ResNest200','ResNest269']
    model_names = ['RegNetX400','RegNetX1.6','RegNetY400','RegNetY1.6']
    input_shape = [224,244,3]
    n_classes=81
    fc_activation='softmax' #softmax sigmoid

    # for model_name in model_names:
    #     print('model_name',model_name)
    #     model = get_model(model_name=model_name,input_shape=input_shape,n_classes=n_classes,
    #                 verbose=True,fc_activation=fc_activation)
    #     print('-'*10)

    #RegNetY600 set
    model = get_model(model_name="RegNet",input_shape=input_shape,n_classes=n_classes,
                verbose=True,fc_activation=fc_activation,stage_depth=[1,3,7,4],
                stage_width=[48,112,256,608],stage_G=16,SEstyle_atten="SE",active='mish')
    print('-'*10)