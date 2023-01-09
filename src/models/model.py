import timm


def ResNeStModel():
    
    return timm.create_model('resnest50d', pretrained=False)
