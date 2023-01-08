import timm


def ResNeStModel():
    
    return timm.create_model('resnest14d', pretrained=False)
