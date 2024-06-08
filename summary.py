#--------------------------------------------#
#   该部分代码用于看网络结构
#--------------------------------------------#
import torch
from thop import clever_format, profile
from torchsummary import summary
from torchstat import stat
# from ptflops import get_model_complexity_info
from nets.unet import Unet
from nets.bisenetv2 import BiSeNetV2
from nets.build_BiSeNet import BiSeNet
if __name__ == "__main__":
    # input_shape     = [512, 512]
    input_shape = [128, 128]
    num_classes     = 2
    backbone        = 'vgg'
    
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = Unet(num_classes = num_classes, backbone = 'vgg').to(device)
    # model= BiSeNetV2(n_classes=num_classes).to(device)
    model = BiSeNet(num_classes=num_classes,context_path='resnet18').to(device)
    summary(model, (3, input_shape[0], input_shape[1]))
    
    dummy_input     = torch.randn(1, 3, input_shape[0], input_shape[1]).to(device)
    flops, params   = profile(model.to(device), (dummy_input, ), verbose=False)
    #--------------------------------------------------------#
    #   flops * 2是因为profile没有将卷积作为两个operations
    #   有些论文将卷积算乘法、加法两个operations。此时乘2
    #   有些论文只考虑乘法的运算次数，忽略加法。此时不乘2
    #   本代码选择乘2，参考YOLOX。
    #--------------------------------------------------------#
    flops           = flops * 2
    flops, params   = clever_format([flops, params], "%.3f")
    print('Total GFLOPS: %s' % (flops))
    print('Total params: %s' % (params))

    # macs, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True,
    #                                          print_per_layer_stat=True, verbose=True)
    #
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    stat(model, (3, input_shape[0], input_shape[1]))








