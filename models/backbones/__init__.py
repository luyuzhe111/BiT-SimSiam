from .cifar_resnet_1 import resnet18 as resnet18_cifar_variant1
from .cifar_resnet_1 import resnet50 as resnet50_cifar_variant1
from .cifar_resnet_2 import ResNet18 as resnet18_cifar_variant2
from .cifar_resnet_v2 import resnet50_s as simsiam_cifar_resnet50_gn
from .cifar_resnet_v2_bn import resnet50_s as simsiam_cifar_resnet50_bn

from .resnet import resnet50 as simsiam_resnet50_common_scratch
from .simsiam_resnet_v2 import resnet50_s as simsiam_resnet50_small_scratch
from .simsiam_resnet_v2 import resnet50_m as simsiam_resnet50_medium_scratch
from .resnet_v2 import resnet50_s as bit_s_resnet50
from .resnet_v2 import resnet50_m as bit_m_resnet50



