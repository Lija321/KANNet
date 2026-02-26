from typing import Callable

from .efficientnet_b0 import get_model as efficientnet_b0
from .efficientnet_b0 import get_preprocess as efficientnet_b0_preprocess
from .efficientnet_b1 import get_model as efficientnet_b1
from .efficientnet_b1 import get_preprocess as efficientnet_b1_preprocess
from .kan_resnet18 import get_model as kan_resnet18
from .kan_resnet50 import get_model as kan_resnet50
from .mobilenetv2 import get_model as mobilenetv2
from .mobilenetv2 import get_preprocess as mobilenetv2_preprocess
from .resnet18 import get_model as resnet18
from .resnet18 import get_preprocess as resnet18_preprocess
from .resnet50 import get_model as resnet50
from .resnet50 import get_preprocess as resnet50_preprocess
from .shufflenetv2 import get_model as shufflenetv2
from .shufflenetv2 import get_preprocess as shufflenetv2_preprocess
from .vgg16 import get_model as vgg16
from .vgg16 import get_preprocess as vgg16_preprocess
from .tiny_convnet import get_model as tiny_convnet
from .tiny_kannet import get_model as tiny_kannet

MODEL_REGISTRY: dict[str, Callable] = {
    "resnet18": resnet18,
    "resnet50": resnet50,
    "vgg16": vgg16,
    "mobilenetv2": mobilenetv2,
    "shufflenetv2": shufflenetv2,
    "efficientnet_b0": efficientnet_b0,
    "efficientnet_b1": efficientnet_b1,
    "kan_resnet18_tiny": lambda num_classes, pretrained=False, pretrained_path=None: kan_resnet18(
        num_classes=num_classes, pretrained=pretrained, variant="tiny", pretrained_path=pretrained_path
    ),
    "kan_resnet18_small": lambda num_classes, pretrained=False, pretrained_path=None: kan_resnet18(
        num_classes=num_classes, pretrained=pretrained, variant="small", pretrained_path=pretrained_path
    ),
    "kan_resnet18_base": lambda num_classes, pretrained=False, pretrained_path=None: kan_resnet18(
        num_classes=num_classes, pretrained=pretrained, variant="base", pretrained_path=pretrained_path
    ),
    "kan_resnet50_tiny": lambda num_classes, pretrained=False, pretrained_path=None: kan_resnet50(
        num_classes=num_classes, pretrained=pretrained, variant="tiny", pretrained_path=pretrained_path
    ),
    "kan_resnet50_small": lambda num_classes, pretrained=False, pretrained_path=None: kan_resnet50(
        num_classes=num_classes, pretrained=pretrained, variant="small", pretrained_path=pretrained_path
    ),
    "kan_resnet50_base": lambda num_classes, pretrained=False, pretrained_path=None: kan_resnet50(
        num_classes=num_classes, pretrained=pretrained, variant="base", pretrained_path=pretrained_path
    ),
    "tiny_convnet_w32": lambda num_classes, pretrained=False, pretrained_path=None: tiny_convnet(
        num_classes=num_classes,
        pretrained=pretrained,
        width=32,
    ),
    "tiny_kannet_w32": lambda num_classes, pretrained=False, pretrained_path=None: tiny_kannet(
        num_classes=num_classes,
        pretrained=pretrained,
        width=32,
        apply_kan_at_8x8=True,
    ),
    # Extra safe ultra-light version (VERY recommended for debugging)
    "tiny_kannet_w16": lambda num_classes, pretrained=False, pretrained_path=None: tiny_kannet(
        num_classes=num_classes,
        pretrained=pretrained,
        width=16,
        apply_kan_at_8x8=True,
    ),
}

PREPROCESS_REGISTRY: dict[str, Callable] = {
    "resnet18": resnet18_preprocess,
    "resnet50": resnet50_preprocess,
    "vgg16": vgg16_preprocess,
    "mobilenetv2": mobilenetv2_preprocess,
    "shufflenetv2": shufflenetv2_preprocess,
    "efficientnet_b0": efficientnet_b0_preprocess,
    "efficientnet_b1": efficientnet_b1_preprocess,
    "kan_resnet18_tiny": resnet18_preprocess,
    "kan_resnet18_small": resnet18_preprocess,
    "kan_resnet18_base": resnet18_preprocess,
    "kan_resnet50_tiny": resnet50_preprocess,
    "kan_resnet50_small": resnet50_preprocess,
    "kan_resnet50_base": resnet50_preprocess,
    "tiny_convnet_w32": resnet18_preprocess,
    "tiny_kannet_w32": resnet18_preprocess,
    "tiny_kannet_w16": resnet18_preprocess,
}
