use crate::layers::{Linear, Conv2d};
use crate::Tensor;

pub struct AlexNet{
    layer1: Conv2d,
    layer2: Conv2d,
    layer3: Conv2d,
    layer4: Conv2d,
    layer5: Conv2d,
    
}