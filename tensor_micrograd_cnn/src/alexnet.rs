use crate::layers::{Linear, Conv2d};
use crate::Tensor;

pub struct AlexNet{
    c2d_layer1: Conv2d,
    c2d_layer2: Conv2d,
    c2d_layer3: Conv2d,
    c2d_layer4: Conv2d,
    c2d_layer5: Conv2d,
    fc_layer6: Linear,
    fc_layer7: Linear,
    fc_layer8: Linear,
    num_classes: usize,
}

impl AlexNet{

    pub fn new(num_classes: usize) -> Self{
        AlexNet{ 
            c2d_layer1: Conv2d::new(3, 96, 11,4, 2, true),
            c2d_layer2: Conv2d::new(96, 256, 5,1, 2, true),
            c2d_layer3: Conv2d::new(256, 384, 3,1, 1, true),
            c2d_layer4: Conv2d::new(384, 384, 3,1, 1, true),
            c2d_layer5: Conv2d::new(384, 256, 3,1, 1, true),
            
            fc_layer6: Linear::new(6 * 6 * 256, 4096, true),
            fc_layer7: Linear::new(4096, 4096, true),
            fc_layer8: Linear::new(4096, num_classes, true),
            num_classes,
        }
    }

    pub fn forward(&self, input: &Tensor) -> Tensor{

        let mut x = self.c2d_layer1.forward(input).relu();
        x = x.maxpool_2d(3, 2);

        x = self.c2d_layer2.forward(&x).relu();
        x = x.maxpool_2d(3, 2);

        x = self.c2d_layer3.forward(&x).relu();
        x = self.c2d_layer4.forward(&x).relu();

        x = self.c2d_layer5.forward(&x).relu();
        x = x.maxpool_2d(3, 2);

        let batch = x.shape()[0];
        x = x.reshape(vec![batch, 6 * 6 * 256]);

        //FC ovde sadrzi i batcheve
        x = self.fc_layer6.forward(&x).relu();
        x = self.fc_layer7.forward(&x).relu();
        x = self.fc_layer8.forward(&x);

        x
    }

    pub fn parameters(&self) -> Vec<Tensor>{
        let mut params = Vec::new();
        params.extend(self.c2d_layer1.parameters());
        params.extend(self.c2d_layer2.parameters());
        params.extend(self.c2d_layer3.parameters());
        params.extend(self.c2d_layer4.parameters());
        params.extend(self.c2d_layer5.parameters());
        params.extend(self.fc_layer6.parameters());
        params.extend(self.fc_layer7.parameters());
        params.extend(self.fc_layer8.parameters());

        params
    }
}