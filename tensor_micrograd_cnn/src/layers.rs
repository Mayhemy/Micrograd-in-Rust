use crate::Tensor;

pub struct Linear{
    pub weights: Tensor,
    pub biases: Option<Tensor>,
}

impl Linear{
    pub fn new(in_features: usize, out_features: usize, bias: bool) -> Self{

        let weight_tensor = Tensor::kaiming_he_init(vec![in_features, out_features]);
        let bias_tensor;
        if bias{
            let bias_t = Tensor::zeros(vec![out_features]);
            bias_t.set_requires_grad(true);
            bias_tensor = Some(bias_t);
        }else{
            bias_tensor = None;
        }

        Linear { weights: weight_tensor, biases: bias_tensor}
    }

    pub fn forward(&self, input : &Tensor) -> Tensor{
        let mut output = input.matmul(&self.weights);
        if let Some(ref bias)= self.biases{
            // zato sto je bias shape samo vec![out_features] onda mogu da uzmem 0
            output = output + bias.reshape(vec![1, bias.shape()[0]]);
        }
        output
    }

    pub fn parameters(&self) -> Vec<Tensor>{
        let mut params = vec![self.weights.clone()];
        if let Some(ref bias) = self.biases{
            params.push(bias.clone());
        }
        params
    }
}

pub struct Conv2d{
    pub weights: Tensor,
    pub biases:Option<Tensor>,
    pub stride: usize,
    pub padding: usize
}

impl Conv2d{

    pub fn new(in_features: usize, out_features: usize, kernel_size: usize, stride: usize, padding: usize, bias: bool) -> Self{
        let weight_tensor = Tensor::kaiming_he_init(vec![out_features, in_features, kernel_size, kernel_size]);
        let bias_tensor;
        if bias{
            let bias_t = Tensor::zeros(vec![out_features]);
            bias_t.set_requires_grad(true);
            bias_tensor = Some(bias_t);
        }else{
            bias_tensor = None;
        }

        Conv2d{weights: weight_tensor, biases: bias_tensor, stride: stride, padding: padding}
    }

    pub fn forward(&self, input: &Tensor) -> Tensor{
        let output = input.conv2d(&self.weights, self.stride, self.padding, self.biases.as_ref());
        output
    }

    pub fn parameters(&self) -> Vec<Tensor>{
        let mut params = vec![self.weights.clone()];
        if let Some(ref bias) = self.biases{
            params.push(bias.clone());
        }
        params
    }
}
