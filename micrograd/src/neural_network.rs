use crate::ValueWrapper;
use rand::Rng;

pub trait Call{
    fn call(&self, inputs: Vec<ValueWrapper>) -> Vec<ValueWrapper>;
}
pub trait Parameters{
    fn parameters(&self) -> Vec<ValueWrapper>;
}
 
pub struct Node{
    weights : Vec<ValueWrapper>,
    bias: ValueWrapper
}
impl Node{
    pub fn new(number_of_inputs : usize) -> Self{
        let mut rng = rand::thread_rng();
        let mut weights = Vec::new();
        for _ in 0..number_of_inputs{
            weights.push(ValueWrapper::minimal_new(rng.gen_range(-1.0..1.0)));
        }
        let bias = ValueWrapper::minimal_new(rng.gen_range(-1.0..1.0));
        Node { weights, bias}
    }
}
 
impl Call for Node{
    fn call(&self, inputs: Vec<ValueWrapper>) -> Vec<ValueWrapper>{
        assert_eq!(inputs.len(), self.weights.len());
        let mut sum = self.bias.clone();
        for i in 0..inputs.len(){
            sum = sum + self.weights[i].clone() * inputs[i].clone();    
        }
        vec![sum.tanh()]
    }
}
 
impl Parameters for Node{
    fn parameters(&self) -> Vec<ValueWrapper>{
        let mut all_params = self.weights.clone();
        all_params.push(self.bias.clone());
        all_params
    }
}

pub struct Layer{
    neurons: Vec<Node>
}

impl Layer{
    pub fn new(number_of_inputs : usize, number_of_outputs: usize) -> Self{
        let mut neurons = Vec::new();
        for _ in 0..number_of_outputs{
            let node : Node = Node::new(number_of_inputs);
            neurons.push(node)
        }
        Layer{ neurons}
    }
}

impl Call for Layer{
    fn call(&self, inputs: Vec<ValueWrapper>) -> Vec<ValueWrapper>{
        let mut out = Vec::new();
        for neuron in &self.neurons{
            let single_out = neuron.call(inputs.clone());
            out.push(single_out[0].clone());
        }
        out
    }
}

impl Parameters for Layer{
    fn parameters(&self) -> Vec<ValueWrapper>{
        let mut all_params = Vec::new();
        for neuron in &self.neurons{
            all_params.extend(neuron.parameters());
        }
        all_params
    }
}

pub struct MLP{
    layers: Vec<Layer>
}

impl MLP{
    pub fn new(number_of_inputs : usize, number_of_outputs : Vec<usize>) -> Self{
        let mut layers_args = Vec::new();
        layers_args.push(number_of_inputs);
        layers_args.extend(number_of_outputs);
        
        let mut layers = Vec::new();
        for curr_idx in 0..(layers_args.len() - 1){
            layers.push(Layer::new(layers_args[curr_idx], layers_args[curr_idx + 1]));
        }
        
        MLP { layers }
    }
}

impl Call for MLP{
    fn call(&self, inputs: Vec<ValueWrapper>) -> Vec<ValueWrapper> {
        let mut curr_inputs = inputs;
        for layer in &self.layers{
                curr_inputs = layer.call(curr_inputs);
        }
        curr_inputs   
    }
}

impl Parameters for MLP{
    fn parameters(&self) -> Vec<ValueWrapper>{
        let mut all_params = Vec::new();
        for layer in &self.layers{
            all_params.extend(layer.parameters());
        }
        all_params
    }
}

