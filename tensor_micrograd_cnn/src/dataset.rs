use core::num;
use std::collections::HashMap;
use crate::alexnet::{AlexNet};
use crate::{Model, Tensor};
use crate::data_loader::{Loader};
use crate::tensor::Precision;

pub struct S_Gradient_Descent{
    learning_rate : Precision,
    momentum: Precision,
    weight_decay: Precision,
    velocities: HashMap<usize, Vec<Precision>>,
}

impl S_Gradient_Descent{
    pub fn new(learning_rate : Precision, momentum: Precision, weight_decay: Precision) -> Self{
        S_Gradient_Descent { learning_rate: learning_rate, momentum: momentum, weight_decay: weight_decay, velocities: HashMap::new()}
    }
    

    pub fn update_params(&mut self, parameters: &[Tensor]){
        for parameter in parameters{
            parameter.s_gradient_descent_update_params(self.learning_rate, self.momentum, self.weight_decay, &mut self.velocities);
        }
    }

    pub fn zero_grad(&self, parameters: &[Tensor]){
        for param in parameters{
            param.zero_grad();
        }
    }

}

pub fn train(){
    let learning_rate = 0.001 as Precision;
    let momentum = 0.9 as Precision; // beta je 0.9 kod alexneta, EMA 0.9 znaci da otp. uzima 10 prethodnika u obzir
    let weight_decay = 0.0005 as Precision; // l2 reg je 0.0005
    let number_of_epochs = 10;
    let batch_size = 4;
    let image_height = 224;
    let image_width = 224;
    let model = AlexNet::new(10);
    let mut optimizer = S_Gradient_Descent::new(learning_rate, momentum, weight_decay);
    let parameters = model.parameters();

    //data ovde
    let os_name = std::env::consts::OS;
    println!("Current OS: {}", os_name);
    let mut loader;

    match os_name {
        "windows" => loader = Loader::new("D:\\diplomski\\Micrograd-in-Rust\\tensor_micrograd_cnn\\src\\dataset", batch_size, image_height, image_width),
        "linux" => loader = Loader::new("/mnt/d/diplomski/Micrograd-in-Rust/tensor_micrograd_cnn/src/dataset", batch_size, image_height, image_width),
        _ => loader = Loader::new("D:\\diplomski\\Micrograd-in-Rust\\tensor_micrograd_cnn\\src\\dataset", batch_size, image_height, image_width),
    }

    for i in 0..number_of_epochs{
        loader.reset_epoch();
        let mut batch_idx = 0 as usize;
        let mut batch_loss = 0.0;
        //println!("JEDNA EPOHA");

        while loader.has_batches(){
            //println!("JEDNA EPOHA BATCH");

            let(input_image, target_labels) = loader.next_batch();
            //println!("JEDNA EPOHA BATCH1");

            let output = model.forward(&input_image);
            //println!("JEDNA EPOHA BATCH2");


            let loss_tensor = Tensor::cross_entropy_loss(&output, &target_labels);
            let loss_val = loss_tensor.get_loss();

            //println!("JEDNA EPOHA BATCH3");

    
            optimizer.zero_grad(&parameters);
            loss_tensor.init_backward();
            optimizer.update_params(&parameters);

            if batch_idx % 5 == 0 {
                println!("Epoch {}/{}, Batch {}, Loss = {:.6}", i, number_of_epochs, batch_idx, loss_val);
            }
            batch_loss += loss_val;
            batch_idx += 1;
        }

        println!("Epoch {}/{} done, Avg epoch loss per batch: {:.6}", i, number_of_epochs, batch_loss/(batch_idx as f32));

    }
}