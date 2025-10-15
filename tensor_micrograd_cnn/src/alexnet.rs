use crate::layers::{Linear, Conv2d};
use crate::model::Model;
use crate::Tensor;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::io::prelude::*;
use std::io;
use crate::tensor::Precision;

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

impl Model for AlexNet{

    fn new(num_classes: usize) -> Self{
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

    fn forward(&self, input: &Tensor) -> Tensor{
        //println!("FORWARD0");

        let mut x = self.c2d_layer1.forward(input).relu();
        //println!("FORWARD1");

        x = x.maxpool_2d(3, 2);
        //println!("FORWARD2");


        x = self.c2d_layer2.forward(&x).relu();
        //println!("FORWARD3");

        x = x.maxpool_2d(3, 2);
        //println!("FORWARD4");


        x = self.c2d_layer3.forward(&x).relu();
        //println!("FORWARD5");

        x = self.c2d_layer4.forward(&x).relu();
        //println!("FORWARD6");


        x = self.c2d_layer5.forward(&x).relu();
        //println!("FORWARD7");

        x = x.maxpool_2d(3, 2);
        //println!("FORWARD8");


        let batch = x.shape()[0];
        x = x.reshape(vec![batch, 6 * 6 * 256]);

        //FC ovde sadrzi i batcheve
        x = self.fc_layer6.forward(&x).relu();
        //println!("FORWARD9");

        x = self.fc_layer7.forward(&x).relu();
        //println!("FORWARD10");

        x = self.fc_layer8.forward(&x);
        //println!("FORWARD11");

        x
    }

    fn parameters(&self) -> Vec<Tensor>{
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

    fn load_parameters(&self, filename: &str) -> bool {
        let mut file_name_str;
        if filename.is_empty(){
            file_name_str = "weights.txt";
        }else{
            file_name_str = filename;
        }

        let file = match File::open(file_name_str){
            Ok(file) => file,
            Err(_) => return false,
        };
        let mut buf_reader = BufReader::new(file);
        let mut contents = String::new();
        if buf_reader.read_to_string(&mut contents).is_err(){
            return false;
        };

        let mut lines = contents.lines();
        let mut lines_string_vec = Vec::new();

        for line in lines{
            lines_string_vec.push(line.to_string());
        }

        if lines_string_vec.len() < 2{
            return false;
        }

        let mut expected_precision;

        if cfg!(feature = "dtype-f64"){
            expected_precision = "precision=f64";
        }else{
            expected_precision = "precision=f32";
        }

        if lines_string_vec[0] != expected_precision{
            return false;
        }

        let mut line_counter = 2;
        for param in self.parameters(){
            if !param.load_weights(&lines_string_vec, &mut line_counter){
                return false;
            }
        }
        return true
    }

    fn save_parameters(&self, filename: &str, loss: Precision) -> bool {
        let mut file_name_str;
        if filename.is_empty(){
            file_name_str = "weights.txt";
        }else{
            file_name_str = filename;
        }

        let file_loss = match File::open(file_name_str){
            Ok(file) => {
                let mut buf_reader = BufReader::new(file);
                let mut contents = String::new();
                // ovde mora u if else jer je if expression i oba brancha if-a (kao i matcha) moraju da vracaju isti type tako da moram
                // else da stavim da vraca Some ako ovaj prvi vraca none, inace compiler error da ne moze biti ()
                if buf_reader.read_to_string(&mut contents).is_err(){
                    None
                }else{
                    let mut lines = contents.lines();

                    //izbacujemo precision line jer nam ne treba ovde
                    let _ = lines.next();

                    match lines.next(){
                        Some(loss_line) => loss_line.parse::<Precision>().ok(),
                        None => None
                    }
                }
            },
            Err(_) => None,
        };

        if let Some(file_loss_val) = file_loss{
            if file_loss_val < loss{
                return true
            }
        }

        let file = match  File::create(file_name_str){
            Ok(file) => file,
            Err(_) => return false,
        };

        let mut buf_writer = BufWriter::new(file);

        if cfg!(feature = "dtype-f64"){
            writeln!(buf_writer, "precision=f64").expect("Something went wrong");
        }else{
            writeln!(buf_writer, "precision=f32").expect("Something went wrong");
        }

        writeln!(buf_writer, "{}", loss).expect("Something went wrong");
        writeln!(buf_writer,"").expect("Something went wrong");

        //for ()

        // if writeln!(buf_writer).is_err(){
        //     return false;
        // }

        for param in self.parameters(){
            if !param.update_write_buf(&mut buf_writer){
                return false;
            }
        }

        if buf_writer.flush().is_err(){
            return false;
        }

        return true
    }
}