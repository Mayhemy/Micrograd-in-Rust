use std::error::Error;
use std::fs;
use std::path::Path;
use rand::Rng;
use crate::Tensor;

use image::{io::Reader as ImageReader, EncodableLayout, GenericImageView};

pub fn is_image(image_name: &str) -> bool{
    let string_name = image_name.to_ascii_lowercase();
    if string_name.ends_with(".jpg") || string_name.ends_with(".jpeg") || string_name.ends_with(".png"){
        return true
    }
    return false
}

// gugl kaze da je ovaj fischer-yates shuffle dobar
pub fn shuffle(items: &mut Vec<(String, usize)>){
    let mut rng = rand::thread_rng();
    let mut i = items.len();
    while i > 1{
        i -= 1;
        let idx = rng.gen_range(0..i+1);
        let tmp = items[i].clone();
        items[i] = items[idx].clone();
        items[idx] = tmp;
    }
}

pub fn class_folders(root: &str) -> Vec<String>{
    let mut folder_names  = Vec::new();
    // unwrapujem result ovaj
    for entry in fs::read_dir(root).expect("The folder doesnt exist or you have no perms to read it (root)") {
        // unwrapujem read dir
        let entry = entry.expect("No file or you have no perms to read it (entry)");
        let path = entry.path();
        if path.is_dir(){
            if let Some(var_1) = path.file_name(){
                if let Some(name) = var_1.to_str(){
                    folder_names.push(name.to_string());
                }
            }
        }
    }

    folder_names.sort();
    folder_names
}

pub fn class_path_info(root: &str) -> Vec<(String, usize)>{
    let mut items : Vec<(String, usize)> = Vec::new();
    let classes = class_folders(root);

    println!("Classes: {}", classes.len());

    let mut class_idx = 0;
    while class_idx < classes.len(){
        let relative_class_path = &classes[class_idx];
        let path_to_class = Path::new(root).join(relative_class_path);


        for entry in fs::read_dir(path_to_class).expect("The folder doesnt exist or you have no perms to read it"){
            let entry = entry.expect("No file or you have no perms to read it");
            let path = entry.path();

            if path.is_file(){
                if let Some(folder_name) = path.file_name(){
                    if let Some(folder_name_str) = folder_name.to_str(){
                        if is_image(folder_name_str){
                            items.push((path.to_string_lossy().to_string(), class_idx));
                        }
                    }
                }
            }
        }

        class_idx += 1;
    }

    items
}

//in place samo prosledim output
fn load_image_nchw(path: &str, output: &mut [f32], input_height: usize , input_width: usize, offset: usize) -> bool {

    let result_open = ImageReader::open(path);
    let result_of_decode = result_open.unwrap().decode();
    let result_rgb = result_of_decode.unwrap().to_rgb8();
    let resized_img = image::imageops::resize(&result_rgb, input_width as u32, input_height as u32, image::imageops::FilterType::Lanczos3 );

    let channel_size = input_height * input_width;

    let mean =[0.485, 0.456, 0.406];
    let std = [0.229, 0.224, 0.225];

    for i in 0..input_height{
        for j in 0..input_width{
            
            let pixel = resized_img.get_pixel(j as u32, i as u32);
            let curr_idx = i * input_width + j;
            
            let r = pixel[0]as f32/ 255.0;
            let g = pixel[1]as f32/ 255.0;
            let b = pixel[2]as f32/ 255.0;

            output[offset + 0 * channel_size + curr_idx] = (r - mean[0]) / std[0];
            output[offset + 1 * channel_size + curr_idx] = (g - mean[1]) / std[1];
            output[offset + 2 * channel_size + curr_idx] = (b - mean[2]) / std[2];
        }
    }

    true
}

pub struct Loader{
    pub batch_size: usize,
    pub dataset_class_and_size : Vec<(String, usize)>,
    pub current_element : usize,
    pub image_height: usize,
    pub image_width: usize,
}

impl Loader{

    pub fn new(root: &str, batch_size: usize, image_height: usize, image_width: usize) -> Self{
        let mut dataset_info = class_path_info(root);
        shuffle(&mut dataset_info);
        Loader{ batch_size: batch_size, dataset_class_and_size: dataset_info, current_element: 0, image_height: image_height, image_width: image_width}
    }

    pub fn reset_epoch(&mut self){
        self.current_element = 0;
        shuffle(&mut self.dataset_class_and_size);
    }

    pub fn next_batch(&mut self) -> (Tensor, Tensor){

        let total_number_of_items = self.dataset_class_and_size.len();

        let mut number_of_items_to_read = 0;
        if self.current_element + self.batch_size > total_number_of_items{
            number_of_items_to_read = total_number_of_items - self.current_element;
        }else{
            number_of_items_to_read = self.batch_size;
        }
        let total_image_size = 3 * self.image_height * self.image_width;
        let mut image_data = vec![0.0f32; number_of_items_to_read * total_image_size];
        let mut label_data = vec![0.0f32; number_of_items_to_read];

        let mut counter = 0;
        while counter < number_of_items_to_read{

            let (path, class_idx) = &self.dataset_class_and_size[self.current_element + counter];
            let next_elem_offset = counter * total_image_size;

            let image_loaded = load_image_nchw(path, &mut image_data, self.image_height, self.image_width, next_elem_offset);

            label_data[counter] = *class_idx as f32;
            counter += 1;
        }

        self.current_element += number_of_items_to_read;

        let image_batch = Tensor::new_data(image_data, vec![number_of_items_to_read, 3, self.image_height, self.image_width]);
        let label_batch = Tensor::new_data(label_data, vec![number_of_items_to_read]);


        (image_batch, label_batch)
    }

    pub fn has_batches(&self) -> bool{
        self.current_element < self.dataset_class_and_size.len()
    }
}