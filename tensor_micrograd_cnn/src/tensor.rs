use core::num;
use std::rc::Rc;
use std::cell::RefCell;
use std::hash::{Hash, Hasher};
use std::collections::HashSet;
use std::usize::MAX;
use rand::Rng;
use std::cmp;
use std::ops::{Add, Mul};
use rand_distr::{Normal, Distribution};
use rayon::prelude::*;

#[derive(Debug, Clone)]
pub enum TensorOp{
	Reshape,
    Add,
    Mul,
    MatMul,
    ReLU,
    Im2Col,
    Conv2d,
    MaxPool2d,
    //Sum,
    //Mean,
    None
}


#[cfg(feature = "dtype-f32")]
pub type Precision = f32;
#[cfg(feature = "dtype-f64")]
pub type Precision = f64;


pub struct TensorCore{
    data: Vec<Precision>, // xi
    shape: Vec<usize>, // bice nam korisno
    requires_grad: bool, // 
    grad: Vec<Precision>,
    backward: Option<Box<dyn Fn()>>,
	op: TensorOp,
	prev: Vec<Rc<RefCell<TensorCore>>>,
}

// Najbolje sto sam nasao online, mora ovako jer backward funkcija mora specijalno da se formatira i ne moze da se implementira Debug za Box<dyn Fn()>
impl std::fmt::Debug for TensorCore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TensorCore")
            .field("shape", &self.shape)
            .field("requires_grad", &self.requires_grad)
            .field("op", &self.op)
            .field("data", &self.data)
            .field("has_backward", &self.backward.is_some())
            .field("prev_count", &self.prev.len())
            .finish()
    }
}

// Need to wrap my value so that i can implement Add and Mul since Rc<RefCell<Value>> is external and Add and mul are external??
// no idea why this is an issue in rust but ok...
#[derive(Debug)]
pub struct Tensor(pub Rc<RefCell<TensorCore>>);

impl Clone for Tensor {
    fn clone(&self) -> Self {
        Tensor(self.0.clone())
    }
}

impl PartialEq for Tensor {
	fn eq(&self, other: &Self) -> bool{
		Rc::ptr_eq(&self.0, &other.0)
	}
}

impl Eq for Tensor {}

impl Hash for Tensor {
	fn hash<H: Hasher>(&self, state: &mut H){
		let pointer_raw: *const RefCell<TensorCore> = &*self.0;
		pointer_raw.hash(state);
	}
}


impl Tensor{
    fn new(data :Vec<Precision>, shape: Vec<usize>, requires_grad: bool, children: Vec<Rc<RefCell<TensorCore>>>, op: TensorOp) -> Tensor{
        // zbog rustovog nasledjivanja ownershipa ovo mora
        let data_len = data.len();

        let tensor = TensorCore{
            data,
            shape: shape,
            requires_grad,
            grad: vec![0.0 as Precision; data_len],
            backward: None,
            op,
            prev: children,
        };
        Tensor(Rc::new(RefCell::new(tensor)))
    }

    pub fn new_data(data :Vec<Precision>, shape: Vec<usize>) -> Self{
        Tensor::new(data, shape, false, vec![], TensorOp::None)
    }
    
    pub fn zeros(shape: Vec<usize>) -> Tensor{
        let mut flattened_size = 1;
        for dimension in &shape{
            flattened_size *= dimension;
        }
        let data = vec![0.0; flattened_size];
        let requires_grad = false;
        let children = vec![];
        let op = TensorOp::None;
        Tensor::new(data, shape, requires_grad, children, op)
    }

    pub fn ones(shape: Vec<usize>) -> Tensor{
        let mut flattened_size = 1;
        for dimension in &shape{
            flattened_size *= dimension;
        }
        let data = vec![1.0; flattened_size];
        let requires_grad = false;
        let children = vec![];
        let op = TensorOp::None;
        Tensor::new(data, shape, requires_grad, children, op)
    }

    pub fn randn_uniform_init(shape: Vec<usize>) -> Tensor{
        let mut flattened_size = 1;
        for dimension in &shape{
            flattened_size *= dimension;
        }
        let mut rng = rand::thread_rng();
        let mut data = Vec::with_capacity(flattened_size);
        for i in 0..flattened_size{
            data.push(rng.gen_range(-1.0 as Precision..1.0 as Precision));
        }
        let requires_grad = true;
        let children = vec![];
        let op = TensorOp::None;

        Tensor::new(data, shape, requires_grad, children, op)
    }

    pub fn kaiming_he_init(shape: Vec<usize>) -> Tensor{
        let mut inputs_to_each_neuron = 1;
        // za conv uzimam ceo filter a za linear samo in_features/ in_channels
        // 2 opcije
        // za linear shape len je 2
        // za 
        if shape.len() == 2{
            inputs_to_each_neuron *= shape[0];
        }else if shape.len() == 4{
            inputs_to_each_neuron *= shape[1] * shape[2] * shape[3];
        }

        let mut flattened_size = 1;
        for dimension in &shape{
            flattened_size *= dimension;
        }

        let bound = (2.0 / inputs_to_each_neuron as f64).sqrt();
        let normal = Normal::new(0.0, bound).unwrap(); // `unwrap` is okay here, as std_dev > 0
        
        let mut rng = rand::thread_rng();
        let mut data = Vec::with_capacity(flattened_size);
        for i in 0..flattened_size{
            let value = normal.sample(&mut rng) as Precision;
            data.push(value);
        }
        let requires_grad = true;
        let children = vec![];
        let op = TensorOp::None;

        Tensor::new(data, shape, requires_grad, children, op)
    }

    pub fn zero_grad(&self){
        self.0.borrow_mut().grad.fill(0.0 as Precision);
    }

    pub fn set_requires_grad(&self, requires_grad: bool){
        self.0.borrow_mut().requires_grad = requires_grad;
    }

    pub fn get_loss(&self) -> f32{
        self.0.borrow().data[0].clone()
    }

    pub fn reshape(&self, new_shape: Vec<usize>) -> Tensor{
        let data = self.0.borrow().data.clone();

        let mut new_size: usize = 1;
        let mut old_size: usize = 1;

        let old_shape = &self.0.borrow().shape;

        for i in 0..old_shape.len(){
            old_size *=  old_shape[i];
        }
        for i in 0..new_shape.len(){
            new_size *=  new_shape[i];
        }
        assert_eq!(old_size, new_size, "The overall number of elements must stay the same when reshaping!");
        let result = Tensor::new(data, new_shape, true, vec![self.0.clone()], TensorOp::Reshape);

        let result_copy = result.0.clone();
        let self_copy = self.0.clone();

        result.0.borrow_mut().backward = Some(Box::new(move || {
            let result_borrow = result_copy.borrow();
            let mut self_borrow = self_copy.borrow_mut();

            for i in 0..result_borrow.grad.len(){
                self_borrow.grad[i] += result_borrow.grad[i];
            }
        }));
        
        result 
    }

    pub fn index_1d(indices: Vec<usize>, shape: &[usize]) -> usize{
        // indeksi ce nam biti u obliku (1,0,1,1) a shape je (2,2,2,2), krecemo od kraja i trazimo 1d index
        let mut index_1d = 0;
        let mut multiplier = 1;
        for i in (0..indices.len()).rev(){
            index_1d += indices[i] * multiplier;
            multiplier *= shape[i]; // ovo radimo jer i za indices i za shape idemo od kraja
        }
        index_1d
    }
    
    pub fn shape(&self) -> Vec<usize>{
        self.0.borrow().shape.clone()
    }

    // inverzna operacija od index_1d (od 1d indeksa pravimo n-dimenzioni indeks)
    // Ovde pravimo mnogo arrayeva svakim pozivom fora, pa bi optimizacija bila da napisemo funkciju koja in place menja array
    pub fn index_1d_to_nd(index_1d: usize, shape: &[usize]) -> Vec<usize>{
        let mut indices = vec![0; shape.len()];
        let mut remainder = index_1d;
        for i in (0..shape.len()).rev(){
            indices[i] = remainder % shape[i];
            remainder /= shape[i];
        }
        indices
    }

    pub fn index_1d_to_nd_inplace(mut index_1d: usize, shape: &[usize], result_idx: &mut [usize]){

        for i in (0..shape.len()).rev(){
            result_idx[i] = index_1d % shape[i];
            index_1d /= shape[i];
        }
    }

    //ovo se zove samo na loss tako da ce data_len biti 1 => grad = [1.0]
    pub fn init_backward(&self){
        let data_len = self.0.borrow().data.len();
        self.0.borrow_mut().grad = vec![1.0; data_len];
        let topological_order = self.build_topological_graph();

        // Kao u microgradu krecemo od kraja i racunamo gradijente
        for tensor in topological_order.iter().rev(){
            if let Some(ref backward_fn) = tensor.0.borrow().backward{
                backward_fn();
            }
        }
    }

    pub fn cross_entropy_loss(prediction: &Tensor, target: &Tensor) -> Tensor{
        let prediction_borrow = prediction.0.borrow();
        let target_borrow = target.0.borrow();

        let (batches, num_classes) = (prediction.shape()[0], prediction.shape()[1]);

        //let mut max_prediction: Vec<f32> = vec![Precision::MIN; batches];
        //let mut log_sum_exp: Vec<f32> = vec![0.0 as Precision; batches];
        //let mut negative_log_likelihood: Vec<f32> = vec![0.0 as Precision; batches];


        let mut batch_loss = 0.0 as Precision;

        for i in 0..batches{
            let mut row_max = Precision::MIN;
            for j in 0..num_classes{
                let curr_val = prediction_borrow.data[i * num_classes + j ];
                if curr_val > row_max{
                    row_max = curr_val;
                }
            }

            let mut sum_exp = 0.0 as Precision;;
            for j in 0..num_classes{
                sum_exp += (prediction_borrow.data[i * num_classes + j] - row_max).exp()
            }

            let lse = sum_exp.ln() + row_max;

            let true_class = prediction_borrow.data[i * num_classes + target_borrow.data[i] as usize];

            batch_loss += lse - true_class;

        }

        batch_loss = batch_loss / batches as Precision;

        let result = Tensor::new(vec![batch_loss], vec![1], true, vec![prediction.0.clone()], TensorOp::None);


        let prediction_copy = prediction.clone();
        let target_copy = target.clone();

        result.0.borrow_mut().backward = Some(Box::new(move || {
            let (max_per_batch, denominator_per_batch) = {
                let logits = prediction_copy.0.borrow();
                let targets = target_copy.0.borrow();
                //let mut logits_mutable = prediction_copy.0.borrow_mut();

                let mut max_per_batch = vec![0.0 as Precision; batches];
                let mut denominator_per_batch = vec![0.0 as Precision; batches];

                for i in 0..batches {
                    let mut batch_max = Precision::MIN;
                    for j in 0..num_classes{
                        if logits.data[i * num_classes + j] > batch_max{
                            batch_max = logits.data[i * num_classes + j];
                        }
                    }
                    max_per_batch[i] = batch_max;
                    let mut denominator_sum = 0.0 as Precision;
                    for j in 0..num_classes {
                        denominator_sum += (logits.data[i * num_classes + j] - batch_max).exp();
                    }
                    denominator_per_batch[i] = denominator_sum;
                }
                (max_per_batch, denominator_per_batch)
            };
                    let targets = target_copy.0.borrow();
                    let mut logits_mutable = prediction_copy.0.borrow_mut();
                    for i in 0..batches{
                        for j in 0..num_classes {
                            let numerator = (logits_mutable.data[i * num_classes + j] - max_per_batch[i]).exp();
                            let softmax = numerator / denominator_per_batch[i];
                            // ovde je ovo de facto one-hot, cak i ako ne konstruktujemo one-hot array
                            let mut one_hot = 0.0;
                            if j == targets.data[i] as usize {
                                one_hot = 1.0;
                            }
                            logits_mutable.grad[i * num_classes + j] += (softmax - one_hot) / (batches as Precision);
                        }
                    }
        }));
        

        result

    }

    pub fn s_gradient_descent_update_params(&self, learning_rate : Precision){
        let mut params = self.0.borrow_mut();
        for i in 0..params.data.len(){
            params.data[i] -= learning_rate * params.grad[i];
        }
    }

    pub fn build_topological_graph(&self) -> Vec<Tensor>{
        let mut topologically_sorted: Vec<Tensor> = Vec::new();
		let mut visited: HashSet<Tensor> = HashSet::new();
		self.topo_recursive(&mut topologically_sorted,&mut visited);
		topologically_sorted
    }

    pub fn topo_recursive(&self, topologically_sorted: &mut Vec<Tensor>, visited: &mut HashSet<Tensor>){

        // za contains moramo da implementiramo PartialEq i Eq za TensorWrapper
        if !visited.contains(self){
            visited.insert(self.clone());
            for child in self.0.borrow().prev.iter(){
                // posto su prev storovani kao Vec<Rc<RefCell<Tensor>> moramo svaki da castujemo u wrapper
                let wrapper = Tensor(child.clone());
                wrapper.topo_recursive(topologically_sorted, visited);
            }
            topologically_sorted.push(self.clone());
        }
    }

    //daje nam shape resulta za ab
    pub fn util_calculate_broadcast_shapes(shape_a: &[usize], shape_b: &[usize]) -> Result<Vec<usize>, String>{
        // rev je O(n) tako da cemo implementirati bez reva
        let max_len = cmp::max(shape_a.len(), shape_b.len());
        let mut broadcast_shape = vec![0; max_len];
        for i in 0..max_len{
            // MORA curr_a,b = 1 ne 0 jer dole imam else kad je curr_a == 1, result_dim = curr_b
            //otkrio bug automatski AI test generation
            let mut curr_a = 1;
            let mut curr_b = 1;

            if i < shape_a.len(){
                curr_a = shape_a[shape_a.len() - i - 1]
            }

            if i < shape_b.len(){
                curr_b = shape_b[shape_b.len() - i - 1]
            }

            let mut result_dimension = 1;

            if curr_a == curr_b{
                result_dimension = curr_a;
            }else{
                if curr_a == 1{
                    result_dimension = curr_b;
                }else if curr_b == 1{
                    result_dimension = curr_a;
                }else{
                    return Err(format!("Operands could not be broadcast together: curr_a = {}, curr_b = {}", curr_a, curr_b));
                    //return Err(format!("Operands could not be broadcast together: shape_a[{}] = {} != shape_b[{}] = {}", i, shape_a[i], i, shape_b[i]));
                }
            }
            broadcast_shape[max_len - i - 1] = result_dimension;
        }

        Ok(broadcast_shape)
    }

    // ova funkcija nam daje 1d index za prvobitni array(original) za neki broadcast shape i broadcast indices
    //lose, implementiraj opet kad razumes glupane
    //mislim da je sad dobro?
    pub fn util_get_broadcast_idx_value(original_shape: &[usize], broadcast_shape: &[usize], broadcast_indices: &[usize]) -> usize{
        let difference_in_shape_dimensions = broadcast_shape.len() - original_shape.len();
        let mut original_indices = vec![0; original_shape.len()];
        for i in 0..original_shape.len(){
            let broadcast_shape_index = i + difference_in_shape_dimensions;

            if original_shape[i] == 1 && broadcast_shape[broadcast_shape_index] > 1{
                original_indices[i] = 0;
            }else{
                original_indices[i] = broadcast_indices[broadcast_shape_index];
            }
        }

        let flattened_idx = Self::index_1d(original_indices, original_shape);
        flattened_idx
    }


    pub fn matmul(&self, other: &Tensor) -> Tensor{
        // zbog borrow checkera moram da len izracunam u jednom bloku
        //let self_shape = self.0.borrow();
        //let other_shape = other.0.borrow();

        let(self_shape_len, other_shape_len) = {
            let self_borrow = self.0.borrow();
            let other_borrow = other.0.borrow();

            (self_borrow.shape.len(), other_borrow.shape.len())
        };
        match (self_shape_len, other_shape_len) {
            (2,2) => self.matmul_2d(&other),
            (3,3) => self.matmul_batched_3d(&other),
            (3,2) => self.matmul_3d_and_2d(&other),
            _ => panic!(),
        }
    }

    fn matmul_2d(&self, other: &Tensor) -> Tensor{
        let min_parallel_len = 8;
        let shape_a = &self.0.borrow().shape;
        let shape_b = &other.0.borrow().shape;

        assert_eq!(shape_a[1], shape_b[0]);

        let (m, n) = (shape_a[0], shape_a[1]);
        let (x, y)   = (shape_b[0], shape_b[1]);

        let a_borrow = self.0.borrow();
        let b_borrow = other.0.borrow();

        let a_data = &a_borrow.data;
        let b_data = &b_borrow.data;

        let mut result_data = vec![0.0; m*y];


        result_data.par_chunks_mut(y).with_min_len(min_parallel_len).enumerate().for_each(|(i, result_row)|{
            let a_row = i * n;
            for j in 0..y{
                let mut element = 0.0 as Precision;
                for k in 0..n{
                    element = a_data[a_row] * b_data[y * k + j];
                }
                result_row[j] = element;
            }
        });

        // for i in 0..m{
        //     let a_row = i * n;
        //     let result_row = i * y;
        //     for j in 0..y{
        //         let mut element = 0.0;
        //         //mogu i n i x
        //         for k in 0..n{
        //             element += a_data[a_row + k] * b_data[y * k + j]
        //         }
        //         result_data[result_row + j] = element;
        //     }
        // }

        let result = Tensor::new(result_data, vec![m, y], true, vec![self.0.clone(), other.0.clone()], TensorOp::MatMul);

        //backwards sad
        // ovo mora jer ce kasnije ovaj backwards biti pozvan kada ne budu vise ove varijable postojale
        let result_copy = result.0.clone();
        let self_copy = self.0.clone();
        let other_copy = other.0.clone();
        let min_parallel_len_heap = min_parallel_len.clone();

        // a je prvi clan b je drugi clan c = a + b; c = self + other
        result.0.borrow_mut().backward = Some(Box::new(move || {

        {
            let result_grad = result_copy.borrow().grad.clone();
            let right_operand_data = other_copy.borrow().data.clone();
            let mut left_operand = self_copy.borrow_mut();

             // n == x
            // posto je G * B^T => G je shape-a m,y (shape_a[0], shape_b[1]) a B^T je (shape_b[1],shape_b[0)) y,x - iteriramo po m i x tj. po A jer racunamo izvod za A
            // posto mi ne transponujemo matricu direktno to znaci da pri indeksiranju moramo da je "transponujemo" pa onda mul i transpose zajedno daju efekat
            // da mnozimo red sa redom.
            left_operand.grad.par_chunks_mut(n).with_min_len(min_parallel_len_heap).enumerate().for_each(|(i, left_operand_row)|{
                let result_row = i * y;
                for j in 0..n{
                    let b_col = j * y;
                    let mut element = 0.0;
                    for k in 0..y{
                        element += result_grad[result_row + k] * right_operand_data[b_col + k];
                    }
                    left_operand_row[j] += element;
                }
            });
        };
        {
            let result_grad = result_copy.borrow().grad.clone();
            let left_operand_data = self_copy.borrow().data.clone();
            let mut right_operand = other_copy.borrow_mut();
            // n == x ne zaboraviti

            //m n , x y
            right_operand.grad.par_chunks_mut(y).with_min_len(min_parallel_len_heap).enumerate().for_each(|(i, right_operand_row)|{
                for j in 0..y{
                    let mut element = 0.0;
                    for k in 0..m{
                        element += left_operand_data[k * n + i] * result_grad[k * y + j];
                    }
                    right_operand_row[j] += element;
                }
            })
        }}));
        // {
        //     if is_same_operand{
        //         let grad_result = result_copy.borrow();
        //         //let a_data = &self_copy.borrow().data;
        //         let b_data_vec = other_copy.borrow().data.clone();
        //         let mut self_grad = self_copy.borrow_mut();
        //         //let mut other_grad = other_copy.borrow_mut();
    
    
        //         // n == x
        //         // posto je G * B^T => G je shape-a m,y (shape_a[0], shape_b[1]) a B^T je (shape_b[1],shape_b[0)) y,x - iteriramo po m i x tj. po A jer racunamo izvod za A
        //         // posto mi ne transponujemo matricu direktno to znaci da pri indeksiranju moramo da je "transponujemo" pa onda mul i transpose zajedno daju efekat
        //         // da mnozimo red sa redom.
        //         for i in 0..m{
        //             let result_row = i * y;
        //             let a_row =  i * n;
        //             for j in 0..n{
        //                 let b_col = j * y;
        //                 let mut element = 0.0;
        //                 for k in 0..y{
        //                     element += grad_result.grad[result_row + k] * b_data_vec[b_col + k];
        //                 }
        //                 self_grad.grad[a_row + j] += element;
        //             }
        //         }
        //     }else{
        //         let grad_result = result_copy.borrow();
        //         //let a_data = &self_copy.borrow().data;
        //         let b_data = other_copy.borrow();
        //         let mut self_grad = self_copy.borrow_mut();
        //         //let mut other_grad = other_copy.borrow_mut();


        //         // n == x
        //         // posto je G * B^T => G je shape-a m,y (shape_a[0], shape_b[1]) a B^T je (shape_b[1],shape_b[0)) y,x - iteriramo po m i x tj. po A jer racunamo izvod za A
        //         // posto mi ne transponujemo matricu direktno to znaci da pri indeksiranju moramo da je "transponujemo" pa onda mul i transpose zajedno daju efekat
        //         // da mnozimo red sa redom.
        //         for i in 0..m{
        //             let result_row = i * y;
        //             let a_row =  i * n;
        //             for j in 0..n{
        //                 let b_col = j * y;
        //                 let mut element = 0.0;
        //                 for k in 0..y{
        //                     element += grad_result.grad[result_row + k] * b_data.data[b_col + k];
        //                 }
        //                 self_grad.grad[a_row + j] += element;
        //             }
        //         }
        //     }
        // };
        // {
        //     if is_same_operand{
        //         let grad_result = result_copy.borrow();
        //         let a_data_vec = self_copy.borrow().data.clone();
        //         let mut other_grad = other_copy.borrow_mut();
        //         // n == x ne zaboraviti
    
        //         //m n , x y
        //         for i in 0..x{
        //             let b_row = i * y;
        //             for j in 0..y{
        //                 let mut element = 0.0;
        //                 for k in 0..m{
        //                     element += a_data_vec[k * n + i] * grad_result.grad[k * y + j];
        //                 }
        //                 other_grad.grad[b_row + j] += element;
        //             }
        //         }
        //     }else{
        //         let grad_result = result_copy.borrow();
        //         let a_data = self_copy.borrow();
        //         let mut other_grad = other_copy.borrow_mut();
        //         // n == x ne zaboraviti
    
        //         //m n , x y
        //         for i in 0..x{
        //             let b_row = i * y;
        //             for j in 0..y{
        //                 let mut element = 0.0;
        //                 for k in 0..m{
        //                     element += a_data.data[k * n + i] * grad_result.grad[k * y + j];
        //                 }
        //                 other_grad.grad[b_row + j] += element;
        //             }
        //         }
        //     }
        // }
        // }));

        result
    }

    fn matmul_batched_3d(&self, other: &Tensor) -> Tensor{
        let shape_a = &self.0.borrow().shape;
        let shape_b = &other.0.borrow().shape;

        assert_eq!(shape_a[0], shape_b[0]);
        assert_eq!(shape_a[2], shape_b[1]);

        let (batch, m, n ) = (shape_a[0], shape_a[1], shape_a[2]);
        let (_, n, y)   = (shape_b[0], shape_b[1], shape_b[2]);

        let a_borrow = self.0.borrow();
        let b_borrow = other.0.borrow();

        let a_data = &a_borrow.data;
        let b_data = &b_borrow.data;

        let mut result_data = vec![0.0; batch * m*y];

        for b in 0..batch{
            let a_offset = b * (m * n);
            let b_offset = b * (n * y);
            let c_offset = b * (m * y);
            for i in 0..m{
                for j in 0..y{
                    let mut sum = 0.0;
                    for k in 0..n{
                        let a_idx = a_offset +  i * n + k;
                        let b_idx = b_offset + k * y + j;
                        sum += a_data[a_idx] * b_data[b_idx];
                    }
                    result_data[c_offset + i * y + j] = sum;
                }
            }
        }

        let result = Tensor::new(result_data, vec![batch, m, y], true, vec![self.0.clone(), other.0.clone()], TensorOp::MatMul);


        let result_copy = result.0.clone();
        let self_copy = self.0.clone();
        let other_copy = other.0.clone();

        result.0.borrow_mut().backward = Some(Box::new(move || {

        }));

        result
    }
    fn matmul_3d_and_2d(&self, other: &Tensor) -> Tensor{

        let shape_a = &self.0.borrow().shape;
        let shape_b = &other.0.borrow().shape;

        assert_eq!(shape_a[2], shape_b[0]);

        let (batch, m, n ) = (shape_a[0], shape_a[1], shape_a[2]);
        let (n, y)   = (shape_b[0], shape_b[1]);

        let a_borrow = self.0.borrow();
        let b_borrow = other.0.borrow();

        let a_data = &a_borrow.data;
        let b_data = &b_borrow.data;

        let mut result_data = vec![0.0; batch * m*y];

        for b in 0..batch{
            let a_offset = b * (m * n);
            let c_offset = b * (m * y);
            for i in 0..m{
                let a_row = a_offset + i * n;
                let c_row = c_offset + i * y;
                for j in 0..y{
                    let mut sum = 0.0;
                    for k in 0..n{
                        sum += a_data[a_row + k] * b_data[k * y + j];
                    }
                    result_data[c_row + j] = sum;
                }
            }
        }

        let result = Tensor::new(result_data, vec![batch, m, y], true, vec![self.0.clone(), other.0.clone()], TensorOp::MatMul);

        //backwards sad
        // ovo mora jer ce kasnije ovaj backwards biti pozvan kada ne budu vise ove varijable postojale
        let result_copy = result.0.clone();
        let self_copy = self.0.clone();
        let other_copy = other.0.clone();

        let is_same_operator = Rc::ptr_eq(&self_copy, &other_copy);

        // a je prvi clan b je drugi clan c = a + b; c = self + other
        result.0.borrow_mut().backward = Some(Box::new(move || {
            {
                if is_same_operator{
                    let grad_result = result_copy.borrow();
                    //let a_data = &self_copy.borrow().data;
                    let b_data_vec = other_copy.borrow().data.clone();
                    let mut self_grad = self_copy.borrow_mut();
                    //let mut other_grad = other_copy.borrow_mut();
    
    
                    // n == x
                    // Sada radimo [batch, m, n] i [x, y], znaci kada racunamo gA = G(b,m,y) i B^T(y, n) ali trenutni je B(n, y), za svaki batch - prakticno posto je mnozenje
                    // i T onda kao da mnozim red iz prve i red iz druge matrice kad se sve to sredi
                    for b in 0..batch{
                        let result_base_idx = b * (m * y);
                        let self_base_idx = b * m * n;
                        for i in 0..m{
                            let result_offset = i * y;
                            let self_offset = i * n;
                            for j in 0..n{
                                let b_offset = j * y;
                                let mut element = 0.0;
                                for k in 0..y{
                                    let result_idx = result_base_idx + result_offset + k;
                                    let b_idx = b_offset + k;
                                    element += grad_result.grad[result_idx] * b_data_vec[b_idx];
                                }
                                self_grad.grad[self_base_idx + self_offset + j] += element;
                            }
                        }
                    }
                }else{
                    let grad_result = result_copy.borrow();
                    //let a_data = &self_copy.borrow().data;
                    let b_data = other_copy.borrow();
                    let mut self_grad = self_copy.borrow_mut();
                    //let mut other_grad = other_copy.borrow_mut();


                    // n == x
                    // Sada radimo [batch, m, n] i [x, y], znaci kada racunamo gA = G(b,m,y) i B^T(y, n) ali trenutni je B(n, y), za svaki batch - prakticno posto je mnozenje
                    // i T onda kao da mnozim red iz prve i red iz druge matrice kad se sve to sredi
                    for b in 0..batch{
                        let result_base_idx = b * (m * y);
                        let self_base_idx = b * m * n;
                        for i in 0..m{
                            let result_offset = i * y;
                            let self_offset = i * n;
                            for j in 0..n{
                                let b_offset = j * y;
                                let mut element = 0.0;
                                for k in 0..y{
                                    let result_idx = result_base_idx + result_offset + k;
                                    let b_idx = b_offset + k;
                                    element += grad_result.grad[result_idx] * b_data.data[b_idx];
                                }
                                self_grad.grad[self_base_idx + self_offset + j] += element;
                            }
                        }
                    }
                }
            };
            {
                if is_same_operator{
                    let grad_result = result_copy.borrow();
                    let a_data_vec = self_copy.borrow().data.clone();
                    let mut other_grad = other_copy.borrow_mut();
    
                    // ovde moram da akumuliram preko batcheva a svaki batch je transponovan (mn - ny)
                    for i in 0..n{
                        let other_offset = i * y;
                        for j in 0..y{
                            let mut element = 0.0;
                            for b in 0..batch{
                                let result_base_idx = b * m * y;
                                let a_base_idx = b * m * n;
                                for k in 0..m{
                                    let a_idx = a_base_idx + k * n + i;
                                    let result_idx = result_base_idx + k * y + j;
                                    element += a_data_vec[a_idx] * grad_result.grad[result_idx];
                                }
                            }
                            other_grad.grad[other_offset + j] += element;
                        }
                    }
                }else{
                    let grad_result = result_copy.borrow();
                    let a_data = self_copy.borrow();
                    let mut other_grad = other_copy.borrow_mut();
    
                    // ovde moram da akumuliram preko batcheva a svaki batch je transponovan (mn - ny)
                    for i in 0..n{
                        let other_offset = i * y;
                        for j in 0..y{
                            let mut element = 0.0;
                            for b in 0..batch{
                                let result_base_idx = b * m * y;
                                let a_base_idx = b * m * n;
                                for k in 0..m{
                                    let a_idx = a_base_idx + k * n + i;
                                    let result_idx = result_base_idx + k * y + j;
                                    element += a_data.data[a_idx] * grad_result.grad[result_idx];
                                }
                            }
                            other_grad.grad[other_offset + j] += element;
                        }
                    }
                }
            }
        }));

        result
    }

    pub fn relu(&self) -> Tensor{
        
        let data_borrow = self.0.borrow();
        let data = &data_borrow.data;
        let shape = data_borrow.shape.clone();
        
        let mut result_data = vec![0.0; data.len()];

        for i in 0..data.len(){
            result_data[i] = data[i].max(0.0);
        }

        let result = Tensor::new(result_data, shape,true,vec![self.0.clone()], TensorOp::ReLU);

        let result_copy = result.0.clone();
        let self_copy = self.0.clone();

        result.0.borrow_mut().backward = Some(Box::new(move || {

            let result_borrow = result_copy.borrow();
            let input_data = self_copy.borrow().data.clone(); // ovo radim da ne bih imao ref na immutable i onda odmah ispod mutable (error baca)
            let mut self_grad = self_copy.borrow_mut();
            
            for i in 0..result_borrow.grad.len(){
                if input_data[i] > 0.0{
                    self_grad.grad[i] += result_borrow.grad[i];
                }
            }
        }));
        result
    }

    pub fn maxpool_2d(&self, kernel_size: usize, stride: usize) -> Tensor{
        let data_borrow = self.0.borrow();
        let data = &data_borrow.data;

        assert_eq!(data_borrow.shape.len(), 4, "Maxpool needs 4d input.");

        let (batch, channels, height, width) = (data_borrow.shape[0], data_borrow.shape[1], data_borrow.shape[2], data_borrow.shape[3]);

        assert!(kernel_size <= height && kernel_size <= width);

        assert!(stride >= 1 && kernel_size > 0);

        let out_height = (height - kernel_size)/ stride + 1;
        let out_width = (width - kernel_size)/ stride + 1;

        let mut result_data = Vec::with_capacity(batch * channels * out_height * out_width);
        let mut max_indices = Vec::with_capacity(batch * channels * out_height * out_width);

        for b in 0..batch{
            for c in 0..channels{
                let base = (b * channels + c) * height * width;
                let mut y = 0;
                while y + kernel_size <= height{
                    let mut x = 0;
                    while x + kernel_size <= width{
                        let mut max_val = Precision::NEG_INFINITY;
                        let mut max_idx = 0;

                        let mut ky = 0;
                        while ky < kernel_size{
                            let curr_elem_base = base + (y + ky) * width;
                            let mut kx = 0;
                            while kx < kernel_size{
                                let curr_element = data[curr_elem_base + x + kx];
                                if curr_element > max_val{
                                    max_val = curr_element;
                                    max_idx = curr_elem_base + x + kx;
                                }
                                kx += 1;
                            }
                            ky += 1;
                        }
                        result_data.push(max_val);
                        max_indices.push(max_idx);

                        x += stride;
                    }
                    y += stride;
                }
            }

        }

        let result = Tensor::new(result_data, vec![batch, channels, out_height, out_width],true,vec![self.0.clone()], TensorOp::MaxPool2d);

        let result_copy = result.0.clone();
        let self_copy = self.0.clone();
        let max_indices_backwards = Rc::new(max_indices);
        result.0.borrow_mut().backward = Some(Box::new(move || {
            let result_borrow = result_copy.borrow();
            let mut self_grad = self_copy.borrow_mut();

            for i in 0..max_indices_backwards.len(){
                self_grad.grad[max_indices_backwards[i]] += result_borrow.grad[i];
            }
        }));
        result
    }

    pub fn im2col(&self, kernel_h: usize, kernel_w: usize, stride: usize, padding: usize) -> Tensor{

        let data_borrow = self.0.borrow();
        let data = &data_borrow.data;
        let shape = data_borrow.shape.clone();

        let (batches, channels, height, width) = (shape[0], shape[1], shape[2], shape[3]);
        let receptive_field = kernel_h * kernel_w * channels;

        let out_height = (height + 2 * padding - kernel_h)/stride + 1;
        let out_width = (width + 2 * padding - kernel_w)/stride +1;
        let number_of_receptive_fields = out_height * out_width;
        let mut col_data = vec![0.0 as Precision; batches * number_of_receptive_fields * receptive_field];

        for b in 0..batches{
            let base_data_idx = b * channels * height * width;
            for o_h in 0..out_height{
                for o_w in 0..out_width{

                    for c in 0..channels{

                        let current_channel_idx = c * height * width;

                        let col_matrix_col_idx = b * number_of_receptive_fields + o_h * out_width + o_w;
                    
                        for kh in 0..kernel_h{
                            for kw in 0..kernel_w{
                                let input_idx_x = o_w * stride + kw;
                                let input_idx_y = o_h * stride + kh;

                                let value;

                                // ovde mi je dosta pomogao onaj stanford cs231n vizuelizacija konvolucije
                                if input_idx_x >= padding && input_idx_x - padding < width && input_idx_y >= padding  && input_idx_y - padding < height{
                                    let real_idx_x = input_idx_x - padding;
                                    let real_idx_y = input_idx_y - padding;
                                    value = data[base_data_idx + current_channel_idx + real_idx_y * width + real_idx_x];
                                }else{
                                    value = 0.0 as Precision;
                                }

                                let number_of_rows_to_skip = (kh * kernel_w + kw) + (c * kernel_w * kernel_h);
                            
                                // prakticno [363 x 3025] number_of_receptive_fields je 3025 a 363 mi je filter_x * filter_y * channels, batchevi se stackuju takodje po koloni
                                //znaci prvo odradim prvi batch pa drugi itd...
                                let col_matrix_row_idx = number_of_rows_to_skip * batches * number_of_receptive_fields;

                                let col_idx = col_matrix_row_idx + col_matrix_col_idx;
                                col_data[col_idx] = value;
                            }
                        }
                    }
                }
            }
        }
        let result = Tensor::new(col_data, vec![receptive_field, batches * number_of_receptive_fields], true,vec![self.0.clone()], TensorOp::Im2Col);
        
        let result_copy = result.0.clone();
        let self_copy = self.0.clone();

        let (kernel_h_backwards, kernel_w_backwards, stride_backwards, padding_backwards) = (kernel_h, kernel_w, stride, padding);
        let (batches_backwards, channels_backwards, height_backwards, width_backwards) = (batches, channels, height, width);
        let (out_h_backwards, out_w_backwards) = (out_height, out_width);

        let (number_of_receptive_fields_backwards) = (number_of_receptive_fields);

        result.0.borrow_mut().backward = Some(Box::new(move || {
            let result_grad = result_copy.borrow();
            let mut self_borrow = self_copy.borrow_mut();

            for b in 0..batches_backwards{
            let base_data_idx = b * channels_backwards * height_backwards * width_backwards;
                for o_h in 0..out_h_backwards{
                    for o_w in 0..out_w_backwards{

                        for c in 0..channels_backwards{

                            let current_channel_idx = c * height_backwards * width_backwards;

                            let col_matrix_col_idx = b * number_of_receptive_fields_backwards + o_h * out_w_backwards + o_w;
                        
                            for kh in 0..kernel_h_backwards{
                                for kw in 0..kernel_w_backwards{
                                    let input_idx_x = o_w * stride_backwards + kw;
                                    let input_idx_y = o_h * stride_backwards + kh;

                                    //let value;

                                    let number_of_rows_to_skip = (kh * kernel_w_backwards + kw) + (c * kernel_w_backwards * kernel_h_backwards);
                                
                                    // prakticno [363 x 3025] number_of_receptive_fields je 3025 a 363 mi je filter_x * filter_y * channels, batchevi se stackuju takodje po koloni
                                    //znaci prvo odradim prvi batch pa drugi itd...
                                    let col_matrix_row_idx = number_of_rows_to_skip * batches_backwards * number_of_receptive_fields_backwards;

                                    let col_idx = col_matrix_row_idx + col_matrix_col_idx;

                                    // ovde mi je dosta pomogao onaj stanford cs231n vizuelizacija konvolucije
                                    if input_idx_x >= padding_backwards && input_idx_x - padding_backwards < width_backwards && input_idx_y >= padding_backwards  && input_idx_y - padding_backwards < height_backwards{
                                        let real_idx_x = input_idx_x - padding_backwards;
                                        let real_idx_y = input_idx_y - padding_backwards;
                                        //value = self_borrow.data[base_data_idx + current_channel_idx + real_idx_y * width + real_idx_x];
                                        let in_idx = base_data_idx + current_channel_idx + real_idx_y * width_backwards + real_idx_x;

                                        self_borrow.grad[in_idx] += result_grad.grad[col_idx];

                                    }
                                    //col_data[col_idx] = value;
                                }
                            }
                        }
                    }
                }
            }
        }));
        
        result
    }

    pub fn conv2d(&self, kernel: &Tensor, stride: usize, padding: usize, bias: Option<&Tensor>) -> Tensor{

        let shape = self.shape();

        let (batches, channels, height, width) = (shape[0], shape[1], shape[2], shape[3]);

        let kernel_shape = kernel.shape();

        let (number_of_filters, channels, kernel_height, kernel_width) = (kernel_shape[0], kernel_shape[1], kernel_shape[2], kernel_shape[3]);
        
        let out_height = (height + 2 * padding - kernel_height)/stride + 1;
        let out_width = (width + 2 * padding - kernel_width)/stride +1;
        
        let X_col = self.im2col(kernel_height, kernel_width, stride, padding);

        let W_row = kernel.reshape(vec![number_of_filters, channels * kernel_height * kernel_width]);

        let conv2d_result = W_row.matmul(&X_col);

        let mut reshaped_conv2d_result = conv2d_result.reshape(vec![batches, number_of_filters, out_height, out_width]);

        //clone u bias tensoru klonira RC, tako da clone pointuje na isti TensorCore sto znaci da ce bias biti dobar
        if let Some(bias_tensor) = bias{
            reshaped_conv2d_result = reshaped_conv2d_result + bias_tensor.reshape(vec![1, number_of_filters, 1, 1]);
        } 
        
        reshaped_conv2d_result
    }
}

impl Add for Tensor{
    type Output = Self;

	fn add(self, other: Self) -> Self{
        // da ne bih u foru sve vreme borrow pozivao...
        let self_borrow = self.0.borrow();
        let other_borrow = other.0.borrow();

        let self_data = &self_borrow.data;
        let other_data = &other_borrow.data;

        let a_shape = &self_borrow.shape;
        let b_shape = &other_borrow.shape;


        //result shape samo izmnozen - prakticno length(broj svih elemenata nekog tensora)
        let mut result_len = 1;
        let mut result_shape;
        match Self::util_calculate_broadcast_shapes(&a_shape, &b_shape){
            Ok(result) => {
                result_shape = result;
                for shape in &result_shape{
                    result_len *= shape;
                }
            },
            Err(err) => panic!("{:?}", err)
        }
        let mut result_data = vec![0.0; result_len];

        // ovde zelim da dobijem indeks prvo pozicije i u result_shape-u
        // pa onda posle toga trazim
        if a_shape == b_shape{
            for i in 0..result_len{
                result_data[i] = self_data[i] + other_data[i];
            }
        }else{
            let mut result_idx_buff = vec![0usize; result_shape.len()];
            for i in 0..result_len{
                Self::index_1d_to_nd_inplace(i, &result_shape, &mut result_idx_buff);

                let a_idx = Self::util_get_broadcast_idx_value(a_shape, &result_shape, &result_idx_buff);
                let b_idx = Self::util_get_broadcast_idx_value(b_shape, &result_shape, &result_idx_buff);

                result_data[i] = self_data[a_idx] + other_data[b_idx];

                //result_data[i] = self.0.borrow().data[i] + other.0.borrow().data[i];
            }
        }

        let result = Tensor::new(result_data, result_shape.clone(), true, vec![self.0.clone(), other.0.clone()], TensorOp::Add);


        // ovo mora jer ce kasnije ovaj backwards biti pozvan kada ne budu vise ove varijable postojale
        let result_copy = result.0.clone();
        let result_shape_copy = result_shape.clone();
        let self_copy = self.0.clone();
        let other_copy = other.0.clone();
        let a_shape_copy = a_shape.to_vec();
        let b_shape_copy = b_shape.to_vec();

        // a je prvi clan b je drugi clan c = a + b; c = self + other
        result.0.borrow_mut().backward = Some(Box::new(move || {
            {
                let result_borrow = result_copy.borrow();
                let mut self_grad = self_copy.borrow_mut();

                let mut result_idx_buff = vec![0usize; result_shape_copy.len()];

                for i in 0..result_borrow.grad.len(){
                    Self::index_1d_to_nd_inplace(i, &result_shape_copy, &mut result_idx_buff);

                    let a_idx = Self::util_get_broadcast_idx_value(&a_shape_copy, &result_shape_copy, &result_idx_buff);

                    self_grad.grad[a_idx] += result_borrow.grad[i]; //grad outputova uvek ce biti vise jer se broadcastuje
                }
            };
            {
                let result_borrow = result_copy.borrow();
                let mut other_grad = other_copy.borrow_mut();

                let mut result_idx_buff = vec![0usize; result_shape_copy.len()];

                for i in 0..result_borrow.grad.len(){
                    Self::index_1d_to_nd_inplace(i, &result_shape_copy, &mut result_idx_buff);

                    let b_idx = Self::util_get_broadcast_idx_value(&b_shape_copy, &result_shape_copy, &result_idx_buff);

                    other_grad.grad[b_idx] += result_borrow.grad[i];
                }
            }
        }));

        result

    }
}



// ovo je obican mul znaci element sa elementom ( NE MATMUL)
impl Mul for Tensor{
    type Output = Self;

    fn mul(self, other: Self) -> Self{
        //isto kao u Add-u
        let self_borrow = self.0.borrow();
        let other_borrow = other.0.borrow();

        let self_data = &self_borrow.data;
        let other_data = &other_borrow.data;

        let a_shape = &self_borrow.shape;
        let b_shape = &other_borrow.shape;

        let mut result_len = 1;
        let result_shape;
        match Self::util_calculate_broadcast_shapes(&a_shape, &b_shape){
            Ok(result) => {
                result_shape = result;
                for shape in &result_shape{
            result_len *= shape; 
                }
            },
            Err(err) => panic!("{:?}", err)
        }

        let mut result_data = vec![0.0; result_len];

        if a_shape == b_shape{
            for i in 0..result_len{
                result_data[i] = self_data[i] * other_data[i];
            }
        }else{
            let mut result_idx_buff = vec![0usize; result_shape.len()];

            for i in 0..result_len{
                Self::index_1d_to_nd_inplace(i, &result_shape, &mut result_idx_buff);

                let a_idx = Self::util_get_broadcast_idx_value(&a_shape, &result_shape, &result_idx_buff);
                let b_idx = Self::util_get_broadcast_idx_value(&b_shape, &result_shape, &result_idx_buff);

                result_data[i] = self_data[a_idx] * other_data[b_idx];
            }
        }

        let result = Tensor::new(result_data, result_shape.clone(), true, vec![self.0.clone(), other.0.clone()], TensorOp::Mul);


        // ovo mora jer ce kasnije ovaj backwards biti pozvan kada ne budu vise ove varijable postojale
        let result_copy = result.0.clone();
        let result_shape_copy = result_shape.clone();
        let self_copy = self.0.clone();
        let other_copy = other.0.clone();
        let a_shape_copy = a_shape.to_vec();
        let b_shape_copy = b_shape.to_vec();

        let is_same_operand = Rc::ptr_eq(&self_copy, &other_copy);

        // a je prvi clan b je drugi clan c = a + b; c = self + other
        result.0.borrow_mut().backward = Some(Box::new(move || {
            {
                if is_same_operand{
                    // ako si isti operandi onda ne smem self i other da imam isti borrow mut i immutable za istu stvar
                    let grad_output = result_copy.borrow();
                    //let a_data = &self_copy.borrow().data;
                    let b_data_vec = other_copy.borrow().data.clone();
                    let mut self_grad = self_copy.borrow_mut();
                    //let mut other_grad = other_copy.borrow_mut();


                    let mut result_idx_buff = vec![0usize; result_shape_copy.len()];
    
                    for i in 0..grad_output.grad.len(){
                        Self::index_1d_to_nd_inplace(i, &result_shape_copy, &mut result_idx_buff);
    
                        let a_idx = Self::util_get_broadcast_idx_value(&a_shape_copy, &result_shape_copy, &result_idx_buff);
                        let b_idx = Self::util_get_broadcast_idx_value(&b_shape_copy, &result_shape_copy, &result_idx_buff);
    
                        self_grad.grad[a_idx] += grad_output.grad[i] * b_data_vec[b_idx];
                    }
                }else{
                    let grad_output = result_copy.borrow();
                    //let a_data = &self_copy.borrow().data;
                    let b_data = other_copy.borrow();
                    let mut self_grad = self_copy.borrow_mut();
                    //let mut other_grad = other_copy.borrow_mut();

                    let mut result_idx_buff = vec![0usize; result_shape_copy.len()];
    
                    for i in 0..grad_output.grad.len(){
                        Self::index_1d_to_nd_inplace(i, &result_shape_copy, &mut result_idx_buff);
    
                        let a_idx = Self::util_get_broadcast_idx_value(&a_shape_copy, &result_shape_copy, &result_idx_buff);
                        let b_idx = Self::util_get_broadcast_idx_value(&b_shape_copy, &result_shape_copy, &result_idx_buff);
    
                        self_grad.grad[a_idx] += grad_output.grad[i] * b_data.data[b_idx];
                    }
                }
            };
            {
                if is_same_operand{
                    let grad_output = result_copy.borrow();
                    let a_data_vec = self_copy.borrow().data.clone();
                    let mut other_grad = other_copy.borrow_mut();

                    let mut result_idx_buff = vec![0usize; result_shape_copy.len()];

                    for i in 0..grad_output.grad.len(){
                        Self::index_1d_to_nd_inplace(i, &result_shape_copy, &mut result_idx_buff);
    
                        let a_idx = Self::util_get_broadcast_idx_value(&a_shape_copy, &result_shape_copy, &result_idx_buff);
                        let b_idx = Self::util_get_broadcast_idx_value(&b_shape_copy, &result_shape_copy, &result_idx_buff);
    
                        other_grad.grad[b_idx] += grad_output.grad[i] * a_data_vec[a_idx];
                    }
                }else{
                    let grad_output = result_copy.borrow();
                    let a_data = self_copy.borrow();
                    let mut other_grad = other_copy.borrow_mut();

                    let mut result_idx_buff = vec![0usize; result_shape_copy.len()];

                    for i in 0..grad_output.grad.len(){
                        Self::index_1d_to_nd_inplace(i, &result_shape_copy, &mut result_idx_buff);
    
                        let a_idx = Self::util_get_broadcast_idx_value(&a_shape_copy, &result_shape_copy, &result_idx_buff);
                        let b_idx = Self::util_get_broadcast_idx_value(&b_shape_copy, &result_shape_copy, &result_idx_buff);
    
                        other_grad.grad[b_idx] += grad_output.grad[i] * a_data.data[a_idx];
                    }
                }
            }
        }));
        result
    }
}

#[cfg(test)]
mod tests {
    include!("tensor_tests.rs");
}