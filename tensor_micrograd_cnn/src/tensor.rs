use std::rc::Rc;
use std::cell::RefCell;
use std::hash::{Hash, Hasher};
use std::collections::HashSet;
use rand::Rng;
use std::cmp;
use std::ops::{Add, Mul};

#[derive(Debug, Clone)]
enum TensorOp{
	Reshape,
    View,
    Add,
    Mul,
    MatMul,
    ReLU,
    Conv2d,
    MaxPool2d,
    //Sum,
    //Mean,
    None
}

pub struct TensorCore{
    data: Vec<f64>, // xi
    shape: Vec<usize>, // bice nam korisno
    requires_grad: bool, // 
    grad: Vec<f64>,
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
    fn new(data :Vec<f64>, shape: Vec<usize>, requires_grad: bool, children: Vec<Rc<RefCell<TensorCore>>>, op: TensorOp) -> Tensor{
        // zbog rustovog nasledjivanja ownershipa ovo mora
        let data_len = data.len();

        let tensor = TensorCore{
            data,
            shape: shape,
            requires_grad,
            grad: vec![0.0; data_len],
            backward: None,
            op,
            prev: children,
        };
        Tensor(Rc::new(RefCell::new(tensor)))
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

    pub fn randn(shape: Vec<usize>) -> Tensor{
        let mut flattened_size = 1;
        for dimension in &shape{
            flattened_size *= dimension;
        }
        let mut rng = rand::thread_rng();
        let data = vec![rng.gen_range(-1.0..1.0); flattened_size];
        let requires_grad = false;
        let children = vec![];
        let op = TensorOp::None;

        Tensor::new(data, shape, requires_grad, children, op)
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

    // inverzna operacija od index_1d (od 1d indeksa pravimo n-dimenzioni indeks)
    pub fn index_1d_to_nd(index_1d: usize, shape: &[usize]) -> Vec<usize>{
        let mut indices = vec![0; shape.len()];
        let mut remainder = index_1d;
        for i in (0..shape.len()).rev(){
            indices[i] = remainder % shape[i];
            remainder /= shape[i];
        }
        indices
    }

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
            let mut curr_a = 0;
            let mut curr_b = 0;

            if i < shape_a.len(){
                curr_a = shape_a[shape_a.len() - i - 1]
            }

            if i < shape_b.len(){
                curr_b = shape_b[shape_b.len() - i - 1]
            }

            let mut result_dimension = 0;

            if curr_a == curr_b{
                result_dimension = curr_a;
            }else{
                if curr_a == 1{
                    result_dimension = curr_b;
                }else if curr_b == 1{
                    result_dimension = curr_a;
                }else{
                    return Err(format!("Operands could not be broadcast together: shape_a[{}] = {} != shape_b[{}] = {}", i, shape_a[i], i, shape_b[i]));
                }
            }
            broadcast_shape[i] = result_dimension;
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
}

impl Add for Tensor{
    type Output = Self;

	fn add(self, other: Self) -> Self{
        let a_shape = &self.0.borrow().shape;
        let b_shape = &other.0.borrow().shape;


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
        for i in 0..result_len{
            let result_idx = Self::index_1d_to_nd(i, &result_shape);

            let a_idx = Self::util_get_broadcast_idx_value(a_shape, &result_shape, &result_idx);
            let b_idx = Self::util_get_broadcast_idx_value(b_shape, &result_shape, &result_idx);

            result_data[i] = self.0.borrow().data[a_idx] + other.0.borrow().data[b_idx];

            //result_data[i] = self.0.borrow().data[i] + other.0.borrow().data[i];
        }

        let result = Tensor::new(result_data, result_shape.clone(), true, vec![self.0.clone(), other.0.clone()], TensorOp::Add);


        // ovo mora jer ce kasnije ovaj backwards biti pozvan kada ne budu vise ove varijable postojale
        let result_copy = result.0.clone();
        let result_shape_copy = result_shape.clone();
        let self_copy = self.0.clone();
        let other_copy = other.0.clone();
        let a_shape_copy = a_shape.clone();
        let b_shape_copy = b_shape.clone();

        // a je prvi clan b je drugi clan c = a + b; c = self + other
        result.0.borrow_mut().backward = Some(Box::new(move || {
            let grad_output = &result_copy.borrow().grad;

            for i in 0..grad_output.len(){
                let result_idx = Self::index_1d_to_nd(i, &result_shape_copy);

                let a_idx = Self::util_get_broadcast_idx_value(&a_shape_copy, &result_shape_copy, &result_idx);
                let b_idx = Self::util_get_broadcast_idx_value(&b_shape_copy, &result_shape_copy, &result_idx);

                self_copy.borrow_mut().grad[a_idx] += grad_output[i]; //grad outputova uvek ce biti vise jer se broadcastuje
                other_copy.borrow_mut().grad[b_idx] += grad_output[i];
            }
        }));

        result

    }
}



// ovo je obican mul znaci element sa elementom ( NE MATMUL)
impl Mul for Tensor{
    type Output = Self;

    fn mul(self, other: Self) -> Self{
        let a_shape = &self.0.borrow().shape;
        let b_shape = &other.0.borrow().shape;

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

        for i in 0..result_len{
            let result_idx = Self::index_1d_to_nd(i, &result_shape);

            let a_idx = Self::util_get_broadcast_idx_value(&a_shape, &result_shape, &result_idx);
            let b_idx = Self::util_get_broadcast_idx_value(&b_shape, &result_shape, &result_idx);

            result_data[i] = self.0.borrow().data[a_idx] * other.0.borrow().data[b_idx];
        }
        let result = Tensor::new(result_data, result_shape.clone(), true, vec![self.0.clone(), other.0.clone()], TensorOp::Mul);


        // ovo mora jer ce kasnije ovaj backwards biti pozvan kada ne budu vise ove varijable postojale
        let result_copy = result.0.clone();
        let result_shape_copy = result_shape.clone();
        let self_copy = self.0.clone();
        let other_copy = other.0.clone();
        let a_shape_copy = a_shape.clone();
        let b_shape_copy = b_shape.clone();

        // a je prvi clan b je drugi clan c = a + b; c = self + other
        result.0.borrow_mut().backward = Some(Box::new(move || {
            let grad_output = &result_copy.borrow().grad;
            let a_data = &self_copy.borrow().data;
            let b_data = &other_copy.borrow().data;

            for i in 0..grad_output.len(){
                let result_indices = Self::index_1d_to_nd(i, &result_shape_copy);

                let a_idx = Self::util_get_broadcast_idx_value(&a_shape_copy, &result_shape_copy, &result_indices);
                let b_idx = Self::util_get_broadcast_idx_value(&b_shape_copy, &result_shape_copy, &result_indices);

                self_copy.borrow_mut().grad[a_idx] += grad_output[i] * b_data[b_idx];
                other_copy.borrow_mut().grad[b_idx] += grad_output[i] * a_data[a_idx];
            }
        }));
        result
    }
}

#[cfg(test)]
mod tests {
    include!("tensor_tests.rs");
}