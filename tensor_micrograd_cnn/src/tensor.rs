use std::rc::Rc;
use std::cell::RefCell;
use std::hash::{Hash, Hasher};
use std::collections::HashSet;
use rand::Rng;
use std::cmp;

#[derive(Debug, Clone)]
enum TensorOp{
	Reshape,
    View,
    Add,
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
    let b = Tensor::ones(vec![1, 3]);
    let c = a + b;
    println!("Broadcasting result: {:?}", c);





        Tensor::new(data, shape, requires_grad, children, op)
    }

    pub fn index_1d(&self, indices: Vec<usize>) -> usize{
        // indeksi ce nam biti u obliku (1,0,1,1) a shape je (2,2,2,2), krecemo od kraja i trazimo 1d index
        let tensor_borrowed = self.0.borrow();
        let shape = &tensor_borrowed.shape;
        let mut index_1d = 0;
        let mut multiplier = 1;
        for i in (0..indices.len()).rev(){
            index_1d += indices[i] * multiplier;
            multiplier *= shape[i]; // ovo radimo jer i za indices i za shape idemo od kraja
        }
        index_1d
    }

    // inverzna operacija od index_1d (od 1d indeksa pravimo n-dimenzioni indeks)
    pub fn index_1d_to_nd(&self, index_1d: usize) -> Vec<usize>{
        let tensor_borrowed = self.0.borrow();
        let shape = &tensor_borrowed.shape;
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
        }

        Ok(broadcast_shape);
    }

    //lose, implementiraj opet kad razumes glupane
    pub fn get_broadcast_idx_value(original_shape: &[usize], broadcast_shape: &[usize]) -> usize{
        let difference_in_shape_dimensions = broadcast_shape.len() - original_shape.len();
        let mut original_indices = vec![0.0, original_shape.len()];
        for i in 0..original_shape.len(){
            let broadcast_shape_index = i + difference_in_shape_dimensions;

            if original_shape[i] == 1 && broadcasted_shape[broadcast_shape_index] > 1{
                original_indices[i]
            }
        }
    }

}

#[cfg(test)]
mod tests {
    include!("tensor_tests.rs");
}