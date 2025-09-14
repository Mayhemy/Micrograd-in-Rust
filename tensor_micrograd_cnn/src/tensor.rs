use std::rc::Rc;
use std::cell::RefCell;
use std::hash::{Hash, Hasher};
use std::collections::HashSet;
use std::usize::MAX;
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
    
    pub fn shape(&self) -> Vec<usize>{
        self.0.borrow().shape.clone()
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
        let self_shape = &self.0.borrow().shape;
        let other_shape = &other.0.borrow().shape;
        match (self_shape.len(), other_shape.len()) {
            (2,2) => self.matmul_2d(&other),
            (3,3) => self.matmul_batched_3d(&other),
            (3,2) => self.matmul_3d_and_2d(&other),
            _ => panic!(),
        }
    }

    fn matmul_2d(&self, other: &Tensor) -> Tensor{
        let shape_a = &self.0.borrow().shape;
        let shape_b = &other.0.borrow().shape;

        assert_eq!(shape_a[1], shape_b[0]);

        let (m, n) = (shape_a[0], shape_a[1]);
        let (x, y)   = (shape_b[0], shape_b[1]);

        let a_data = &self.0.borrow().data;
        let b_data = &other.0.borrow().data;

        let mut result_data = vec![0.0; m*y];

        for i in 0..m{
            for j in 0..y{
                let mut element = 0.0;
                //mogu i n i x
                for k in 0..n{
                    element += a_data[i * n + k] * b_data[y * k + j]
                }
                result_data[i * y + j] = element;
            }
        }

        let result = Tensor::new(result_data, vec![m, y], true, vec![self.0.clone(), other.0.clone()], TensorOp::MatMul);

        //backwards sad
        // ovo mora jer ce kasnije ovaj backwards biti pozvan kada ne budu vise ove varijable postojale
        let result_copy = result.0.clone();
        let self_copy = self.0.clone();
        let other_copy = other.0.clone();

        // a je prvi clan b je drugi clan c = a + b; c = self + other
        result.0.borrow_mut().backward = Some(Box::new(move || {
            let grad_result = &result_copy.borrow().grad;
            let a_data = &self_copy.borrow().data;
            let b_data = &other_copy.borrow().data;


            // n == x
            // posto je G * B^T => G je shape-a m,y (shape_a[0], shape_b[1]) a B^T je (shape_b[1],shape_b[0)) y,x - iteriramo po m i x tj. po A jer racunamo izvod za A
            // posto mi ne transponujemo matricu direktno to znaci da pri indeksiranju moramo da je "transponujemo" pa onda mul i transpose zajedno daju efekat
            // da mnozimo red sa redom.
            for i in 0..m{
                for j in 0..n{
                    let mut element = 0.0;
                    for k in 0..y{
                        element += grad_result[i * y + k] * b_data[j * y + k];
                    }
                    self_copy.borrow_mut().grad[i * n + j] += element;
                }
            }

            // n == x ne zaboraviti

            //m n , x y
            for i in 0..x{
                for j in 0..y{
                    let mut element = 0.0;
                    for k in 0..n{
                        element += a_data[k * n + i] * grad_result[k * y + j];
                    }
                    other_copy.borrow_mut().grad[i * y + j] += element;
                }
            }
        }));

        result
    }

    fn matmul_batched_3d(&self, other: &Tensor) -> Tensor{
        let shape_a = &self.0.borrow().shape;
        let shape_b = &other.0.borrow().shape;

        assert_eq!(shape_a[0], shape_b[0]);
        assert_eq!(shape_a[2], shape_b[1]);

        let (batch, m, n ) = (shape_a[0], shape_a[1], shape_a[2]);
        let (_, n, y)   = (shape_b[0], shape_b[1], shape_b[2]);

        let a_data = &self.0.borrow().data;
        let b_data = &other.0.borrow().data;

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

        let a_data = &self.0.borrow().data;
        let b_data = &other.0.borrow().data;

        let mut result_data = vec![0.0; batch * m*y];

        for b in 0..batch{
            let a_offset = b * (m * n);
            let c_offset = b * (m * y);
            for i in 0..m{
                for j in 0..y{
                    let mut sum = 0.0;
                    for k in 0..n{
                        let a_idx = a_offset +  i * n + k;
                        let b_idx = k * y + j;
                        sum += a_data[a_idx] * b_data[b_idx];
                    }
                    result_data[c_offset + i * y + j] = sum;
                }
            }
        }

        let result = Tensor::new(result_data, vec![batch, m, y], true, vec![self.0.clone(), other.0.clone()], TensorOp::MatMul);

        //backwards sad
        // ovo mora jer ce kasnije ovaj backwards biti pozvan kada ne budu vise ove varijable postojale
        let result_copy = result.0.clone();
        let self_copy = self.0.clone();
        let other_copy = other.0.clone();

        // a je prvi clan b je drugi clan c = a + b; c = self + other
        result.0.borrow_mut().backward = Some(Box::new(move || {
            let grad_result = &result_copy.borrow().grad;
            let a_data = &self_copy.borrow().data;
            let b_data = &other_copy.borrow().data;


            // n == x
            // Sada radimo [batch, m, n] i [x, y], znaci kada racunamo gA = G(b,m,y) i B^T(y, n) ali trenutni je B(n, y), za svaki batch - prakticno posto je mnozenje
            // i T onda kao da mnozim red iz prve i red iz druge matrice kad se sve to sredi
            for b in 0..batch{
                for i in 0..m{
                    for j in 0..n{
                        let mut element = 0.0;
                        for k in 0..y{
                            let result_idx = b * (m * y) + i * y + k;
                            let b_idx = j * y + k;
                            element += grad_result[result_idx] * b_data[b_idx];
                        }
                        self_copy.borrow_mut().grad[b * m * n + i * n + j] += element;
                    }
                }
            }

            // ovde moram da akumuliram preko batcheva a svaki batch je transponovan (mn - ny)
            for i in 0..n{
                for j in 0..y{
                    let mut element = 0.0;
                    for b in 0..batch{
                        for k in 0..m{
                            let a_idx = b * (m * n) + k * n + i;
                            let result_idx = b * (m * y) + k * y + j;
                            element += a_data[a_idx] * grad_result[result_idx];
                        }
                    }
                    other_copy.borrow_mut().grad[i * y + j] += element;
                }
            }
        }));

        result
    }

    pub fn relu(&self) -> Tensor{
        let data = &self.0.borrow().data;
        let shape = self.0.borrow().shape.clone();
        
        let mut result_data = vec![0.0; data.len()];

        for i in 0..data.len(){
            result_data[i] = data[i].max(0.0);
        }

        let result = Tensor::new(result_data, shape,true,vec![self.0.clone()], TensorOp::ReLU);

        let result_copy = result.0.clone();
        let self_copy = self.0.clone();

        result.0.borrow_mut().backward = Some(Box::new(move || {

            let result_grad = &result_copy.borrow().grad;
            let input_data = &self_copy.borrow().data;
            
            for i in 0..result_grad.len(){
                if input_data[i] > 0.0{
                    self_copy.borrow_mut().grad[i] += result_grad[i];
                }
            }
        }));
        result
    }

    pub fn maxpool_2d(&self, kernel_size: usize, stride: usize) -> Tensor{
        let data = &self.0.borrow().data;
        let shape = &self.0.borrow().shape;

        assert_eq!(shape.len(), 4, "Maxpool needs 4d input.");

        let (batch, channels, height, width) = (shape[0], shape[1], shape[2], shape[3]);


    }
}

impl Add for Tensor{
    type Output = Self;

	fn add(self, other: Self) -> Self{
        // da ne bih u foru sve vreme borrow pozivao...
        let self_data = &self.0.borrow().data;
        let other_data = &other.0.borrow().data;

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
        if a_shape == b_shape{
            for i in 0..result_len{
                result_data[i] = self_data[i] + other_data[i];
            }
        }else{
            for i in 0..result_len{
                let result_idx = Self::index_1d_to_nd(i, &result_shape);

                let a_idx = Self::util_get_broadcast_idx_value(a_shape, &result_shape, &result_idx);
                let b_idx = Self::util_get_broadcast_idx_value(b_shape, &result_shape, &result_idx);

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
        //isto kao u Add-u
        let self_data = &self.0.borrow().data;
        let other_data = &other.0.borrow().data;

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

        if a_shape == b_shape{
            for i in 0..result_len{
                result_data[i] = self_data[i] * other_data[i];
            }
        }else{
            for i in 0..result_len{
                let result_idx = Self::index_1d_to_nd(i, &result_shape);

                let a_idx = Self::util_get_broadcast_idx_value(&a_shape, &result_shape, &result_idx);
                let b_idx = Self::util_get_broadcast_idx_value(&b_shape, &result_shape, &result_idx);

                result_data[i] = self_data[a_idx] * other_data[b_idx];
            }
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