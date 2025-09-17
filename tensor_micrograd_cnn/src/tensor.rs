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

    pub fn randn_uniform(shape: Vec<usize>) -> Tensor{
        let mut flattened_size = 1;
        for dimension in &shape{
            flattened_size *= dimension;
        }
        let mut rng = rand::thread_rng();
        let mut data = Vec::with_capacity(flattened_size);
        for i in 0..flattened_size{
            data.push(rng.gen_range(-1.0 as Precision..1.0 as Precision));
        }
        let requires_grad = false;
        let children = vec![];
        let op = TensorOp::None;

        Tensor::new(data, shape, requires_grad, children, op)
    }

    pub fn zero_grad(&self){
        self.0.borrow_mut().grad.fill(0.0 as Precision);
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

        for i in 0..m{
            let a_row = i * n;
            let result_row = i * y;
            for j in 0..y{
                let mut element = 0.0;
                //mogu i n i x
                for k in 0..n{
                    element += a_data[a_row + k] * b_data[y * k + j]
                }
                result_data[result_row + j] = element;
            }
        }

        let result = Tensor::new(result_data, vec![m, y], true, vec![self.0.clone(), other.0.clone()], TensorOp::MatMul);

        //backwards sad
        // ovo mora jer ce kasnije ovaj backwards biti pozvan kada ne budu vise ove varijable postojale
        let result_copy = result.0.clone();
        let self_copy = self.0.clone();
        let other_copy = other.0.clone();


        let is_same_operand = Rc::ptr_eq(&self_copy, &other_copy);

        // a je prvi clan b je drugi clan c = a + b; c = self + other
        result.0.borrow_mut().backward = Some(Box::new(move || {
        {
            if is_same_operand{
                let grad_result = result_copy.borrow();
                //let a_data = &self_copy.borrow().data;
                let b_data_vec = other_copy.borrow().data.clone();
                let mut self_grad = self_copy.borrow_mut();
                //let mut other_grad = other_copy.borrow_mut();
    
    
                // n == x
                // posto je G * B^T => G je shape-a m,y (shape_a[0], shape_b[1]) a B^T je (shape_b[1],shape_b[0)) y,x - iteriramo po m i x tj. po A jer racunamo izvod za A
                // posto mi ne transponujemo matricu direktno to znaci da pri indeksiranju moramo da je "transponujemo" pa onda mul i transpose zajedno daju efekat
                // da mnozimo red sa redom.
                for i in 0..m{
                    let result_row = i * y;
                    let a_row =  i * n;
                    for j in 0..n{
                        let b_col = j * y;
                        let mut element = 0.0;
                        for k in 0..y{
                            element += grad_result.grad[result_row + k] * b_data_vec[b_col + k];
                        }
                        self_grad.grad[a_row + j] += element;
                    }
                }
            }else{
                let grad_result = result_copy.borrow();
                //let a_data = &self_copy.borrow().data;
                let b_data = other_copy.borrow();
                let mut self_grad = self_copy.borrow_mut();
                //let mut other_grad = other_copy.borrow_mut();


                // n == x
                // posto je G * B^T => G je shape-a m,y (shape_a[0], shape_b[1]) a B^T je (shape_b[1],shape_b[0)) y,x - iteriramo po m i x tj. po A jer racunamo izvod za A
                // posto mi ne transponujemo matricu direktno to znaci da pri indeksiranju moramo da je "transponujemo" pa onda mul i transpose zajedno daju efekat
                // da mnozimo red sa redom.
                for i in 0..m{
                    let result_row = i * y;
                    let a_row =  i * n;
                    for j in 0..n{
                        let b_col = j * y;
                        let mut element = 0.0;
                        for k in 0..y{
                            element += grad_result.grad[result_row + k] * b_data.data[b_col + k];
                        }
                        self_grad.grad[a_row + j] += element;
                    }
                }
            }
        };
        {
            if is_same_operand{
                let grad_result = result_copy.borrow();
                let a_data_vec = self_copy.borrow().data.clone();
                let mut other_grad = other_copy.borrow_mut();
                // n == x ne zaboraviti
    
                //m n , x y
                for i in 0..x{
                    let b_row = i * y;
                    for j in 0..y{
                        let mut element = 0.0;
                        for k in 0..m{
                            element += a_data_vec[k * n + i] * grad_result.grad[k * y + j];
                        }
                        other_grad.grad[b_row + j] += element;
                    }
                }
            }else{
                let grad_result = result_copy.borrow();
                let a_data = self_copy.borrow();
                let mut other_grad = other_copy.borrow_mut();
                // n == x ne zaboraviti
    
                //m n , x y
                for i in 0..x{
                    let b_row = i * y;
                    for j in 0..y{
                        let mut element = 0.0;
                        for k in 0..m{
                            element += a_data.data[k * n + i] * grad_result.grad[k * y + j];
                        }
                        other_grad.grad[b_row + j] += element;
                    }
                }
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