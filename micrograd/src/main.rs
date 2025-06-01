use std::rc::Rc;
use std::cell::RefCell;
use std::ops::{Add, Mul};

#[derive(Debug, Clone)]
enum Op{
    Add,
    Mul,
    Tanh,
    Neg,
    Pow,
    Powf,
    None
}
struct Value{
    data: f64,
    grad: f64,
    backward: Option<Box<dyn Fn()>>,
    op: Op,
    prev: Vec<Rc<RefCell<Value>>>,
}

// Need to wrap my value so that i can implement Add and Mul since Rc<RefCell<Value>> is external and Add and mul are external??
// no idea why this is an issue in rust but ok...
#[derive(Clone)]
struct ValueWrapper(Rc<RefCell<Value>>);

impl ValueWrapper{
    fn new(data: f64, children : Vec<Rc<RefCell<Value>>>, op: Op) -> Self {
        ValueWrapper(Rc::new(RefCell::new(Value{
            data,
            grad: 0.0,
            backward: None,
            op: op,
            prev: children,
        })))
    }
    fn minimal_new(data: f64) -> Self{
        ValueWrapper(Rc::new(RefCell::new(Value{
            data,
            grad: 0.0,
            backward: None,
            op: Op::None,
            prev: vec![],
        })))
    }
    
    fn tanh(&self) -> ValueWrapper{
        //unary operations can be done as a reference, but lets stick to Andrej's implementation
        let operation_result = self.0.borrow().data.tanh();
        
        let result = ValueWrapper::new(operation_result, vec![self.0.clone()], Op::Tanh);
        
        let result_reference = result.0.clone();
        let self_reference = self.0.clone();
        result.0.borrow_mut().backward = Some(Box::new(move || {
            let local_derivative = 1.0 - operation_result*operation_result;
            self_reference.borrow_mut().grad += result_reference.borrow().grad * local_derivative;
        }));
        result
    }
    
    fn powf(&self, value: f64) -> ValueWrapper {
        let operation_result = self.0.borrow().data.powf(value);
        
        let result = ValueWrapper::new(operation_result, vec![self.0.clone()], Op::Powf);
        
        let result_reference = result.0.clone();
        let self_reference = self.0.clone();
        
        result.0.borrow_mut().backward = Some(Box::new(move || {
           let local_derivative =  value * self_reference.borrow().data.powf(value - 1.0);
           self_reference.borrow_mut().grad += result_reference.borrow().grad * local_derivative;
        }));  
        result     
    }
    
    fn pow(&self, other: &ValueWrapper) -> ValueWrapper{
        let operation_result = self.0.borrow().data.powf(other.0.borrow().data);
        
        let result = ValueWrapper::new(operation_result, vec![self.0.clone(), other.0.clone()], Op::Pow);
        
        let result_reference = result.0.clone();
        let self_reference = self.0.clone();
        let other_reference = other.0.clone();
        
        result.0.borrow_mut().backward = Some(Box::new(move || {
            let grad_output = result_reference.borrow().data;
           
            if self_reference.borrow().data > 0.0 {
                let base_derivative = other_reference.borrow().data * self_reference.borrow().data.powf(other_reference.borrow().data - 1.0);
                self_reference.borrow_mut().grad += base_derivative * result_reference.borrow().grad; 
            }
            
            if self_reference.borrow().data > 0.0 {
                let exponent_derivative = grad_output * self_reference.borrow().data.ln();
                other_reference.borrow_mut().grad += exponent_derivative * result_reference.borrow().grad;
            }
        }));  
        result  
    }
}


impl Add for ValueWrapper{
    type Output = Self;
    
    fn add(self, other: Self) -> Self{
        let left_val = self.0.clone();
        let right_val = other.0.clone();
        let operation_result = left_val.borrow().data + right_val.borrow().data;
        
        let result = ValueWrapper::new(operation_result, vec![left_val.clone(),right_val.clone()], Op::Add);
        
        let result_reference = result.0.clone();
        result.0.borrow_mut().backward = Some(Box::new(move || {
            left_val.borrow_mut().grad += 1.0 * result_reference.borrow().grad;
            right_val.borrow_mut().grad += 1.0 * result_reference.borrow().grad;
        }));
        
        result
    }
}
impl Mul for ValueWrapper{
    type Output = Self;
    
    fn mul(self, other: Self) -> Self{
        let left_val = self.0.clone();
        let right_val = other.0.clone();
        let operation_result = left_val.borrow().data * right_val.borrow().data;
        
        let result = ValueWrapper::new(operation_result, vec![left_val.clone(),right_val.clone()], Op::Mul);
        
        let result_reference = result.0.clone();
        result.0.borrow_mut().backward = Some(Box::new(move || {
            left_val.borrow_mut().grad += right_val.borrow().data * result_reference.borrow().grad;
            right_val.borrow_mut().grad += left_val.borrow().data * result_reference.borrow().grad;
        }));
        
        result
    }      
}
 

fn main() {
    // Example usage
    let a = ValueWrapper::minimal_new(2.0);
    let b = ValueWrapper::minimal_new(-3.0);
    let c = ValueWrapper::minimal_new(10.0);
    
    let e = a.clone() + b.clone();
    let d = e.clone() + c.clone();
    
    let f = ValueWrapper::minimal_new(-2.0);
    let l = d.clone() * f.clone();
    
    // Set gradient for loss
    l.0.borrow_mut().grad = 1.0;
    
    // Manual backward pass (you'd need to implement topological sort)
    if let Some(ref backward_fn) = l.0.borrow().backward {
        backward_fn();
    }
    if let Some(ref backward_fn) = d.0.borrow().backward {
        backward_fn();
    }
    if let Some(ref backward_fn) = e.0.borrow().backward {
        backward_fn();
    }
    
    println!("a.grad: {}", a.0.borrow().grad);
    println!("b.grad: {}", b.0.borrow().grad);
    println!("c.grad: {}", c.0.borrow().grad);
    println!("f.grad: {}", f.0.borrow().grad);
}

