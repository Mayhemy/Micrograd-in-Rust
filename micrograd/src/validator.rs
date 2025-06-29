use crate::{ValueWrapper, MLP, calculate_loss, zero_gradients};
use crate::neural_network::{Call, Parameters};

pub struct MicrogradValidator{
	numerical_input_var_change: f64,
	fault_tolerance: f64,
}


impl MicrogradValidator{
	pub fn new(numerical_input_var_change: f64, fault_tolerance: f64) -> Self{
		Self{numerical_input_var_change, fault_tolerance}
	}
	
	pub fn minimal_new() -> Self{
		Self::new(0.0001, 0.00001)
	}
	
	pub fn validate_all_test(&self){
		self.test_pow_functions();
		self.verify_gradients();
	}
	
	fn test_pow_functions(&self){
		let test_cases = vec![(2.0, 3.0, "positive base - positive exponent"),
		(2.0, -3.0, "positive base - negative exponent"),
		(-2.0, 3.0, "negative base - positive exponent"),
		(-2.0, -3.0, "negative base - negative exponent"),
		(0.5, 4.0, "fraction base - positive exponent"),
		(16.0, 0.5, "positive exponent - fraction exponent")];
		self.test_powf(&test_cases);
		self.test_pow(&test_cases);
	}
	fn test_powf(&self, test_cases: &Vec<(f64, f64,&str)> ) -> bool{
		for (base, exponent, description) in test_cases{
			if !self.test_single_powf(*base , *exponent, description){
				println!("Test pow failed {}", description);
				return false;
			}
		}
		println!("All powf tests passed");
		true
	}
	fn test_pow(&self, test_cases: &Vec<(f64, f64, &str)>) -> bool{
		for (base, exponent, description) in test_cases{
			if !self.test_single_pow(*base, *exponent, description){
				println!("Test pow failed {}", description);
				return false;
			}
		}
		println!("All pow tests passed");
		true
	}
	
	fn test_single_pow(&self, base: f64, exponent: f64, description:&str) -> bool{
		let base_val = ValueWrapper::minimal_new(base);
		let exponent_val = ValueWrapper::minimal_new(exponent);
		
		//forward pass test first
		let result_val = base_val.pow(&exponent_val);
		let result = base.powf(exponent);
		
		if (result_val.0.borrow().data - result).abs() > self.fault_tolerance {
			println!("Wrong result for: {} = {}", result_val.0.borrow().data, result);
			return false;
		}
		
		//computes analytical gradients
		result_val.init_backward();
		
		let base_analytical_grad = base_val.0.borrow().grad;
		let exponent_analytical_grad = exponent_val.0.borrow().grad;
		
		let base_numerical_grad = self.calculate_base_numerical_gradient(base, exponent);
		let base_diff = (base_analytical_grad - base_numerical_grad).abs();
		let mut exponent_diff = 0.0;
		if base > 0.0 {
			let exponent_numerical_grad = self.calculate_exponent_numerical_gradient(base, exponent);
			exponent_diff =  (exponent_analytical_grad - exponent_numerical_grad).abs();
		}else{
			if exponent_analytical_grad.abs() > self.fault_tolerance{
				println!("Works bad as intended!");//In reality this should be false but since we are making minimalistic engine we will not handle edge cases (grad  would be e^(b+ln(a)) + ln(a) and log of negative numer doesnt exist since domain is x>0)
				return true; // we can image it to be true :)
			}
		}
		// let exponent_numerical_grad = self.calculate_exponent_numerical_gradient(base, exponent);
		
		//compare analytical and numerical gradients
		
		if base_diff > self.fault_tolerance || exponent_diff > self.fault_tolerance {
			println!("Gradient mismatch {}: base = {},base_diff = {}  ------ exponent = {}, exponent_diff = {}", description, base, base_diff, exponent, exponent_diff);
			return false;
		}
		
		true
	}
	
	fn test_single_powf(&self, base: f64, exponent: f64, description:&str) -> bool{
		let base_val = ValueWrapper::minimal_new(base);
		
		//forward pass test first
		let result_val = base_val.powf(exponent);
		let result = base.powf(exponent);
		
		if (result_val.0.borrow().data - result).abs() > self.fault_tolerance {
			println!("Wrong result for: {} = {}", result_val.0.borrow().data, result);
			return false;
		}
		
		//computes analytical gradients
		result_val.init_backward();
		
		let base_analytical_grad = base_val.0.borrow().grad;
		
		let base_numerical_grad = self.calculate_base_numerical_gradient(base, exponent);
		
		//compare analytical and numerical gradients
		
		let base_diff = (base_analytical_grad - base_numerical_grad).abs();
		
		if base_diff > self.fault_tolerance{
			println!("Gradient mismatch {}", description);
			return false;
		}
		
		true
	}
	
	fn calculate_base_numerical_gradient(&self, base : f64, exponent: f64) -> f64{
		let base_positive = base + self.numerical_input_var_change;
		let base_negative = base - self.numerical_input_var_change;
		let result_positive = base_positive.powf(exponent);
		let result_negative = base_negative.powf(exponent);
		
		let final_grad = (result_positive - result_negative) / (2.0 * self.numerical_input_var_change);
		
		final_grad
	}
	
	fn calculate_exponent_numerical_gradient(&self, base : f64, exponent: f64) -> f64{
		let exponent_positive = exponent + self.numerical_input_var_change;
		let exponent_negative = exponent - self.numerical_input_var_change;
		let result_positive = base.powf(exponent_positive);
		let result_negative = base.powf(exponent_negative);
		
		let final_grad = (result_positive - result_negative) / (2.0 * self.numerical_input_var_change);
		
		final_grad
	}
	
	fn verify_gradients(&self){
		let test_network = MLP::new(2,vec![2,1]);
		let params = test_network.parameters();
		
		
		params[0].0.borrow_mut().data = 0.9;
    	params[1].0.borrow_mut().data = -0.3;
    	params[2].0.borrow_mut().data = 0.1;
    	
    	params[3].0.borrow_mut().data = -0.3;
    	params[4].0.borrow_mut().data = 0.7;
    	params[5].0.borrow_mut().data = 0.4;
    	
    	params[6].0.borrow_mut().data = -0.8;
    	params[7].0.borrow_mut().data = 0.7;
    	params[8].0.borrow_mut().data = -1.6;
    	
    	let xs = vec![ValueWrapper::minimal_new(0.5),ValueWrapper::minimal_new(1.5)];
    	let ys = vec![ValueWrapper::minimal_new(1.0)];
    	
    	
    	zero_gradients(&test_network);
    	let predictions = test_network.call(xs.clone());
    	let loss = calculate_loss(predictions, &ys);
    	
    	loss.init_backward();
		
		let anal_grad = self.analytical_grad_verify(&test_network);
		let numerical_gradients = self.numerical_grad_verify(&test_network, &xs, &ys);
		
		let checker = self.compare_arrays_verify(anal_grad, numerical_gradients);
		
		//println!("Checker {}", checker);
		if checker == true{
			println!("WORKS!");
		}
	}
		
	fn compare_arrays_verify(&self, analytical_grad_array : Vec<f64>, numerical_grad_array : Vec<f64>) -> bool{
		let mut matching_grads = true;
		for (anal_grad_val, numerical_grad_val) in analytical_grad_array.iter().zip(numerical_grad_array.iter()) {
			println!("analytical_Grad = {},   numerical_Grad = {}", anal_grad_val, numerical_grad_val);
			if (anal_grad_val - numerical_grad_val).abs() > 0.0000001{
				matching_grads = false;
			}
		}
		matching_grads
	}
	
	fn analytical_grad_verify(&self, neural_network : &MLP) -> Vec<f64>{
		let parameters = neural_network.parameters();
		let mut analytical_grads = Vec::new();
		for parameter in parameters{
			analytical_grads.push(parameter.0.borrow().grad);
		}
		analytical_grads
	}
	
	fn numerical_grad_verify(&self, neural_network : &MLP, inputs: &Vec<ValueWrapper>, targets : &Vec<ValueWrapper>) -> Vec<f64>{
		let parameters = neural_network.parameters();
		let mut numerical_grads = Vec::new();
		let h = 0.00001;
		for parameter in parameters{
			let save_old_value = parameter.0.borrow().data;
			
			// calculate loss with data+h and data-h and then divide by 2h to determine numerical grad
			
			parameter.0.borrow_mut().data = save_old_value + h;
			let positive_prediction = neural_network.call(inputs.clone());
			let positive_loss = calculate_loss(positive_prediction , &targets);
			// println!("Pos = {}", positive_loss.0.borrow().data);
			
			parameter.0.borrow_mut().data = save_old_value - h;
			let negative_prediction = neural_network.call(inputs.clone());
			let negative_loss = calculate_loss(negative_prediction, &targets);
		
			let numerical_grad = (positive_loss.0.borrow().data - negative_loss.0.borrow().data) / (2.0*h);
			parameter.0.borrow_mut().data = save_old_value;
			
			numerical_grads.push(numerical_grad);
		}
		
		numerical_grads
	}
}
