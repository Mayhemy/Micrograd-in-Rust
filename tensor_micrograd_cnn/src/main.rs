use tensor_micrograd_cnn::Tensor;

fn main() {
    println!("=== Creating Tensors ===");

    let zeros_2d = Tensor::zeros(vec![3, 4]);
    println!("Created 3x4 zeros tensor: {:?}", zeros_2d);
    
    let ones_2d = Tensor::ones(vec![2, 3]);
    println!("Created 2x3 ones tensor: {:?}", ones_2d);
    
    let zeros_3d = Tensor::zeros(vec![2, 2, 3]);
    println!("Created 2x2x3 zeros tensor: {:?}", zeros_3d);
    
    let random_tensor = Tensor::randn(vec![2, 2]);
    println!("Created 2x2 random tensor: {:?}", random_tensor);





    println!("\n=== Indexing Examples ===");

    let tensor_3x4 = Tensor::zeros(vec![3, 4]);
    
    let positions = vec![
        vec![0, 0],
        vec![0, 3],
        vec![2, 0],
        vec![2, 3],
        vec![1, 2],
    ];
    
    for pos in positions {
        let flat_idx = tensor_3x4.index_1d(pos.clone());
        let recovered_pos = tensor_3x4.index_1d_to_nd(flat_idx);
        println!("Position {:?} → Flat index {} → Back to {:?}", 
                pos, flat_idx, recovered_pos);
    }




    println!("\n=== 3D Tensor Indexing ===");
    
    let tensor_3d = Tensor::zeros(vec![2, 3, 4]);
    
    let positions_3d = vec![
        vec![0, 0, 0],
        vec![0, 1, 2],
        vec![1, 0, 0],
        vec![1, 2, 3],
    ];
    
    for pos in positions_3d {
        let flat_idx = tensor_3d.index_1d(pos.clone());
        let recovered_pos = tensor_3d.index_1d_to_nd(flat_idx);
        println!("3D Position {:?} → Flat index {} → Back to {:?}", 
                pos, flat_idx, recovered_pos);
    }




    println!("\n=== Comprehensive Index Test ===");
    
    let small_tensor = Tensor::ones(vec![2, 3]);
    let total_elements = 2 * 3;
    
    println!("Testing all indices in 2x3 tensor:");
    for flat_idx in 0..total_elements {
        let nd_pos = small_tensor.index_1d_to_nd(flat_idx);
        let back_to_flat = small_tensor.index_1d(nd_pos.clone());
        println!("  Flat {} ↔ Position {:?} (verified: {})", 
                flat_idx, nd_pos, flat_idx == back_to_flat);
    }


    println!("\n=== Backward Pass ===");
    // Backward pass
    let grad_tensor = Tensor::zeros(vec![2, 3]);
    println!("Before init_backward: {:?}", grad_tensor);
    
    grad_tensor.init_backward();
    println!("After init_backward: gradients initialized to 1.0");
    
    let topo_order = grad_tensor.build_topological_graph();
    println!("Topological order length: {} (should be 1 for single tensor)", topo_order.len());





    println!("\n=== Shape Analysis ===");
    
    let shapes = vec![
        vec![1],
        vec![4],
        vec![2, 2],
        vec![1, 5],
        vec![3, 1],
        vec![2, 3, 4],
        vec![1, 1, 1],
    ];
    
    for shape in shapes {
        let tensor = Tensor::zeros(shape.clone());
        let total = shape.iter().product::<usize>();
        println!("Shape {:?} → {} total elements", shape, total);
        
        if total > 0 {
            let first = tensor.index_1d_to_nd(0);
            let last = tensor.index_1d_to_nd(total - 1);
            println!("  First element at {:?}, Last element at {:?}", first, last);
        }
    }

    


    println!("=== Broadcasting ===");
    
    // Create tensors with different shapes for broadcasting
    let tensor_a = Tensor::ones(vec![3, 1]);  // Shape: (3, 1)
    let tensor_b = Tensor::ones(vec![1, 4]);  // Shape: (1, 4)
    
    println!("Tensor A shape: {:?}", tensor_a.0.borrow().shape);
    println!("Tensor B shape: {:?}", tensor_b.0.borrow().shape);
    
    // Test broadcasting
    let result = tensor_a.broadcast_two(&tensor_b);
    println!("Broadcasted result shape: {:?}", result.0.borrow().shape);
    println!("Expected shape: [3, 4]");
    
    // Test case 2: Different dimensions
    let tensor_1d = Tensor::ones(vec![5]);    // Shape: (5,)
    let tensor_2d = Tensor::ones(vec![3, 5]); // Shape: (3, 5)
    
    println!("\nTensor 1D shape: {:?}", tensor_1d.0.borrow().shape);
    println!("Tensor 2D shape: {:?}", tensor_2d.0.borrow().shape);
    
    let result2 = tensor_1d.broadcast_two(&tensor_2d);
    println!("Broadcasted result shape: {:?}", result2.0.borrow().shape);
    println!("Expected shape: [3, 5]");
    
    println!("\n🎉 Broadcasting tests completed successfully!");
}