#[cfg(test)]
mod tests {
    use crate::Tensor;
    use std::collections::HashSet;

    #[test]
    fn test_tensor_creation_zeros() {
        let shape = vec![2, 3];
        let t = Tensor::zeros(shape.clone());
        let total = shape.iter().product::<usize>();

        // round-trip all valid indices
        for i in 0..total {
            let nd = t.index_1d_to_nd(i);
            assert_eq!(nd.len(), shape.len(), "expected {} dims", shape.len());
            assert_eq!(t.index_1d(nd.clone()), i);
        }

        println!("✅ test_tensor_creation_zeros");
    }

    #[test]
    fn test_tensor_creation_ones() {
        let shape = vec![2, 2];
        let t = Tensor::ones(shape.clone());
        let total = shape.iter().product::<usize>();

        for i in 0..total {
            let nd = t.index_1d_to_nd(i);
            assert_eq!(t.index_1d(nd.clone()), i);
        }

        println!("✅ test_tensor_creation_ones");
    }

    #[test]
    fn test_tensor_creation_randn() {
        let shape = vec![3, 4];
        let t = Tensor::randn(shape.clone());
        let total = shape.iter().product::<usize>();

        // we can't see the raw data here, but at least the shape
        // and indexing logic must still hold
        let last = t.index_1d_to_nd(total - 1);
        assert_eq!(last, vec![2, 3]);

        for i in 0..total {
            let nd = t.index_1d_to_nd(i);
            assert_eq!(t.index_1d(nd.clone()), i);
        }

        println!("✅ test_tensor_creation_randn");
    }

    #[test]
    fn test_index_1d_2d_tensor() {
        let t = Tensor::zeros(vec![3, 4]);

        // corners
        assert_eq!(t.index_1d(vec![0, 0]), 0);
        assert_eq!(t.index_1d(vec![0, 3]), 3);
        assert_eq!(t.index_1d(vec![2, 0]), 8);
        assert_eq!(t.index_1d(vec![2, 3]), 11);

        // middle
        assert_eq!(t.index_1d(vec![1, 2]), 6);

        println!("✅ test_index_1d_2d_tensor");
    }

    #[test]
    fn test_index_1d_3d_tensor() {
        let t = Tensor::zeros(vec![2, 3, 4]);

        assert_eq!(t.index_1d(vec![0, 0, 0]), 0);
        assert_eq!(t.index_1d(vec![0, 0, 1]), 1);
        assert_eq!(t.index_1d(vec![0, 1, 0]), 4);
        assert_eq!(t.index_1d(vec![1, 0, 0]), 12);
        assert_eq!(t.index_1d(vec![1, 2, 3]), 23);

        println!("✅ test_index_1d_3d_tensor");
    }

    #[test]
    fn test_index_1d_to_nd_roundtrip() {
        let t = Tensor::zeros(vec![2, 3, 4]);
        let cases = vec![
            vec![0, 0, 0],
            vec![0, 1, 2],
            vec![1, 0, 3],
            vec![1, 2, 1],
            vec![1, 2, 3],
        ];

        for orig in cases {
            let flat = t.index_1d(orig.clone());
            let back = t.index_1d_to_nd(flat);
            assert_eq!(back, orig);
        }

        println!("✅ test_index_1d_to_nd_roundtrip");
    }

    #[test]
    fn test_tensor_equality_and_hash() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let t1_clone = t1.clone();

        // pointer‐equality only
        assert_eq!(t1, t1_clone);
        assert_ne!(t1, t2);

        // hashing behavior
        let mut set = HashSet::new();
        assert!(set.insert(t1.clone()));
        assert!(!set.insert(t1.clone()), "same tensor shouldn't insert twice");
        assert!(set.insert(t2));

        println!("✅ test_tensor_equality_and_hash");
    }

    #[test]
    fn test_topological_sort_empty() {
        let t = Tensor::zeros(vec![2, 2]);
        let order = t.build_topological_graph();

        assert_eq!(order.len(), 1);
        assert_eq!(order[0], t);

        println!("✅ test_topological_sort_empty");
    }

    #[test]
    fn test_init_backward_does_not_panic() {
        let t = Tensor::zeros(vec![2, 3]);
        // this will set the internal grad to ones,
        // but we can't read it directly—just ensure no panic
        t.init_backward();

        // verify indexing still works
        let total = 2 * 3;
        for i in 0..total {
            let nd = t.index_1d_to_nd(i);
            assert_eq!(t.index_1d(nd.clone()), i);
        }

        println!("✅ test_init_backward_does_not_panic");
    }

    #[test]
    fn test_tensor_data_access() {
        let t = Tensor::ones(vec![2, 3]);

        // exercise the borrow_mut path
        {
            let _core = t.0.borrow_mut();
            // we can't inspect private fields here, but at least we know
            // we can borrow mutably without panic
        }
        // and then borrow again immutably
        {
            let _ = t.0.borrow();
        }

        println!("✅ test_tensor_data_access");
    }
}