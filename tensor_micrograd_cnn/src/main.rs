use std::time::Instant;
use tensor_micrograd_cnn::Tensor;
use tensor_micrograd_cnn::train::train;

fn header(t: &str) {
    println!("\n=== {t} ===");
}

fn main() {

    train();
    println!("🦀 Tensor Micrograd CNN — Forward/Backward micro-benches");

    // ============================
    // im2col: forward & backward
    // ============================
    header("🧱 im2col (B=2, C=3, H=W=64, k=3, stride=1, pad=1)");
    let (b, c, h, w) = (2, 3, 64, 64);
    let (kh, kw, stride, pad) = (3, 3, 1, 1);
    let x_im2col = Tensor::randn_uniform_init(vec![b, c, h, w]);

    // warm-up
    let _ = x_im2col.im2col(kh, kw, stride, pad);

    // forward timing
    let t0 = Instant::now();
    let x_col = x_im2col.im2col(kh, kw, stride, pad);
    let fwd_im2col = t0.elapsed();

    // sanity on shape: rows = C*Kh*Kw, cols = B*out_h*out_w
    let out_h = (h + 2 * pad - kh) / stride + 1;
    let out_w = (w + 2 * pad - kw) / stride + 1;
    let expect_shape = vec![c * kh * kw, b * out_h * out_w];
    assert_eq!(x_col.shape(), expect_shape, "im2col shape mismatch");

    // backward timing
    x_im2col.zero_grad();
    let t1 = Instant::now();
    x_col.init_backward(); // seeds grad=1 over x_col and backprops into x_im2col
    let bwd_im2col = t1.elapsed();

    println!(
        "im2col: shape {:?} -> {:?} | forward: {:?} | backward: {:?}",
        vec![b, c, h, w],
        x_col.shape(),
        fwd_im2col,
        bwd_im2col
    );

    // ============================
    // conv2d: forward & backward
    // ============================
    header("🧮 Conv2d (B=8, Cin=3, H=W=64, Cout=16, k=3, stride=1, pad=1)");
    let (b2, cin, cout, hh, ww) = (8, 3, 16, 64, 64);
    let (k_h, k_w, s, p) = (3, 3, 1, 1);

    let x_conv = Tensor::randn_uniform_init(vec![b2, cin, hh, ww]);     // requires_grad = true
    let w = Tensor::kaiming_he_init(vec![cout, cin, k_h, k_w]);         // requires_grad = true
    let b_bias = Tensor::zeros(vec![cout]);                             // bias broadcasted

    // warm-up
    let _ = x_conv.conv2d(&w, s, p, Some(&b_bias));

    // forward timing
    let t2 = Instant::now();
    let y = x_conv.conv2d(&w, s, p, Some(&b_bias));
    let fwd_conv = t2.elapsed();

    // expected output shape
    let oh = (hh + 2 * p - k_h) / s + 1;
    let ow = (ww + 2 * p - k_w) / s + 1;
    assert_eq!(y.shape(), vec![b2, cout, oh, ow], "conv2d shape mismatch");

    // backward timing
    x_conv.zero_grad();
    w.zero_grad();
    b_bias.zero_grad();
    let t3 = Instant::now();
    y.init_backward(); // backprops into x_conv, w, b_bias (via im2col + matmul paths)
    let bwd_conv = t3.elapsed();

    println!(
        "conv2d: out {:?} | forward: {:?} | backward: {:?}",
        y.shape(),
        fwd_conv,
        bwd_conv
    );

    // ============================
    // 2D matmul: forward & backward
    // ============================
    header("🧮 MatMul 2D (A: 512x512, B: 512x512)");
    let (m, n, ydim) = (512usize, 512usize, 512usize);
    let a = Tensor::randn_uniform_init(vec![m, n]);
    let b = Tensor::randn_uniform_init(vec![n, ydim]);

    // warm-up
    let _ = a.matmul(&b);

    // forward timing
    let t4 = Instant::now();
    let c = a.matmul(&b); // [m, ydim]
    let fwd_mm = t4.elapsed();
    assert_eq!(c.shape(), vec![m, ydim], "matmul 2D shape mismatch");

    // backward timing
    a.zero_grad();
    b.zero_grad();
    let t5 = Instant::now();
    c.init_backward(); // computes dA = dC @ B^T and dB = A^T @ dC
    let bwd_mm = t5.elapsed();

    println!(
        "matmul: [{}x{}] @ [{}x{}] -> [{}x{}] | forward: {:?} | backward: {:?}",
        m, n, n, ydim, m, ydim, fwd_mm, bwd_mm
    );

    // ============================
    // Summary
    // ============================
    header("🏁 Summary");
    println!("im2col   → fwd: {:?} | bwd: {:?}", fwd_im2col, bwd_im2col);
    println!("conv2d   → fwd: {:?} | bwd: {:?}", fwd_conv, bwd_conv);
    println!("matmul2D → fwd: {:?} | bwd: {:?}", fwd_mm, bwd_mm);

    println!("\n🎉 Done.");
}
