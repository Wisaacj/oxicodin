use std::time::Instant;

use engine::Value;
use mnist::{Mnist, MnistBuilder};
use ndarray::Array2;
use neural::{Module, MLP};
use rand::seq::SliceRandom;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

pub mod engine;
pub mod neural;

pub fn load_mnist() -> (Array2<Value>, Vec<u8>, Array2<Value>, Vec<u8>) {
    // Deconstruct the returned Mnist struct.
    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(50_000)
        .validation_set_length(10_000)
        .test_set_length(10_000)
        .finalize();

    let train_data = Array2::from_shape_vec((50_000, 784), trn_img)
        .expect("Error converting images to Array2 struct")
        .map(|x| Value::new(*x as f32 / 255.0));
    println!("train data shape: {:?}", train_data.shape());

    let test_data = Array2::from_shape_vec((10_000, 784), tst_img)
        .expect("Error converting images to Array2 struct")
        .map(|x| Value::new(*x as f32 / 255.0));
    println!("test data shape: {:?}", test_data.shape());

    (train_data, trn_lbl, test_data, tst_lbl)
}

pub fn mse_loss(predictions: &Array2<Value>, targets: &[u8]) -> Value {
    let batch_size = predictions.nrows();
    let num_classes = predictions.ncols();

    let loss: Value = (0..batch_size)
        .into_par_iter()
        .map(|i| {
            let pred_row = predictions.row(i);
            let target = targets[i] as usize;

            (0..num_classes)
                .map(|j| {
                    let pred = &pred_row[j];
                    let target_value = if j == target { 1.0 } else { 0.0 };
                    (pred - target_value).pow(2.0)
                })
                .sum::<Value>()
        })
        .sum();

    &loss / (batch_size as f32 * num_classes as f32)
}

pub fn train_epoch(
    mlp: &MLP,
    train_images: &Array2<Value>,
    train_labels: &[u8],
    learning_rate: f32,
    batch_size: usize,
) {
    let mut rng = rand::thread_rng();

    // Shuffle the indices
    let mut indices: Vec<usize> = (0..train_images.nrows()).collect();
    indices.shuffle(&mut rng);

    println!(
        "training for an epoch of {} mini-batches of size {}\n",
        indices.len() / batch_size,
        batch_size
    );
    println!("{:=^52}", " training progress ");

    // Perform mini-batch stochastic gradient descent
    for (batch_num, batch_indices) in indices.chunks(batch_size).enumerate() {
        let t0 = Instant::now();
        let batch_inputs: Array2<Value> =
            Array2::from_shape_fn((batch_indices.len(), train_images.ncols()), |(i, j)| {
                train_images[[batch_indices[i], j]].clone()
            });

        let batch_targets: Vec<u8> = batch_indices.iter().map(|&i| train_labels[i]).collect();

        // Forward pass & compute loss
        let predictions: Array2<Value> = mlp.forward_batch(&batch_inputs);
        let loss = mse_loss(&predictions, &batch_targets);

        // Reset gradients & backward pass
        mlp.zero_grad();
        loss.backward();

        // Update parameters via mini-batch stochastic gradient descent
        for param in mlp.parameters() {
            param.accum_data(-learning_rate * param.grad());
            // param.borrow_mut().data -= learning_rate * param.grad();
        }

        let dt = Instant::now().duration_since(t0);
        println!(
            "batch: {:5} | loss: {:.6} | dt: {:>9.2?}",
            batch_num,
            loss.data(),
            dt
        );
    }
}

#[cfg(test)]
mod tests {
    use crate::engine::Value;
    use crate::neural::MLP;
    use crate::{load_mnist, mse_loss, train_epoch};
    use ndarray::Array2;

    #[test]
    fn mnist_training() {
        println!("loading mnist data...");
        let (train_data, train_labels, test_data, test_labels) = load_mnist();

        let mlp = MLP::new(28 * 28, &[128, 64, 10], Value::tanh);

        let num_epochs = 5;
        let learning_rate = 0.01;
        let batch_size = 32;

        for epoch in 0..num_epochs {
            train_epoch(&mlp, &train_data, &train_labels, learning_rate, batch_size);

            // Evaluate on the test set
            let test_predictions: Array2<Value> = mlp.forward_batch(&test_data);
            let test_loss = mse_loss(&test_predictions, &test_labels);
            println!("epoch: {:?} | test loss: {}", epoch + 1, test_loss.data());
        }
    }

    // #[test]
    // fn multi_layer_perceptron() {
    //     let xs = vec![
    //         vec![Value::new(2.0), Value::new(3.0), Value::new(-1.0)],
    //         vec![Value::new(3.0), Value::new(-1.0), Value::new(0.5)],
    //         vec![Value::new(0.5), Value::new(1.0), Value::new(1.0)],
    //         vec![Value::new(1.0), Value::new(1.0), Value::new(-1.0)],
    //     ];
    //     // Desired targets
    //     let ys = vec![
    //         Value::new(1.0),
    //         Value::new(-1.0),
    //         Value::new(-1.0),
    //         Value::new(1.0),
    //     ];
    //     const LEARNING_RATE: f32 = 0.05;

    //     let mlp = MLP::new(3, &[4, 4, 1], Value::tanh);

    //     // Training loop
    //     for k in 0..50 {
    //         // Forward pass
    //         let ypreds: Vec<Vec<Value>> = xs.iter().map(|x| mlp.forward(x)).collect();
    //         // Squared error loss
    //         let loss = ypreds
    //             .iter()
    //             .flatten() // Only one value per prediction
    //             .zip(ys.iter())
    //             .map(|(ypred, y)| (ypred - y).pow(2.0))
    //             .sum::<Value>();

    //         // Backward pass
    //         mlp.zero_grad();
    //         loss.backward();

    //         // Update parameters via gradient descent
    //         mlp.parameters()
    //             .iter()
    //             .for_each(|p| p.borrow_mut().data -= LEARNING_RATE * p.grad());

    //         println!("{:?} {:?}", k, loss.data());
    //         println!(
    //             "{:?}",
    //             ypreds
    //                 .iter()
    //                 .flatten()
    //                 .map(|y| y.data())
    //                 .collect::<Vec<f32>>()
    //         );
    //     }
    // }

    #[test]
    fn manual_neuron() {
        // Inputs x1, x2
        let x1 = Value::new(2.0);
        let x2 = Value::new(0.0);
        // Weights w1, w2
        let w1 = Value::new(-3.0);
        let w2 = Value::new(1.0);
        // Bias of the neuron
        let b = Value::new(6.88137);

        let x1w1 = &x1 * &w1;
        let x2w2 = &x2 * &w2;
        let x1w1_x2w2 = &x1w1 + &x2w2;
        // Unactivated output of the neuron
        let n = &x1w1_x2w2 + &b;
        // Tanh activation
        let e = (2.0 * &n).exp();
        let o = &(&e - 1.0) / &(&e + 1.0);
        // Backpropagate the gradient
        o.backward();

        println!("{:#?}", o);
        assert_eq!((x1.grad() * 10.0) as i32, -15);
        assert_eq!((w1.grad() * 10.0) as i32, 10);
        assert_eq!((x2.grad() * 10.0) as i32, 5);
        assert_eq!((w2.grad() * 10.0) as i32, 0);
    }

    #[test]
    fn manual_neuron_fused_activation() {
        // Inputs x1, x2
        let x1 = Value::new(2.0);
        let x2 = Value::new(0.0);
        // Weights w1, w2
        let w1 = Value::new(-3.0);
        let w2 = Value::new(1.0);
        // Bias of the neuron
        let b = Value::new(6.88137);

        let x1w1 = &x1 * &w1;
        let x2w2 = &x2 * &w2;
        let x1w1_x2w2 = &x1w1 + &x2w2;
        // Unactivated output of the neuron
        let n = &x1w1_x2w2 + &b;
        // Tanh activation
        let o = n.tanh();
        // Backpropagate the gradient
        o.backward();

        println!("{:#?}", o);
        assert_eq!((x1.grad() * 10.0) as i32, -15);
        assert_eq!((w1.grad() * 10.0) as i32, 10);
        assert_eq!((x2.grad() * 10.0) as i32, 5);
        assert_eq!((w2.grad() * 10.0) as i32, 0);
    }

    #[test]
    fn it_adds() {
        let a = Value::new(3.0);
        let b = Value::new(4.0);
        let c = &a + &b;

        println!("{:#?}", c);
    }

    #[test]
    fn it_adds_f32() {
        let a = Value::new(3.0);
        let b = 4.0;
        let c = &a + b;

        println!("{:#?}", c);
    }

    #[test]
    fn it_radds_f32() {
        let a = Value::new(3.0);
        let b = 4.0;
        let c = b + &a;

        println!("{:#?}", c);
    }

    #[test]
    fn it_multiplies() {
        let a = Value::new(3.0);
        let b = Value::new(4.0);
        let c = &a * &b;

        println!("{:#?}", c);
    }

    #[test]
    fn it_multiplies_f32() {
        let a = Value::new(3.0);
        let b = 4.0;
        let c = &a * b;

        println!("{:#?}", c);
    }

    #[test]
    fn it_rmultiplies_f32() {
        let a = Value::new(3.0);
        let b = 4.0;
        let c = b * &a;

        println!("{:#?}", c);
    }

    #[test]
    fn it_divides() {
        let a = Value::new(3.0);
        let b = Value::new(4.0);
        let c = &a / &b;

        println!("{:#?}", c);
    }

    #[test]
    fn it_goes_backward() {
        let a = Value::new(2.0);
        let b = Value::new(-3.0);
        let c = Value::new(10.0);
        let e = &a * &b;
        let d = &e + &c;
        let f = Value::new(-2.0);
        let l = &d * &f; // l for loss

        l.backward();

        println!("a.data: {:?}, a.grad: {:?}", a.data(), a.grad());
        println!("b.data: {:?}, b.grad: {:?}", b.data(), b.grad());
        println!("c.data: {:?}, c.grad: {:?}", c.data(), c.grad());
        println!("d.data: {:?}, d.grad: {:?}", d.data(), d.grad());
        println!("e.data: {:?}, e.grad: {:?}", e.data(), e.grad());
        println!("f.data: {:?}, f.grad: {:?}", f.data(), f.grad());
        println!("l.data: {:?}, l.grad: {:?}", l.data(), l.grad());

        println!("\n{:#?}", l.lock()._prev);
    }
}
