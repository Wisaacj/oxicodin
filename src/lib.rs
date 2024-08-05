pub mod engine;
pub mod neural;

#[cfg(test)]
mod tests {
    use crate::engine::Value;
    use crate::neural::{Module, MLP};

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

        println!(
            "a.data: {:?}, a.grad: {:?}",
            a.borrow().data,
            a.borrow().grad
        );
        println!(
            "b.data: {:?}, b.grad: {:?}",
            b.borrow().data,
            b.borrow().grad
        );
        println!(
            "c.data: {:?}, c.grad: {:?}",
            c.borrow().data,
            c.borrow().grad
        );
        println!(
            "d.data: {:?}, d.grad: {:?}",
            d.borrow().data,
            d.borrow().grad
        );
        println!(
            "e.data: {:?}, e.grad: {:?}",
            e.borrow().data,
            e.borrow().grad
        );
        println!(
            "f.data: {:?}, f.grad: {:?}",
            f.borrow().data,
            f.borrow().grad
        );
        println!(
            "l.data: {:?}, l.grad: {:?}",
            l.borrow().data,
            l.borrow().grad
        );

        println!("\n{:#?}", l.borrow()._prev);
    }

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
        assert_eq!((x1.borrow().grad * 10.0) as i32, -15);
        assert_eq!((w1.borrow().grad * 10.0) as i32, 10);
        assert_eq!((x2.borrow().grad * 10.0) as i32, 5);
        assert_eq!((w2.borrow().grad * 10.0) as i32, 0);
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
        assert_eq!((x1.borrow().grad * 10.0) as i32, -15);
        assert_eq!((w1.borrow().grad * 10.0) as i32, 10);
        assert_eq!((x2.borrow().grad * 10.0) as i32, 5);
        assert_eq!((w2.borrow().grad * 10.0) as i32, 0);
    }

    #[test]
    fn multi_layer_perceptron() {
        let xs = vec![
            vec![Value::new(2.0), Value::new(3.0), Value::new(-1.0)],
            vec![Value::new(3.0), Value::new(-1.0), Value::new(0.5)],
            vec![Value::new(0.5), Value::new(1.0), Value::new(1.0)],
            vec![Value::new(1.0), Value::new(1.0), Value::new(-1.0)],
        ];
        // Desired targets
        let ys = vec![
            Value::new(1.0),
            Value::new(-1.0),
            Value::new(-1.0),
            Value::new(1.0),
        ];
        const LEARNING_RATE: f32 = 0.05;

        let mlp = MLP::new(3, &[4, 4, 1]);

        // Training loop
        for k in 0..50 {
            // Forward pass
            let ypreds: Vec<Vec<Value>> = xs.iter().map(|x| mlp.forward(x)).collect();
            // Squared error loss
            let loss = ypreds
                .iter()
                .flatten()
                .zip(ys.iter())
                .map(|(ypred, y)| (ypred - y).pow(2.0))
                .sum::<Value>();

            // Backward pass
            mlp.zero_grad();
            loss.backward();

            // Update parameters via gradient descent
            mlp.parameters()
                .iter()
                .for_each(|p| p.borrow_mut().data -= LEARNING_RATE * p.grad());

            println!("{:?} {:?}", k, loss.data());
            println!(
                "{:?}",
                ypreds
                    .iter()
                    .flatten()
                    .map(|y| y.data())
                    .collect::<Vec<f32>>()
            );
        }
    }
}
