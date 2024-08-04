pub mod engine;

#[cfg(test)]
mod tests {
    use crate::engine::Value;

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
}
