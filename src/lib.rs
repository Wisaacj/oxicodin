use std::{
    cell::{Ref, RefCell, RefMut},
    collections::HashSet,
    ops::{Add, Mul},
    rc::Rc,
};

#[derive(Debug)]
pub struct InnerValue {
    pub data: f32,
    pub grad: f32,
    pub _backward: fn(&InnerValue),
    pub _prev: Option<[Value; 2]>,
}

impl InnerValue {
    pub fn new(data: f32, children: Option<[Value; 2]>) -> InnerValue {
        InnerValue {
            data,
            grad: 0.0,
            _backward: |_: &InnerValue| {},
            _prev: children,
        }
    }
}

#[derive(Debug)]
pub struct Value {
    pub v: Rc<RefCell<InnerValue>>,
}

impl Value {
    pub fn new(data: f32, children: Option<[Value; 2]>) -> Value {
        Value::from_inner_value(InnerValue::new(data, children))
    }

    pub fn clone(value: &Rc<RefCell<InnerValue>>) -> Value {
        Value {
            v: Rc::clone(value),
        }
    }

    fn from_inner_value(value: InnerValue) -> Value {
        Value {
            v: Rc::new(RefCell::new(value)),
        }
    }

    pub fn borrow_mut(&self) -> RefMut<'_, InnerValue> {
        self.v.borrow_mut()
    }

    pub fn borrow(&self) -> Ref<'_, InnerValue> {
        self.v.borrow()
    }

    pub fn backward(&self) {
        // let topo = Vec::new();
        // let visited = HashSet::new();
    }
}

impl Add<&Value> for &Value {
    type Output = Value;

    fn add(self, rhs: &Value) -> Self::Output {
        let out = Value::new(
            self.borrow().data + rhs.borrow().data,
            Some([Value::clone(&self.v), Value::clone(&rhs.v)]),
        );

        // Defining a closure to calculate/accumulate the gradients of out's children.
        out.borrow_mut()._backward = |parent: &InnerValue| {
            if let Some(children) = &parent._prev {
                let mut lhs = children[0].borrow_mut();
                let mut rhs = children[1].borrow_mut();

                lhs.grad += 1.0 * parent.grad;
                rhs.grad += 1.0 * parent.grad;
            };
        };

        out
    }
}

impl Mul<&Value> for &Value {
    type Output = Value;

    fn mul(self, rhs: &Value) -> Self::Output {
        let out = Value::new(
            self.borrow().data * rhs.borrow().data,
            Some([Value::clone(&self.v), Value::clone(&rhs.v)]),
        );

        // Defining a closure to calculate/accumulate the gradients of out's children.
        out.borrow_mut()._backward = |parent: &InnerValue| {
            if let Some(children) = &parent._prev {
                let mut lhs = children[0].borrow_mut();
                let mut rhs = children[1].borrow_mut();

                lhs.grad += rhs.data * parent.grad;
                rhs.grad += lhs.data * parent.grad;
            };
        };

        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_adds() {
        let a = Value::new(3.0, None);
        let b = Value::new(4.0, None);
        let c = &a + &b;

        println!("{:?}", c.borrow().data);
        println!("{:?}", c.borrow()._prev);
    }

    #[test]
    fn it_multiplies() {
        let a = Value::new(3.0, None);
        let b = Value::new(4.0, None);
        let c = &a * &b;

        println!("{:?}", c.borrow().data);
        println!("{:?}", c.borrow()._prev);
    }

    #[test]
    fn it_goes_backward() {
        let a = Value::new(3.0, None);
        let b = Value::new(4.0, None);
        let c = &a * &b;

        c.borrow_mut().grad = 1.0;
        (c.borrow()._backward)(&c.borrow());

        println!(
            "c.data: {:?}, c.grad: {:?}",
            c.borrow().data,
            c.borrow().grad
        );
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

        println!("\n{:?}", c.borrow()._prev);
    }
}
