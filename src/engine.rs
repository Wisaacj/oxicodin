use std::{
    cell::{Ref, RefCell, RefMut},
    collections::HashSet,
    ops::{Add, Div, Mul, Neg, Sub},
    rc::Rc,
};

#[derive(Debug)]
pub enum Op {
    Add,
    Mul,
    Div,
    Sub,
    Pow,
    Exp,
    None, // maybe we can remove this variant and instead use an option
}

#[derive(Debug)]
pub struct InnerValue {
    pub data: f32,
    pub grad: f32,
    pub _backward: fn(&InnerValue),
    pub _op: Op,
    pub _prev: Option<[Value; 2]>,
}

impl InnerValue {
    pub fn new(data: f32, children: Option<[Value; 2]>, op: Op) -> InnerValue {
        InnerValue {
            data,
            grad: 0.0,
            _backward: |_: &InnerValue| {},
            _prev: children,
            _op: op,
        }
    }
}

#[derive(Debug)]
/// Value is designed to provide shared, mutable access to the inner value through
/// reference counting and interior mutability.
pub struct Value {
    pub v: Rc<RefCell<InnerValue>>,
}

impl Value {
    pub fn new(data: f32) -> Value {
        Value {
            v: Rc::new(RefCell::new(InnerValue::new(data, None, Op::None))),
        }
    }

    /// New with children
    pub fn new_subtree(data: f32, children: [Value; 2], op: Op) -> Value {
        Value {
            v: Rc::new(RefCell::new(InnerValue::new(data, Some(children), op))),
        }
    }

    pub fn backward(&self) {
        let mut topo = Vec::new();
        let mut visited = HashSet::<*const InnerValue>::new();

        fn topological_sort(
            vertex: &Value,
            topo: &mut Vec<Value>,
            visited: &mut HashSet<*const InnerValue>,
        ) {
            let ptr = vertex.v.as_ptr();

            if visited.insert(ptr) {
                if let Some(children) = &vertex.borrow()._prev {
                    for child in children.iter() {
                        topological_sort(child, topo, visited);
                    }
                }
                topo.push(Value::clone(vertex));
            }
        }

        topological_sort(self, &mut topo, &mut visited);

        // Set the gradient of the output (self) to 1.0
        self.set_grad(1.0);

        // Reverse the topological order and call `_backward` on each node
        for node in topo.iter().rev() {
            let backward_fn = node.borrow()._backward;
            backward_fn(&node.borrow());
        }
    }
}

/// API improvments
impl Value {
    pub fn data(&self) -> f32 {
        self.borrow().data
    }

    pub fn set_data(&self, data: f32) {
        self.borrow_mut().data = data;
    }

    pub fn accum_data(&self, data: f32) {
        self.borrow_mut().data += data;
    }

    pub fn grad(&self) -> f32 {
        self.borrow().grad
    }

    pub fn set_grad(&self, grad: f32) {
        self.borrow_mut().grad = grad;
    }

    pub fn accum_grad(&self, grad: f32) {
        self.borrow_mut().grad += grad;
    }

    pub fn borrow_mut(&self) -> RefMut<'_, InnerValue> {
        self.v.borrow_mut()
    }

    pub fn borrow(&self) -> Ref<'_, InnerValue> {
        self.v.borrow()
    }
}

/// Activation functions
impl Value {
    pub fn tanh(&self) -> Value {
        let x = self.data();
        let t = ((2.0 * x).exp() - 1.0) / ((2.0 * x).exp() + 1.0);
        let out = Value::new_subtree(t, [Value::clone(self), Value::new(0.0)], Op::None);

        out.borrow_mut()._backward = |parent: &InnerValue| {
            if let Some(children) = &parent._prev {
                let mut logit = children[0].borrow_mut();

                // The derivative of tanh is (1 - tanh^2)
                logit.grad += (1.0 - parent.data.powi(2)) * parent.grad;
            }
        };

        out
    }

    pub fn relu(&self) -> Value {
        let out = Value::new_subtree(
            self.data().max(0.0),
            [Value::clone(self), Value::new(0.0)],
            Op::None,
        );

        out.borrow_mut()._backward = |parent: &InnerValue| {
            if let Some(children) = &parent._prev {
                let mut logit = children[0].borrow_mut();

                // The derivative of ReLU is 1 if x > 0, else 0
                logit.grad += (parent.data > 0.0) as i32 as f32 * parent.grad;
            }
        };

        out
    }
}

impl Clone for Value {
    /// Clones the Rc pointer to the inner value.
    ///
    /// Calling Rc::clone() does not perform a deep clone of the data inside Rc. Instead,
    /// it only increments the reference count and returns a new Rc pointing to the same data.
    /// This is an O(1) operation and is very fast.
    fn clone(&self) -> Value {
        Value {
            v: Rc::clone(&self.v),
        }
    }
}

impl Add<&Value> for &Value {
    type Output = Value;

    fn add(self, rhs: &Value) -> Self::Output {
        let out = Value::new_subtree(
            self.data() + rhs.data(),
            [Value::clone(self), Value::clone(rhs)],
            Op::Add,
        );

        // Defining a closure to calculate & accumulate the gradients of out's children.
        out.borrow_mut()._backward = |parent: &InnerValue| {
            if let Some(children) = &parent._prev {
                let mut lhs = children[0].borrow_mut();
                let mut rhs = children[1].borrow_mut();

                // Route the gradient from `parent` to its children. The local gradient
                // (d out) / (d lhs) = 1.0. This is then multipled by the gradient of the
                // parent according to the chain rule.
                lhs.grad += 1.0 * parent.grad;
                rhs.grad += 1.0 * parent.grad;
            };
        };

        out
    }
}

impl Add<f32> for &Value {
    type Output = Value;

    // Value + f32
    fn add(self, rhs: f32) -> Self::Output {
        let rhs = Value::new(rhs);
        self + &rhs
    }
}

impl Add<&Value> for f32 {
    type Output = Value;

    // f32 + Value
    fn add(self, rhs: &Value) -> Self::Output {
        rhs + self
    }
}

impl Mul<&Value> for &Value {
    type Output = Value;

    fn mul(self, rhs: &Value) -> Self::Output {
        let out = Value::new_subtree(
            self.data() * rhs.data(),
            [Value::clone(self), Value::clone(rhs)],
            Op::Mul,
        );

        out.borrow_mut()._backward = |parent: &InnerValue| {
            if let Some(children) = &parent._prev {
                let mut lhs = children[0].borrow_mut();
                let mut rhs = children[1].borrow_mut();

                // Chain rule baby
                lhs.grad += rhs.data * parent.grad;
                rhs.grad += lhs.data * parent.grad;
            };
        };

        out
    }
}

impl Mul<f32> for &Value {
    type Output = Value;

    // Value * f32
    fn mul(self, rhs: f32) -> Self::Output {
        let rhs = Value::new(rhs);
        self * &rhs
    }
}

impl Mul<&Value> for f32 {
    type Output = Value;

    // f32 * Value
    fn mul(self, rhs: &Value) -> Self::Output {
        rhs * self
    }
}

impl Value {
    pub fn pow(&self, exponent: f32) -> Value {
        let out = Value::new_subtree(
            self.data().powf(exponent),
            [Value::clone(self), Value::new(exponent)],
            Op::Pow,
        );

        out.borrow_mut()._backward = |parent: &InnerValue| {
            if let Some(children) = &parent._prev {
                let mut base = children[0].borrow_mut();
                let exponent = children[1].data();

                base.grad += exponent * base.data.powf(exponent - 1.0) * parent.grad;
            }
        };

        out
    }

    pub fn exp(&self) -> Value {
        let out = Value::new_subtree(
            self.data().exp(),
            [Value::clone(self), Value::new(0.0)],
            Op::Exp,
        );

        out.borrow_mut()._backward = |parent: &InnerValue| {
            if let Some(children) = &parent._prev {
                let mut lhs = children[0].borrow_mut();
                lhs.grad += parent.data * parent.grad;
            }
        };

        out
    }
}

impl Neg for &Value {
    type Output = Value;

    fn neg(self) -> Self::Output {
        self * -1.0
    }
}

impl Div<&Value> for &Value {
    type Output = Value;

    fn div(self, rhs: &Value) -> Self::Output {
        // a / b = a * (1 / b) = a * b^-1
        self * &rhs.pow(-1.0)
    }
}

impl Div<f32> for &Value {
    type Output = Value;

    // Value / f32
    fn div(self, rhs: f32) -> Self::Output {
        let rhs = Value::new(rhs);
        self / &rhs
    }
}

impl Sub<&Value> for &Value {
    type Output = Value;

    fn sub(self, rhs: &Value) -> Self::Output {
        self + &(-rhs)
    }
}

impl Sub<f32> for &Value {
    type Output = Value;

    // Value - f32
    fn sub(self, rhs: f32) -> Self::Output {
        let rhs = Value::new(rhs);
        self - &rhs
    }
}

impl Sub<&Value> for f32 {
    type Output = Value;

    // f32 - Value
    fn sub(self, rhs: &Value) -> Self::Output {
        rhs - self
    }
}

impl std::iter::Sum for Value {
    fn sum<I: Iterator<Item = Value>>(iter: I) -> Self {
        iter.fold(Value::new(0.0), |acc, x| &acc + &x)
    }
}
