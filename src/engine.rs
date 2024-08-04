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
    None,
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
    fn _new(data: f32, children: [Value; 2], op: Op) -> Value {
        Value {
            v: Rc::new(RefCell::new(InnerValue::new(data, Some(children), op))),
        }
    }

    pub fn borrow_mut(&self) -> RefMut<'_, InnerValue> {
        self.v.borrow_mut()
    }

    pub fn borrow(&self) -> Ref<'_, InnerValue> {
        self.v.borrow()
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
        self.borrow_mut().grad = 1.0;

        // Reverse the topological order and call `_backward` on each node
        for node in topo.iter().rev() {
            let backward_fn = node.borrow()._backward;
            backward_fn(&node.borrow());
        }
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
        let out = Value::_new(
            self.borrow().data + rhs.borrow().data,
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
        let out = Value::_new(
            self.borrow().data * rhs.borrow().data,
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
        let out = Value::_new(
            self.borrow().data.powf(exponent),
            [Value::clone(self), Value::new(exponent)],
            Op::Pow,
        );

        out.borrow_mut()._backward = |parent: &InnerValue| {
            if let Some(children) = &parent._prev {
                let mut base = children[0].borrow_mut();
                let exponent = children[1].borrow().data;

                base.grad += exponent * base.data.powf(exponent - 1.0) * parent.grad;
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

impl Sub<&Value> for &Value {
    type Output = Value;

    fn sub(self, rhs: &Value) -> Self::Output {
        self + &(-rhs)
    }
}
