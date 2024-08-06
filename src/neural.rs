use crate::engine::Value;
use ndarray::parallel::prelude::*;
use ndarray::{Array1, Array2, Axis};
use rand::Rng;

type ActivationFn = fn(&Value) -> Value;

pub trait Module {
    fn parameters(&self) -> Vec<&Value>;

    fn zero_grad(&self) {
        self.parameters().iter().for_each(|p| p.set_grad(0.0));
    }
}

pub struct Neuron {
    weights: Array1<Value>,
    bias: Value,
    act_fn: Option<ActivationFn>,
}

impl Neuron {
    pub fn new(n_in: usize, act_fn: Option<ActivationFn>) -> Self {
        let mut rng = rand::thread_rng();
        let weights = (0..n_in)
            .map(|_| Value::new(rng.gen_range(-1.0..=1.0)))
            .collect();
        let bias = Value::new(rng.gen_range(-1.0..=1.0));

        Self {
            weights,
            bias,
            act_fn,
        }
    }

    pub fn forward(&self, x: &Array1<Value>) -> Value {
        assert_eq!(
            self.weights.len(),
            x.len(),
            "Input size must match number of weights"
        );

        let weighted_sum = self
            .weights
            .iter()
            .zip(x.iter())
            .map(|(w, x)| w * x)
            .fold(Value::new(0.0), |acc, v| &acc + &v);

        let pre_activation = &weighted_sum + &self.bias;

        match self.act_fn {
            Some(f) => f(&pre_activation), // activate the neuron
            None => pre_activation,
        }
    }
}

impl Module for Neuron {
    fn parameters(&self) -> Vec<&Value> {
        self.weights
            .iter()
            .chain(std::iter::once(&self.bias))
            .collect()
    }
}

pub struct Layer {
    neurons: Array1<Neuron>,
}

impl Layer {
    pub fn new(n_in: usize, n_out: usize, act_fn: Option<ActivationFn>) -> Self {
        let neurons = (0..n_out).map(|_| Neuron::new(n_in, act_fn)).collect();

        Self { neurons }
    }

    pub fn forward(&self, x: &Array1<Value>) -> Array1<Value> {
        self.neurons
            .par_iter()
            .map(|n| n.forward(x))
            .collect::<Vec<Value>>()
            .into()
    }

    // pub fn forward_batch(&self, x: &Array2<Value>) -> Array2<Value> {
    //     let (batch_size, _) = x.dim();

    //     Array2::from_shape_fn((batch_size, self.neurons.len()), |(i, j)| {
    //         self.neurons[j].forward(&x.row(i).to_owned())
    //     })
    // }

    pub fn forward_batch(&self, x: &Array2<Value>) -> Array2<Value> {
        let (batch_size, _) = x.dim();
        let n_out = self.neurons.len();

        let mut results = Vec::new();
        x.axis_iter(Axis(0))
            .into_par_iter()
            .map(|row| self.forward(&row.to_owned()))
            .collect_into_vec(&mut results);

        Array2::from_shape_vec((batch_size, n_out), results.into_iter().flatten().collect())
            .unwrap()
    }
}

impl Module for Layer {
    fn parameters(&self) -> Vec<&Value> {
        self.neurons.iter().flat_map(|n| n.parameters()).collect()
    }
}

pub struct MLP {
    layers: Vec<Layer>,
}

impl MLP {
    pub fn new(n_in: usize, n_outs: &[usize], act_fn: ActivationFn) -> Self {
        let layers = n_outs
            .iter()
            .enumerate()
            .scan(n_in, |n_in, (idx, &n_out)| {
                let layer_act_fn = if idx == n_outs.len() - 1 {
                    None // Last layer has no activation function.
                } else {
                    Some(act_fn)
                };
                let layer = Layer::new(*n_in, n_out, layer_act_fn);
                *n_in = n_out;
                Some(layer)
            })
            .collect();

        Self { layers }
    }

    pub fn forward(&self, x: &Array1<Value>) -> Array1<Value> {
        self.layers
            .iter()
            .fold(x.clone(), |x, layer| layer.forward(&x))
    }

    pub fn forward_batch(&self, x: &Array2<Value>) -> Array2<Value> {
        self.layers
            .iter()
            .fold(x.clone(), |x, layer| layer.forward_batch(&x))
    }
}

impl Module for MLP {
    fn parameters(&self) -> Vec<&Value> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }
}
