use crate::engine::Value;
use rand::Rng;

pub trait Module {
    fn parameters(&self) -> Vec<&Value>;

    fn zero_grad(&self) {
        self.parameters()
            .iter()
            .for_each(|p| p.borrow_mut().grad = 0.0);
    }
}

pub struct Neuron {
    weights: Vec<Value>,
    bias: Value,
}

impl Neuron {
    pub fn new(n_in: usize) -> Self {
        let mut rng = rand::thread_rng();
        let weights = (0..n_in)
            .map(|_| Value::new(rng.gen_range(-1.0..=1.0)))
            .collect();
        let bias = Value::new(rng.gen_range(-1.0..=1.0));

        Self { weights, bias }
    }

    pub fn forward(&self, x: &[Value]) -> Value {
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
        // Activate the neuron
        pre_activation.tanh()
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
    neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(n_in: usize, n_out: usize) -> Self {
        let neurons = (0..n_out).map(|_| Neuron::new(n_in)).collect();

        Self { neurons }
    }

    pub fn forward(&self, x: &[Value]) -> Vec<Value> {
        self.neurons.iter().map(|n| n.forward(x)).collect()
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
    pub fn new(n_in: usize, n_outs: &[usize]) -> Self {
        let layers = n_outs
            .iter()
            .scan(n_in, |n_in, &n_out| {
                let layer = Layer::new(*n_in, n_out);
                *n_in = n_out;
                Some(layer)
            })
            .collect();

        Self { layers }
    }

    pub fn forward(&self, x: &[Value]) -> Vec<Value> {
        self.layers
            .iter()
            .fold(x.to_vec(), |x, layer| layer.forward(&x))
    }
}

impl Module for MLP {
    fn parameters(&self) -> Vec<&Value> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }
}
