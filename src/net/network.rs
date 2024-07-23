use crate::net::{cost_prime, FMatrix, FVector};
use nalgebra::{DVector, DMatrix};
use super::{sigmoid, sigmoid_prime};
use rand_distr::{Normal, Distribution};
use super::data::{DataPoint, extract_max_value};
use std::iter::zip;
use std::cmp::min;
use rand::thread_rng;
use rand::seq::SliceRandom;

pub struct Network {
    weights: Vec<FMatrix>,
    biases: Vec<FVector>,
    layer_count: usize,
}

impl Network {

    pub fn init_network(sizes: Vec<usize>) -> Network {

        let layer_count = sizes.len();

        let normal = Normal::new(0.0, 1.0).unwrap();
        let mut rng = rand::thread_rng();

        let weights: Vec<FMatrix> = zip(sizes[..sizes.len()-1].iter(), sizes[1..].iter())
            .map(|(prev, next)| {
                DMatrix::from_fn(*next, *prev, |_, _| normal.sample(&mut rng))
            })
            .collect();
        let biases: Vec<FVector> = sizes[1..].iter()
            .map(|size| DVector::from_fn(*size, |_, _| normal.sample(&mut rng)))
            .collect(); 
        
        Network {
            weights,
            biases,
            layer_count
        }
    }

    pub fn feedforward(&self, mut a: FVector) -> FVector {
        for (b, w) in zip(&self.biases, &self.weights) {
            a = ((w * a) + b).map(sigmoid);
        }
        a
    }

    pub fn stochastic_descent(&mut self, mut training_data: Vec<DataPoint>, epochs: usize, batch_size: usize, learning_rate: f64, test_data: Option<Vec<DataPoint>>) {

        if let Some(data) = &test_data {
            println!("Initial configuration: {} / {}", self.eval(data), data.len());
        }

        let rng = &mut thread_rng();

        let n = training_data.len();
        let batch_count = n / batch_size;
        for epoch in 0..epochs {
            training_data.shuffle(rng); //TODO this might be really slow? Might be worth benchmarking.
            for i in 0..batch_count {
                let batch = &training_data[i * batch_size..min((i + 1) * batch_size, n)];
                self.update_batch(batch, learning_rate);
            }
            if let Some(data) = &test_data {
                println!("Epoch {}: {} / {}", epoch, self.eval(data), data.len());
            }
            println!("Completed epoch {}", epoch);
        }
    }

    fn eval(&self, test_data: &Vec<DataPoint>) -> usize {
        let mut correct_values = 0;
        for point in test_data {
            let predicted_value = extract_max_value(&self.feedforward(point.input.clone()));
            let correct_value = extract_max_value(&point.output);
            if predicted_value == correct_value { 
                correct_values += 1; 
            }
        }
        correct_values
    }

    fn update_batch(&mut self, batch: &[DataPoint], learning_rate: f64) {
        // nabla is the gradient symbol.
        let mut nabla_b: Vec<FVector> = self.biases.iter().map(|b| FVector::zeros(b.nrows())).collect();
        let mut nabla_w: Vec<FMatrix> = self.weights.iter().map(|w| FMatrix::zeros(w.nrows(), w.ncols())).collect();

        for point in batch.iter() {
            let (delta_nabla_b, delta_nabla_w) = self.backprop(point); //this is the actual thing
            nabla_b = zip(&nabla_b, delta_nabla_b)
                .map(|(biases, d_biases)| d_biases + biases)
                .collect(); //TODO reallocating new vectors might be bad, could we do it in place instead?
            nabla_w = zip(&nabla_w, delta_nabla_w)
                .map(|(weights, d_weights)| d_weights + weights)
                .collect();
        }
        let batch_size = batch.len() as f64; //nasty conversion because usize is nasty? what's the correct way to do this?
        self.weights = zip(&self.weights, nabla_w)
            .map(|(w, nw)| w - (nw * (learning_rate / batch_size)))
            .collect();
        self.biases = zip(&self.biases, nabla_b)
            .map(|(b, nb)| b - (nb * (learning_rate / batch_size)))
            .collect();
    }

    fn backprop(&self, x: &DataPoint) -> (Vec<FVector>, Vec<FMatrix>) {
        let mut nabla_b: Vec<FVector> = self.biases.iter()
            .map(|b| FVector::zeros(b.nrows()))
            .collect();
        let mut nabla_w: Vec<FMatrix> = self.weights.iter()
            .map(|w| FMatrix::zeros(w.nrows(), w.ncols()))
            .collect();

        let mut activation = x.input.clone();
        let mut activations = vec![activation.clone()];
        let mut zs: Vec<FVector> = vec![];

        for (b, w) in zip(&self.biases, &self.weights) {
            let z = (w * activation) + b;
            activation = z.map(sigmoid);
            zs.push(z);
            activations.push(activation.clone());
        }

        let mut delta = cost_prime(&activations[activations.len() - 1], &x.output)
            .component_mul(&zs[zs.len() - 1].map(sigmoid_prime));
        let nabla_b_len = nabla_b.len();
        let nabla_w_len = nabla_w.len();
        let zs_len = zs.len();
        nabla_w[nabla_w_len - 1] = &delta * activations[activations.len() - 2].transpose();
        nabla_b[nabla_b_len- 1] = delta.clone();

        for l in 2..self.layer_count {
            let z = &zs[zs_len - l];
            let rl = z.map(sigmoid_prime);
            delta = (self.weights[self.weights.len() - l + 1].transpose() * delta).component_mul(&rl);
            nabla_w[nabla_w_len - l] = &delta * (activations[activations.len() - l - 1].transpose());
            nabla_b[nabla_b_len - l] = delta.clone();
        }
        (nabla_b, nabla_w)
    }

}