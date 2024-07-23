pub mod vector;
pub mod network;
pub mod data;
use std::f64;
use nalgebra::{DVector, DMatrix};

pub type Float = f64;
pub type FMatrix = DMatrix<Float>;
pub type FVector = DVector<Float>;

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + f64::exp(-x))
}

fn sigmoid_prime(x: f64) -> f64 {
    sigmoid(x) * (1.0 - sigmoid(x))
}

fn relu(x: f64) -> f64 {
    //f64::max(0.0, x)
    if x <= 0.0 {
        0.01 * x
    } else {
        x
    }
}

fn relu_prime(x: f64) -> f64 {
    if x <= 0.0 {
        0.01
    } else {
        1.0
    }
}

fn cost_prime(output_activations: &FVector, y: &FVector) -> FVector {
    output_activations - y
}