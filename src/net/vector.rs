use std::ops;
use std::iter::zip;
use rand_distr::{Normal, Distribution};
use super::sigmoid;

// A vector of floats for usage in a neural network (e.g. as biases). Provides a variety of convenience operations.
#[derive(Clone)]
pub struct NVector {
    pub contents: Vec<f64>
}

impl NVector {
    pub fn clone_shape(&self) -> NVector{
        NVector {
            contents: vec![0.0; self.contents.len()]
        }
    }
}

#[derive(Clone)]
pub struct NMatrix {
    pub contents: Vec<Vec<f64>>
}

impl NMatrix {
    pub fn clone_shape(&self) -> NMatrix {
        NMatrix {
            contents: self.contents.iter().map(|a| vec![0.0; a.len()]).collect()
        }
    }

    pub fn app_all<F>(lhs: &NMatrix, rhs: &NMatrix, func: F) -> NMatrix 
    where
        F: Fn((&f64, &f64)) -> f64
    {
        NMatrix {
            contents: zip(&lhs.contents, &rhs.contents).map(|(a, b)| zip(a, b).map(&func).collect()).collect()
        }
    }

    pub fn generate_random(num_elements: usize, prev_num_elements: usize) -> NMatrix {
        let mut v: Vec<Vec<f64>> = vec![];
        let normal = Normal::new(0.0, 1.0).unwrap();
        let mut rng = rand::thread_rng();
        v.resize_with(num_elements, || (0..prev_num_elements).into_iter().map(|_| normal.sample(&mut rng)).collect());

        NMatrix {
            contents: v
        }
    }
}

impl ops::Add<&NMatrix> for NMatrix {
    type Output = NMatrix;

    fn add(self, other: &NMatrix) -> Self::Output {
        NMatrix {
            contents: zip(self.contents, &other.contents).map(|(a, b)| zip(a, b).map(|(a1, b1)| a1 + b1).collect()).collect()
        }
    }
}

impl ops::Mul<&NMatrix> for NVector {
    type Output = NVector;

    fn mul(self, trans: &NMatrix) -> NVector {
        debug_assert_eq!(trans.contents[0].len(), self.contents.len());

        NVector {
            contents: zip(self.contents, &trans.contents).map(|(element, row)| {
                row.iter().fold(0.0, |acc, x| acc + (x * element))
            }).collect()
        }
    }

}

impl ops::Add<&NVector> for NVector {
    type Output = NVector;

    fn add(self, other: &NVector) -> NVector {
        NVector {
            contents: zip(self.contents, &other.contents).map(|(a, b)| a + b).collect()
        }
    }
}

impl NVector {

    pub fn app_all<F>(lhs: &NVector, rhs: &NVector, func: F) -> NVector 
    where
        F: Fn((&f64, &f64)) -> f64
    {
        NVector {
            contents: zip(&lhs.contents, &rhs.contents).map(func).collect()
        }
    }

    pub fn generate_random(num_elements: usize) -> NVector {
        let mut v: Vec<f64> = vec![];
        let normal: Normal<f64> = Normal::new(0.0, 1.0).unwrap();
        let mut rng = rand::thread_rng();
        v.resize_with(num_elements, || normal.sample(&mut rng));

        NVector {
            contents: v
        }
    }

    pub fn sigmoid(&self) -> NVector {
        NVector {
            contents: self.contents.iter().map(|x| sigmoid(*x)).collect()
        }
    }

    pub fn relu(&self) -> NVector {
        NVector {
            contents: self.contents.iter().map(|x| f64::max(0.0, *x)).collect()
        }
    }
}