use std::fs;
use std::iter::zip;
use crate::net::{FVector, Float};
use crate::net::network::Network;
use crate::net::data::DataPoint;

const TRAIN_COUNT: usize = 50000;
const IMAGE_SIZE: usize = 28 * 28;
const TEST_COUNT: usize = 10000;
const EPOCHS: usize = 50;
const LEARNING_RATE: Float = 3.0; 
const BATCH_SIZE: usize = 10;

const IMAGES_PATH: &str = r"C:\Users\Hugsl\Documents\hnet\data\train-images.idx3-ubyte";
const LABELS_PATH: &str = r"C:\Users\Hugsl\Documents\hnet\data\train-labels.idx1-ubyte";

pub fn train_network_digits() {
    let mut network = Network::init_network(vec![IMAGE_SIZE, 100, 45, 10]);
    let (train_images, test_images) = load_images(IMAGES_PATH);
    let (train_labels, test_labels) = load_labels(LABELS_PATH);
    let training_data: Vec<DataPoint> = zip(train_images, train_labels)
        .map(|(image, label)| DataPoint {input: image, output: label})
        .collect();
    let testing_data: Vec<DataPoint> = zip(test_images, test_labels)
        .map(|(image, label)| DataPoint {input: image, output: label})
        .collect();

    network.stochastic_descent(training_data, EPOCHS, BATCH_SIZE, LEARNING_RATE, Some(testing_data));
}

fn byte_to_float(byte: &u8) -> Float {
    (*byte as Float) / 255.0 // dubious cast to float, but should clamp it to the range that we care about
}

fn byte_to_output_vec(byte: &u8) -> FVector {
    let mut v = FVector::zeros(10);
    v[*byte as usize] = 1.0;
    v
}

// the first output is the training images, the second is the testing images
fn load_images(path: &str) -> (Vec<FVector>, Vec<FVector>) {
    let data = fs::read(path).unwrap();

    /*
    let mut header_buffer: [u8; 4] = [0; 4];
    header_buffer.copy_from_slice(&data[4..8]);
    let image_count = i32::from_be_bytes(header_buffer);
    header_buffer.copy_from_slice(&data[8..12]);
    let image_rows = i32::from_be_bytes(header_buffer);
    header_buffer.copy_from_slice(&data[12..16]);
    let image_cols = i32::from_be_bytes(header_buffer);
    */

    let mut image_buffer: [u8; IMAGE_SIZE] = [0; IMAGE_SIZE];

    let header_offset = 16;
    let testing_offset = 16 + IMAGE_SIZE * TEST_COUNT;

    let training_images: Vec<FVector> = (0..TRAIN_COUNT).into_iter()
        .map(|i| {
            image_buffer.copy_from_slice(&data[header_offset + i * IMAGE_SIZE..header_offset + (i + 1) * IMAGE_SIZE]);
            FVector::from_iterator(IMAGE_SIZE, image_buffer.iter().map(byte_to_float))
        })
        .collect();

    let testing_images: Vec<FVector> = (0..TEST_COUNT).into_iter()
        .map(|i| {
            image_buffer.copy_from_slice(&data[testing_offset + i * IMAGE_SIZE..testing_offset + (i + 1) * IMAGE_SIZE]);
            FVector::from_iterator(IMAGE_SIZE, image_buffer.iter().map(byte_to_float))
        })
        .collect();

    (training_images, testing_images)
}

fn load_labels(path: &str) -> (Vec<FVector>, Vec<FVector>) {
    let data = fs::read(path).unwrap();

    let training_labels = (0..TRAIN_COUNT).into_iter()
        .map(|i| byte_to_output_vec(&data[8 + i]))
        .collect();

    let testing_offset = 8 + TEST_COUNT;
    let testing_labels = (0..TEST_COUNT).into_iter()
        .map(|i| byte_to_output_vec(&data[testing_offset + i]))
        .collect();

    (training_labels, testing_labels)
}

