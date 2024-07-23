use crate::net::FVector;



// A single data point, with `input` representing the input data
// and `output` representing the desired output for that input data.
pub struct DataPoint {
    pub input: FVector,
    pub output: FVector
}

pub fn extract_max_value(v: &FVector) -> usize {
    v.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(index, _)| index).unwrap()
}



