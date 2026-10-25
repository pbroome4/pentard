use burn::tensor::Tensor;
use burn::prelude::*;

fn main(){
    let mut v = vec![0.0;1000];
    for i in 0..10{
        for j in 0..10{
            for k in 0..10{
                v[i*100 + j*10 +k] = (i+j+k) as f32;
            }
        }
    }
    let td = TensorData::new(v, [10,10,10]);
    let t = Tensor::<3>::from_data(td, &Default::default());
    println!("{}", t);
}