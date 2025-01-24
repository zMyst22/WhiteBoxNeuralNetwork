fn for_prop(input:&[f64;5], weights:&[[f64; 5]; 3], bias:&[f64; 3], activation:fn(&[f64;3])->[f64;3]) -> [f64;3]{
    let mut arr1 = [0_f64; 3];
    for (i,weight) in weights.iter().enumerate(){
        for (w1, z1) in weight.iter().zip(input.iter()){
            arr1[i] += w1 * z1;
        }
        arr1[i] += bias[i];
    }
    println!("Preactivated: {:?}", arr1);
    return activation(&arr1);
}

fn relu(input:&[f64;3]) -> [f64;3]{
    let mut arr1 = [0_f64;3];
    for (i, num) in input.iter().enumerate(){
        if *num > 0.0 {
            arr1[i] = *num;
        }
        else{
            arr1[i] = 0.0;
        }
    }
    return arr1;
}
fn dx_relu(input:&[f64;3]) -> [f64;3]{
    let mut arr1 = [0_f64;3];
    for (i, num) in input.iter().enumerate(){
        if *num > 0.0 {
            arr1[i] = 1.0;
        }
        else{
            arr1[i] = 0.0;
        }
    }
    return arr1;
}
fn sigmoid(input:&[f64;3]) -> [f64;3]{
    let mut arr1 = [0_f64;3];
    for (i, num) in input.iter().enumerate(){
        let denom = (num*-1.0).exp();
        arr1[i] = 1.0/(1.0+denom);
       
    }
    return arr1;
}
fn dx_sigmoid(input:&[f64;3]) -> [f64;3]{
    let arr1 = sigmoid(input);
    let mut arr2 = [0_f64; 3];
    for (i, num) in arr1.iter().enumerate(){
        arr2[i] = num * (1.0 - num);
    }
    return arr2;
}

fn main(){
    const INPUT_LAYER:[f64; 5]= [0.1,0.2,0.3,0.4,0.5];
    const HIDDEN_LAYER:[[f64; 5]; 3] = [[-0.6,-0.7,0.8,0.9,-1.0],[0.6,0.7,0.8,0.9,1.0],[0.6,0.7,0.8,0.9,1.0]];
    const HIDDEN_BIAS:[f64; 3]= [-2.0,-0.25,2.0];

    let skibidi: [f64;3] = for_prop(&INPUT_LAYER, &HIDDEN_LAYER, &HIDDEN_BIAS, relu);
    println!("{:?}", skibidi);
    let skibidi: [f64;3] = for_prop(&INPUT_LAYER, &HIDDEN_LAYER, &HIDDEN_BIAS, dx_relu);
    println!("{:?}", skibidi);
}