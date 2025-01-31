use std::{
    fs::File, time::Instant, collections::HashMap
};
use csv;

// Activation Functions
fn act_handler(script_key:f64) -> fn(&Vec<f64>)->Vec<f64>{
    match script_key {
        1.0 => return relu,
        2.0 => return sigmoid,
        _ => todo!()
    }
}

fn relu(input:&Vec<f64>) -> Vec<f64>{
    let mut arr1 = vec![0_f64;input.len()];
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
fn dx_relu(input:&Vec<f64>) -> Vec<f64>{
    let mut arr1: Vec<f64> = vec![0_f64;input.len()];
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
fn sigmoid(input:&Vec<f64>) -> Vec<f64>{
    let mut arr1: Vec<f64> = vec![0_f64;input.len()];
    for (i, num) in input.iter().enumerate(){
        let denom = (num*-1.0).exp();
        arr1[i] = 1.0/(1.0+denom);
       
    }
    return arr1;
}
fn dx_sigmoid(input:&Vec<f64>) -> Vec<f64>{
    let arr1 = sigmoid(input);
    let mut arr2 = vec![0_f64;input.len()];
    for (i, num) in arr1.iter().enumerate(){
        arr2[i] = num * (1.0 - num);
    }
    return arr2;
}

fn for_prop(input: &Vec<f64>, weights:&Vec<Vec<f64>>, biases:&Vec<f64>, activation:fn(&Vec<f64>)->Vec<f64>) -> Vec<f64>{
    let mut arr1 = vec![0_f64; weights.len()];
    for (i,weight) in weights.iter().enumerate(){
        for (w1, z1) in weight.iter().zip(input.iter()){
            arr1[i] += w1 * z1;
        }
        arr1[i] += biases[i];
    }
    //println!("Preactivated: {:?}", arr1);
    return activation(&arr1);
}

fn read_mnist(file_path:&str) -> Result<Vec<Vec<i32>>, Box<dyn std::error::Error>>{
    let file = File::open(file_path)?;
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_reader(file);
    
    type Record = Vec<i32>;
    let mut layer = vec![];
    for result in rdr.deserialize() {
        let record: Record = result?;
        layer.push(record.clone());
    }
    Ok(layer)
}
 
fn open_model_csv(file_path:&str) -> Result<Vec<Vec<f64>>, Box<dyn std::error::Error>>{
    let file = File::open(file_path)?;
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_reader(file);
    
    type Record = Vec<f64>;
    let mut layer = vec![];
    for result in rdr.deserialize() {
        let record: Record = result?;
        layer.push(record.clone());
    }
    Ok(layer)
}
 
fn test_network(model:HashMap<i32, HashMap<&str, Vec<Vec<f64>>>>, testing_data:Vec<Vec<i32>>){
    let mut right = 0;
    let mut wrong = 0;
    let num_layers: i32 = model.keys().len() as i32 -1;
    for sample in testing_data{
        let correct_num: i32 = sample[0] as i32;
        let mut input_layer = vec![1_f64;sample.len()];
        for (i, old_num) in sample.iter().enumerate(){
            if i < 1{
                continue;
            }
            else{
            let num = *old_num as f64;
            input_layer[i-1] = num/255.00;    
            }
        }
        let activation_script = &model[&0]["activation"][0];
        let mut layer_input = input_layer;
        for j in 0..num_layers{
            let layer_index = j + 1;
            let activation_index = j.clone() as usize;
            let layer_output = for_prop(&layer_input, &model[&layer_index]["weights"], &model[&layer_index]["biases"][0], act_handler(activation_script[activation_index]));
            layer_input = layer_output;
        }
        let guessed_num: usize = layer_input.iter().enumerate()
                                            .max_by(|(_, a), (_, b)| a
                                            .partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                                            .map(|(index, _)| index)
                                            .unwrap();
        if correct_num == guessed_num as i32{
            right += 1;
        }else{
            wrong += 1;
        }
        
        //println!("Correct Num: {:?}\nGuessed Num: {:?}",correct_num, guessed_num);

    }
    println!("Right: {}\nWrong: {}", right, wrong);
}

fn main(){

    let start = Instant::now();

    let training_data = read_mnist("data/mnist_test.csv").expect("fuck you");
    let w1 = open_model_csv("data/w1.csv").expect("fuck you");
    let w2 = open_model_csv("data/w2.csv").expect("fuck you");
    let b1= open_model_csv("data/b1.csv").expect("fuck you");
    let b2= open_model_csv("data/b2.csv").expect("fuck you");
    //let activation_script = vec![relu, sigmoid];
    let activation_script = vec![vec![1.0, 2.0]];

    let mut model: HashMap<i32, HashMap<&str, Vec<Vec<f64>>>>= HashMap::new();
    let mut meta = HashMap::new();
    let mut layer1 = HashMap::new();
    let mut layer2 = HashMap::new();

    meta.insert("activation", activation_script.clone());
    meta.insert("initial", activation_script);

    layer1.insert("weights", w1);
    layer1.insert("biases", b1);
    layer2.insert("weights", w2);
    layer2.insert("biases", b2);
    model.insert(0, meta);
    model.insert(1, layer1);
    model.insert(2,layer2);
    test_network(model, training_data);
    


    let end = start.elapsed();
    println!("{:?}", end);

}