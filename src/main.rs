mod neuralnet;

fn main() {
    let mut net = neuralnet::NeuralNet::new(&vec![2, 4, 2]);

    net.initialize();

    net.forward(vec![0.3, -0.4].into_iter());

    net.dump();
}

// fn main() {
//     let vec_of_elements = vec![1, 2, 3, 4, 5];

//     for (index, window) in vec_of_elements.windows(2).enumerate() {
//         if let [prev, current] = window {
//             println!("Index: {}, Previous: {}, Current: {}", index, prev, current);
//         }
//     }
// }
