use rand::Rng;

#[derive(Debug)]
struct Node {
    output : f64,
    bias : f64,
    weights : Vec<f64>
}

#[derive(Debug)]
struct Layer {
    nodes : Vec<Node>
}

#[derive(Debug)]
pub struct NeuralNet {
    layers : Vec<Layer>
}

fn sigmoid(x : f64) -> f64 {
    return 1.0 / (1.0 + f64::exp(-x));
}

impl NeuralNet {
    pub fn new(layer_sizes : &Vec<usize>) -> Self {
        let mut input_size = 0;
        let net = NeuralNet {
            layers: (0..layer_sizes.len()).map(|i| {
                let layer = Layer::new(layer_sizes[i], input_size);
                input_size = layer_sizes[i];
                return layer;
            }).collect()
        };
        return net;
    }

    pub fn initialize(&mut self) {
        let mut rng = rand::thread_rng();
        self.layers.iter_mut().for_each(|layer| {
            layer.nodes.iter_mut().for_each(|node| {
                node.bias = rng.gen_range(-1.0..=1.0);
                node.weights.iter_mut().for_each(|w| *w = rng.gen_range(-1.0..1.0));
            })
        });
    }

    pub fn forward(&mut self, mut input : impl Iterator<Item = f64>) {
        assert!(!self.layers.is_empty());

        // Fill in the input..
        self.layers[0].nodes.iter_mut().for_each(|node| {
            node.output = input.next().unwrap();
        });

        for i in 0..self.layers.len() - 1 {
            if let [previous, current] = &mut self.layers[i..=i+1] {
                current.process(&previous);
            } else {
                assert!(false);
            }
        }
    }

    pub fn dump(&self) {
        let mut layer_index : usize = 0;
        println!("Neural Net: {} layers", self.layers.len());
        self.layers.iter().for_each(|layer| {
            println!(" - Layer {}, {} nodes, type={}",
                     layer_index,
                     layer.nodes.len(),
                     if layer_index == 0 {
                        "input"
                     } else if layer_index == self.layers.len() - 1 {
                        "output"
                     } else {
                        "hidden"
                     });
            layer_index = layer_index + 1;
            layer.nodes.iter().for_each(|node| {
                println!("   - Node, bias={:.3}, output={:.3}", node.bias, node.output);
                for i in 0..node.weights.len() {
                    println!("     - w[{}] = {:.3}", i, node.weights[i]);
                }
            });
        });
    }
}

impl Layer {
    pub fn new(layer_size : usize, previous_layer_size : usize) -> Self {
        return Layer {
            nodes: (0..layer_size).map(|_| Node {
                output: 0.0,
                bias: 0.0,
                weights: vec![0.0; previous_layer_size]
            }).collect()
        };
    }

    fn process(&mut self, previous : &Layer) {
        self.nodes.iter_mut().for_each(|node| {
            assert_eq!(node.weights.len(), previous.nodes.len());
            let mut sum = node.bias;
            for i in 0..node.weights.len() {
                sum += node.weights[i] * previous.nodes[i].output;
            }
            node.output = sigmoid(sum);
        });
    }
}

