// Copyright 2024 Daniel Müller
//
// Licensed under the terms given in the file named "LICENSE"
// in the root folder of the project

#include "../extra_tiny_ann.hpp"

#define NR_CIFAR100_CLASSES 100

//What is the purpose of this? It demonstrates with a very simple example how
//training neural networks works in principle. But instead of having a neural
//network compute the output we want to bring in line with expected outputs
//from training data, this experiment considers the output as a variable to be
//optimized directly - no neural network is involved. This is not a realistic
//application, but much simpler to understand, converges to a solution very
//quickly and demonstrates that the cross_entropy_backward function works as
//intended.

void clip_to_nonnegative( Buffers& buffers ) {
	map( buffers , [](float x){return (MAX(0.0,x));} );
}

void normalize( Buffers& buffers ) {
	float sum;
	for( Buffer buffer : buffers ) {
		sum = foldl( buffer , 0.0 , [](uint32_t i,float a,float x){return (a+x);} );
		scale( buffer , (0.0==sum)?sum:(1.0 / sum) );
	}
}

void training_iteration( Buffers& layer_outputs , Buffers& layer_output_gradients , Buffers& layer_parameter_updates , Buffers& layer_1st_moment_estimates , Buffers layer_2nd_moment_estimates , Buffer target , Buffer objective , uint32_t& iteration , std::mt19937& rng ) {
	float average_obj,minimum_output,maximum_output,average_output;
	zero_data(layer_output_gradients);
	cross_entropy_forward( layer_outputs.back() , target , objective , NR_CIFAR100_CLASSES , 1 , 1 , 1 ); //loss function
	cross_entropy_backward( layer_outputs.back() , target , layer_output_gradients.back() , NR_CIFAR100_CLASSES , 1 , 1 , 1 ); //loss gradient
	adam( layer_output_gradients , layer_parameter_updates , layer_1st_moment_estimates , layer_2nd_moment_estimates , iteration , 0.01 ); //compute an update from the gradient
	add(layer_outputs,layer_parameter_updates); // the computed update is applied
	clip_to_nonnegative(layer_outputs); //make sure that layer_outputs is ...
	normalize(layer_outputs); //... a probability distribution, as cross entropy is only applicable to those.
	minimum_output = foldl( layer_outputs.back() , std::numeric_limits<float>::max() , [](uint32_t i,float a,float x){return (MIN(a,x));} );
	maximum_output = foldl( layer_outputs.back() , std::numeric_limits<float>::lowest() , [](uint32_t i,float a,float x){return (MAX(a,x));} );
	average_output = foldl( layer_outputs.back() , 0.0 , [](uint32_t i,float a,float x){return (a+x);} ) / length(layer_outputs.back());
	average_obj = foldl( objective , 0.0 , [](uint32_t i,float a,float x){return (a+x);} );
	std::cerr << "iteration " << iteration << ": objective = " << (average_obj) << " gradient norm = " << (norm(layer_output_gradients)) << " update norm = " << (norm(layer_parameter_updates)) << " min = " << minimum_output << " max = " << maximum_output << " avg = " << average_output << std::endl;
}

int main() {
	uint32_t iteration;
	std::mt19937 rng;
	Buffers layer_outputs;
	Buffers layer_output_gradients;
	Buffers layer_parameter_updates;
	Buffers layer_1st_moment_estimates;
	Buffers layer_2nd_moment_estimates;
	Buffer target;
	Buffer objective;
	rng.seed( (uint_fast32_t) time( NULL ) );
	layer_outputs.push_back( Buffer( new std::vector<float>(NR_CIFAR100_CLASSES)) );
	gaussian_init_parameters( layer_outputs , rng , 0.13 );
	map( layer_outputs.back() , [](float x){return (exp(x));} );
	normalize( layer_outputs );
	layer_output_gradients.push_back( Buffer( new std::vector<float>(NR_CIFAR100_CLASSES)) );
	layer_parameter_updates.push_back( Buffer( new std::vector<float>(NR_CIFAR100_CLASSES)) );
	layer_1st_moment_estimates.push_back( Buffer( new std::vector<float>(NR_CIFAR100_CLASSES)) );
	layer_2nd_moment_estimates.push_back( Buffer( new std::vector<float>(NR_CIFAR100_CLASSES)) );
	target = Buffer( new std::vector<float>(NR_CIFAR100_CLASSES));
	objective = Buffer( new std::vector<float>(1));
	zero_data(layer_output_gradients);
	zero_data(layer_parameter_updates);
	zero_data(layer_1st_moment_estimates);
	zero_data(layer_2nd_moment_estimates);
	zero_data(target);
	target->at(0) = 1.0;
	zero_data(objective);
	for( iteration = 0 ; iteration < 1000 ; iteration++ ) {
		training_iteration( layer_outputs , layer_output_gradients , layer_parameter_updates , layer_1st_moment_estimates , layer_2nd_moment_estimates , target , objective , iteration , rng );
	}
	return 0;
}
