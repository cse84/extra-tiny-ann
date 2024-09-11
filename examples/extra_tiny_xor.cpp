// Copyright 2024 Daniel Müller
//
// Licensed under the terms given in the file named "LICENSE"
// in the root folder of the project

#include <fstream>
#include "../extra_tiny_ann.hpp"

//What is the purpose of this? It demonstrates an example of a very small and
//simple problem that is provably impossible to solve with a linear function.
//This demonstration answers the question why neural networks need non-linear
//layers - they need them because otherwise even simple problems such as this
//here would be unsolvable. This is also just about the simplest task for which
//you might seriously consider using a neural network to solve it.
//And why is this called XOR? XOR is short for "exclusive or" and it is one of
//the basic logic functions. The other basic logic functions (AND, OR) can be
//considered solvable with linear functions and thus are even easier than XOR
//to learn. In this file, the boolean input and output variables are
//represented as -1 (false) and +1 (true), which conveniently makes XOR the
//same function as multiplication.

void generate_data_and_targets( std::mt19937& rng , bool test_set , Buffer data , Buffer target , uint32_t batch_size , int32_t index = -1 ) {
	uint_fast32_t i;
	for( i = 0 ; i < batch_size ; i++ ) {
		data->at( 0 * batch_size + i ) = 2.0 * (rng() % 2) - 1.0;
		data->at( 1 * batch_size + i ) = 2.0 * (rng() % 2) - 1.0;
		target->at( i ) = data->at( 0 * batch_size + i ) * data->at( 1 * batch_size + i );
	}
}

void forward_and_backward( Buffers& layer_inputs , Buffers& layer_outputs , Buffers& layer_input_gradients , Buffers& layer_output_gradients , const Buffers& layer_parameters , Buffers& layer_parameter_gradients , Buffers& extra_data , Buffer target , Buffer objective , uint32_t channels , uint32_t batch_size , bool forward_only , bool no_param_gradient ) { // {{{
	int32_t i;
	//neural network forward pass (compute layer outputs)
	forward_pointwise_convolution( layer_inputs[0] , layer_outputs[0] , layer_parameters[0] , 2 , channels , batch_size , 1 , 1 );
	for( i = 0 ; i < 2 ; i++ ) {
		forward_pointwise_convolution( layer_inputs[1+3*i] , layer_outputs[1+3*i] , layer_parameters[1+3*i] , channels , channels , batch_size , 1 , 1 );
		forward_bias( layer_inputs[2+3*i] , layer_outputs[2+3*i] , layer_parameters[2+3*i] , channels , channels , batch_size , 1 , 1 );
		forward_nonlin_leaky_relu( layer_inputs[3+3*i] , layer_outputs[3+3*i] , channels , channels , batch_size , 1 , 1 );
	}
	forward_pointwise_convolution( layer_inputs[7] , layer_outputs[7] , layer_parameters[7] , channels , 1 , batch_size , 1 , 1 );

	l2_forward( layer_outputs[7] , target , objective , 1 , batch_size , 1 , 1 ); //loss function
	if( forward_only ) {
		return;
	}
	l2_backward( layer_outputs[7] , target , layer_output_gradients[7] , 1 , batch_size , 1 , 1 ); //loss gradient

	//neural network backward pass (compute layer input gradients)
	backward_pointwise_convolution( layer_outputs[7] , layer_output_gradients[7] , layer_input_gradients[7] , layer_parameters[7] , channels , 1 , batch_size , 1 , 1 );
	for( i = 1 ; 0 <= i ; i-- ) {
		backward_nonlin_leaky_relu( layer_outputs[3+3*i] , layer_output_gradients[3+3*i] , layer_input_gradients[3+3*i] , layer_parameters[3+3*i] , channels , channels , batch_size , 1 , 1 );
		backward_bias( layer_outputs[2+3*i] , layer_output_gradients[2+3*i] , layer_input_gradients[2+3*i] , layer_parameters[2+3*i] , channels , channels , batch_size , 1 , 1 );
		backward_pointwise_convolution( layer_outputs[1+3*i] , layer_output_gradients[1+3*i] , layer_input_gradients[1+3*i] , layer_parameters[1+3*i] , channels , channels , batch_size , 1 , 1 );
	}
	backward_pointwise_convolution( layer_outputs[0] , layer_output_gradients[0] , layer_input_gradients[0] , layer_parameters[0] , 2 , channels , batch_size , 1 , 1 );
	if( no_param_gradient ) {
		return;
	}

	//compute parameter gradients of each layer of the neural network
	param_gradient_pointwise_convolution( layer_inputs[7] , layer_output_gradients[7] , layer_parameter_gradients[7] , channels , 1 , batch_size , 1 , 1 );
	for( i = 0 ; i < 2 ; i++ ) {
		param_gradient_bias( layer_inputs[2+3*i] , layer_output_gradients[2+3*i] , layer_parameter_gradients[2+3*i] , channels , channels , batch_size , 1 , 1 );
		param_gradient_pointwise_convolution( layer_inputs[1+3*i] , layer_output_gradients[1+3*i] , layer_parameter_gradients[1+3*i] , channels , channels , batch_size , 1 , 1 );
	}
	param_gradient_pointwise_convolution( layer_inputs[0] , layer_output_gradients[0] , layer_parameter_gradients[0] , 2 , channels , batch_size , 1 , 1 );
} // }}}

void set_up_buffers( Buffers& layer_inputs , Buffers& layer_outputs , Buffers& layer_input_gradients , Buffers& layer_output_gradients , Buffers& layer_parameters , Buffers& layer_parameter_gradients , Buffers& layer_parameter_gradients2 , Buffers& extra_data , Buffers& layer_parameter_updates , Buffers& layer_1st_moment_estimates , Buffers& layer_2nd_moment_estimates , Buffer& target , Buffer& objective , uint32_t channels , uint32_t batch_size , bool forward_only , bool no_param_gradient , bool set_up_thread_independent_buffers ) { // {{{
	uint32_t i;
	std::vector<std::tuple<uint32_t,uint32_t,uint32_t,uint32_t>> buffer_sizes = std::vector<std::tuple<uint32_t,uint32_t,uint32_t,uint32_t>>();
	buffer_sizes.push_back(buffer_sizes_pointwise_convolution( 2 , channels , batch_size , 1 , 1 ));
	for( i = 0 ; i < 2 ; i++ ) {
		buffer_sizes.push_back(buffer_sizes_pointwise_convolution(channels , channels , batch_size , 1 , 1));
		buffer_sizes.push_back(buffer_sizes_bias(channels , channels , batch_size , 1 , 1));
		buffer_sizes.push_back(buffer_sizes_nonlin_leaky_relu(channels , channels , batch_size , 1 , 1));
	}
	buffer_sizes.push_back(buffer_sizes_pointwise_convolution(channels , 1 , batch_size , 1 , 1));
	layer_inputs.resize(buffer_sizes.size());
	layer_outputs.resize(buffer_sizes.size());
	layer_input_gradients.resize(buffer_sizes.size());
	layer_output_gradients.resize(buffer_sizes.size());
	layer_parameter_gradients.resize(buffer_sizes.size());
	if( set_up_thread_independent_buffers ) {
		layer_parameters.resize(buffer_sizes.size());
		layer_parameter_gradients2.resize(buffer_sizes.size());
		layer_parameter_updates.resize(buffer_sizes.size());
		layer_1st_moment_estimates.resize(buffer_sizes.size());
		layer_2nd_moment_estimates.resize(buffer_sizes.size());
	}
	extra_data.resize(buffer_sizes.size());
	for( i = 0 ; i < buffer_sizes.size() ; i++ ) {
		layer_outputs.at(i) = Buffer(new std::vector<float>(std::get<0>(buffer_sizes.at(i))));
		if( 0 == i ) {
			layer_inputs.at(i) = Buffer(new std::vector<float>(std::get<1>(buffer_sizes.front())));
		} else {
			layer_inputs.at(i) = layer_outputs.at(i-1);
		}
		extra_data.at(i) = Buffer(new std::vector<float>(std::get<3>(buffer_sizes.at(i))));
		if( set_up_thread_independent_buffers ) {
			layer_parameters.at(i) = Buffer(new std::vector<float>(std::get<2>(buffer_sizes.at(i))));
		}
		if( forward_only || no_param_gradient ) {
			continue;
		}
		layer_parameter_gradients.at(i) = Buffer(new std::vector<float>(std::get<2>(buffer_sizes.at(i))));
		if( set_up_thread_independent_buffers ) {
			layer_parameter_gradients2.at(i) = Buffer(new std::vector<float>(std::get<2>(buffer_sizes.at(i))));
			layer_parameter_updates.at(i) = Buffer(new std::vector<float>(std::get<2>(buffer_sizes.at(i))));
			layer_1st_moment_estimates.at(i) = Buffer(new std::vector<float>(std::get<2>(buffer_sizes.at(i)),0.0));
			layer_2nd_moment_estimates.at(i) = Buffer(new std::vector<float>(std::get<2>(buffer_sizes.at(i)),0.0));
		}
	}
	target = Buffer(new std::vector<float>(std::get<0>(buffer_sizes.back())));
	objective = Buffer(new std::vector<float>(batch_size));
	if( forward_only ) {
		return;
	}
	for( i = 0 ; i < buffer_sizes.size() ; i++ ) {
		layer_input_gradients.at(i) = Buffer(new std::vector<float>(std::get<1>(buffer_sizes.at(i))));
	}
	for( i = 0 ; i < buffer_sizes.size()-1 ; i++ ) {
		layer_output_gradients.at(i) = layer_input_gradients.at(i+1);
	}
	layer_output_gradients.at(buffer_sizes.size()-1) = Buffer(new std::vector<float>(std::get<0>(buffer_sizes.at(i))));
	for( i = 0 ; i < buffer_sizes.size() ; i++ ) {
		//if( layer_outputs.at(i)->size() != layer_output_gradients.at(i)->size() ) {
		//	std::cerr << "whoopsie, buffer size mismatch: " << (layer_outputs.at(i)->size()) << " != " << (layer_output_gradients.at(i)->size()) << std::endl;
		//}
		//if( layer_inputs.at(i)->size() != layer_input_gradients.at(i)->size() ) {
		//	std::cerr << "whoopsie, buffer size mismatch: " << (layer_inputs.at(i)->size()) << " != " << (layer_input_gradients.at(i)->size()) << std::endl;
		//}
	}
} // }}}

float learning_rate_schedule( uint32_t iteration , uint32_t max_iteration ) {
	float warmup_end = 0.1 * max_iteration;
	if( iteration < warmup_end ) {
		return (0.001*(iteration/warmup_end)); //low learning rate during warmup
	}
	return ( 0.001 * cos( ( 3.1415926535897 * 0.5 * ( iteration - warmup_end ) ) / ( max_iteration - warmup_end ) ) );
}

void training_iteration( Buffers& layer_inputs , Buffers& layer_outputs , Buffers& layer_input_gradients , Buffers& layer_output_gradients , Buffers& layer_parameters , Buffers& layer_parameter_gradients , Buffers& layer_parameter_gradients2 , Buffers& extra_data , Buffers& layer_parameter_updates , Buffers& layer_1st_moment_estimates , Buffers layer_2nd_moment_estimates , Buffer target , Buffer objective , uint32_t& iteration , uint32_t iterations , std::mt19937& rng , bool test_set , uint32_t channels , uint32_t batch_size , uint32_t effective_batch_size , float learning_rate ) { // {{{
	uint32_t k;
	float average_obj = 0.0;
	zero_data(layer_parameter_gradients); //actual batch of gradients. smaller than effective batch size to save RAM
	zero_data(layer_parameter_gradients2); //sum many of those batches into a larger effective batch
	for( k = 0 ; k < effective_batch_size ; k += batch_size ) {
		zero_data(layer_inputs);
		generate_data_and_targets( rng , test_set , layer_inputs.front() , target , batch_size );
		//TODO: why did i add this before forward_and_backward? that way i'm losing the gradient of the last loop iteration
		add( layer_parameter_gradients2 , layer_parameter_gradients );
		zero_data( layer_parameter_gradients );
		zero_data(layer_outputs);
		zero_data(layer_input_gradients);
		zero_data(layer_output_gradients);
		forward_and_backward( layer_inputs , layer_outputs , layer_input_gradients , layer_output_gradients , layer_parameters , layer_parameter_gradients , extra_data , target , objective , channels , batch_size , false , false );
		if( 0 == iteration % 100 ) {
			average_obj += foldl( objective , 0.0 , [](uint32_t i,float a,float x){return (a+x);} ) / effective_batch_size;
		}
	}
	scale( layer_parameter_gradients2 , 1.0/effective_batch_size );
	//sgd( layer_parameter_gradients2 , layer_parameter_updates , layer_1st_moment_estimates , layer_2nd_moment_estimates , iteration , learning_rate );
	adam( layer_parameter_gradients2 , layer_parameter_updates , layer_1st_moment_estimates , layer_2nd_moment_estimates , iteration , learning_rate );
	add( layer_parameters , layer_parameter_updates );
	parameter_norm_clipping( layer_parameters , 0.8 );
	if( 0 == iteration % 100 ) {
		std::cerr << std::endl << "iteration " << iteration << " (" << (static_cast<int>(round((1000.0*iteration)/iterations))/10.0) << " %):" << std::endl;
		std::cerr << "average objective of current batch = " << average_obj << " (this should go down during training)" << std::endl;
		std::cerr << "gradient norm = " << (norm(layer_parameter_gradients2)) << " (if this is 0 or explodes, you have a problem)" << std::endl;
		std::cerr << "update norm = " << (norm(layer_parameter_updates)) << " (ditto)" << std::endl;
		std::cerr << "parameter norm = " << (norm(layer_parameters)) << " (ditto)" << std::endl;
		std::cerr << "learning rate = " << learning_rate << std::endl;
		log_norms( layer_parameter_updates );
	}
} // }}}

void train( Buffers& layer_parameters ) { // {{{
	uint32_t iteration;
	std::mt19937 rng;
	Buffers layer_inputs;
	Buffers layer_outputs;
	Buffers layer_input_gradients;
	Buffers layer_output_gradients;
	Buffers layer_parameter_gradients;
	Buffers layer_parameter_gradients2;
	Buffers extra_data;
	Buffers layer_parameter_updates;
	Buffers layer_1st_moment_estimates;
	Buffers layer_2nd_moment_estimates;
	Buffer target;
	Buffer objective;
	//8 channels and 2000 iterations are total overkill, but i didn't want unfortunate initializations
	//to create the impression that it doesn't work. you could just rerun it until you find an initialization
	//that works, but that would create the incorrect impression that getting a good result is based on luck.
	uint32_t channels = 8;
	uint32_t batch_size = 4;
	uint32_t effective_batch_size = 240;
	uint32_t iterations = 2000;
	rng.seed( (uint_fast32_t) time( NULL ) );
	set_up_buffers( layer_inputs , layer_outputs , layer_input_gradients , layer_output_gradients , layer_parameters , layer_parameter_gradients , layer_parameter_gradients2 , extra_data , layer_parameter_updates , layer_1st_moment_estimates , layer_2nd_moment_estimates , target , objective , channels , batch_size , false , false , true );
	gaussian_init_parameters( layer_parameters , rng , 0.8 );
	for( iteration = 0 ; iteration < iterations ; iteration++ ) {
		training_iteration( layer_inputs , layer_outputs , layer_input_gradients , layer_output_gradients , layer_parameters , layer_parameter_gradients , layer_parameter_gradients2 , extra_data , layer_parameter_updates , layer_1st_moment_estimates , layer_2nd_moment_estimates , target , objective , iteration , iterations , rng , false , channels , batch_size , effective_batch_size , learning_rate_schedule( iteration , iterations ) );
	}
} // }}}

void test( Buffers& layer_parameters , bool test_set ) { // {{{
	uint32_t i,iteration;
	std::mt19937 rng;
	Buffers layer_inputs;
	Buffers layer_outputs;
	Buffers layer_input_gradients;
	Buffers layer_output_gradients;
	Buffers layer_parameter_gradients;
	Buffers layer_parameter_gradients2;
	Buffers extra_data;
	Buffers layer_parameter_updates;
	Buffers layer_1st_moment_estimates;
	Buffers layer_2nd_moment_estimates;
	Buffers dummy;
	Buffer target;
	Buffer objective;
	uint32_t channels = 8;
	uint32_t batch_size = 4;
	uint32_t iterations = 1000;
	float average_obj = 0.0;
	double objective_2nd_moment; //needs to be double precision to reduce rounding error
	rng.seed( (uint_fast32_t) time( NULL ) );
	set_up_buffers( layer_inputs , layer_outputs , layer_input_gradients , layer_output_gradients , dummy , layer_parameter_gradients , layer_parameter_gradients2 , extra_data , layer_parameter_updates , layer_1st_moment_estimates , layer_2nd_moment_estimates , target , objective , channels , batch_size , true , true , true );
	objective_2nd_moment = 0;
	for( iteration = 0 ; iteration < iterations ; iteration += batch_size ) {
		zero_data(layer_inputs);
		zero_data(layer_outputs);
		generate_data_and_targets( rng , test_set , layer_inputs.front() , target , batch_size , iteration );
		for( i = 0 ; i < batch_size ; i++ ) {
			std::cerr << ".";
		}
		forward_and_backward( layer_inputs , layer_outputs , layer_input_gradients , layer_output_gradients , layer_parameters , layer_parameter_gradients , extra_data , target , objective , channels , batch_size , true , true );
		average_obj += foldl( objective , 0.0 , [](uint32_t i,float a,float x){return (a+x);} );
		objective_2nd_moment += foldl( objective , 0.0 , [](uint32_t i,float a,float x){return (a+x*x);} );
	}
	average_obj /= iterations;
	objective_2nd_moment /= iterations;
	std::cerr << std::endl;
	std::cerr << "average objective = " << average_obj <<  " standard deviation = " << (sqrt( ( objective_2nd_moment - average_obj * average_obj ) / iterations )) << std::endl;
} // }}}

int main() {
	Buffers layer_parameters;
	train( layer_parameters );
	std::cerr << "evaluation:" << std::endl;
	test( layer_parameters , true );
	return 0;
}

// vim: ts=8 : autoindent : textwidth=0 : foldmethod=marker
