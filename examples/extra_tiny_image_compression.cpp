// Copyright 2024 Daniel Müller
//
// Licensed under the terms given in the file named "LICENSE"
// in the root folder of the project

#include <fstream>
#include <sstream>
#include <thread>
#include "../extra_tiny_ann.hpp"
#include "png_simplified.hpp"

//potential tasks to solve:
//- train on a downsampled input image first, then continue training with the same parameters,
//   but using the full-size input image. will accelerate training a lot.
//- change training iterations to examine how that influences the final loss

typedef struct png_image_t Training_Data;

Training_Data load_image_compression_data( const std::string& filename ) {
	Training_Data result;
	try {
		result = read_png_file(filename);
	} catch(...) {
		std::cerr << "error while trying to read '" << filename << "' because of: " << std::endl;
		throw;
	}
	return result;
}

void generate_data_and_targets( const Training_Data training_data , Buffer data , Buffer target , uint32_t nr_blocks , uint32_t height , uint32_t width , uint32_t block_index ) { // {{{
	uint32_t row,col,channel,row_offset,block_height;
	png_bytep rowp = NULL;
	png_bytep px = NULL;
	block_height = (training_data.height) / nr_blocks;
	if( ( height != block_height ) || ( width != static_cast<uint32_t>(training_data.width) ) ) {
		throw std::length_error( std::string( CURRENT_FUNCTION_NAME ) + ": internal error" );
	}
	row_offset = block_index * block_height;
	for( row = 0 ; row < block_height ; row++ ) {
		rowp = training_data.row_pointers[row_offset+row];
		for( col = 0 ; col < static_cast<uint32_t>(training_data.width) ; col++ ) {
			px = &(rowp[col * 4]);
			data->at( 0 * block_height * width + row * width + col ) = ( 2.0 * col ) / width - 1.0;
		//std::cerr << "fuuuuu " << (data->size()) << " " << block_height << " " << batch_size << " " << height << " " << width << " " << block_index << " " << row << " " << col << std::endl;
			data->at( 1 * block_height * width + row * width + col ) = ( 2.0 * ( row_offset + row ) ) / height - 1.0;
		//std::cerr << "ck" << std::endl;
			for( channel = 0 ; channel < 3 ; channel++ ) {
				target->at( channel * block_height * width + row * width + col ) = px[channel] / 255.0;
			}
		}
	}
} // }}}

void forward_and_backward( Buffers& layer_inputs , Buffers& layer_outputs , Buffers& layer_input_gradients , Buffers& layer_output_gradients , const Buffers& layer_parameters , Buffers& layer_parameter_gradients , Buffers& extra_data , Buffer target , Buffer objective , uint32_t channels , uint32_t batch_size , uint32_t height , uint32_t width , bool forward_only , bool no_param_gradient ) { // {{{
	int32_t i;
	//neural network forward pass (compute layer outputs)
	forward_pointwise_convolution( layer_inputs[0] , layer_outputs[0] , layer_parameters[0] , 2 , channels , batch_size , height , width );
	forward_bias( layer_inputs[1] , layer_outputs[1] , layer_parameters[1] , channels , channels , batch_size , height , width );
	forward_nonlin_leaky_relu( layer_inputs[2] , layer_outputs[2] , channels , channels , batch_size , height , width );
	for( i = 0 ; i < 6 ; i++ ) {
		forward_pointwise_convolution( layer_inputs[3+3*i] , layer_outputs[3+3*i] , layer_parameters[3+3*i] , channels , channels , batch_size , height , width );
		forward_bias( layer_inputs[4+3*i] , layer_outputs[4+3*i] , layer_parameters[4+3*i] , channels , channels , batch_size , height , width );
		forward_nonlin_leaky_relu( layer_inputs[5+3*i] , layer_outputs[5+3*i] , channels , channels , batch_size , height , width );
	}
	forward_pointwise_convolution( layer_inputs[21] , layer_outputs[21] , layer_parameters[21] , channels , 3 , batch_size , height , width );
	forward_bias( layer_inputs[22] , layer_outputs[22] , layer_parameters[22] , 3 , 3 , batch_size , height , width );
	forward_nonlin_sigmoid( layer_inputs[23] , layer_outputs[23] , 3 , 3 , batch_size , height , width );

	l2_forward( layer_outputs[23] , target , objective , 3 , batch_size , height , width ); //loss function
	if( forward_only ) {
		return;
	}
	l2_backward( layer_outputs[23] , target , layer_output_gradients[23] , 3 , batch_size , height , width ); //loss gradient

	//neural network backward pass (compute layer input gradients)
	backward_nonlin_sigmoid( layer_outputs[23] , layer_output_gradients[23] , layer_input_gradients[23] , layer_parameters[23] , 3 , 3 , batch_size , height , width );
	backward_bias( layer_outputs[22] , layer_output_gradients[22] , layer_input_gradients[22] , layer_parameters[22] , 3 , 3 , batch_size , height , width );
	backward_pointwise_convolution( layer_outputs[21] , layer_output_gradients[21] , layer_input_gradients[21] , layer_parameters[21] , channels , 3 , batch_size , height , width );
	for( i = 5 ; 0 <= i ; i-- ) {
		backward_nonlin_leaky_relu( layer_outputs[5+3*i] , layer_output_gradients[5+3*i] , layer_input_gradients[5+3*i] , layer_parameters[5+3*i] , channels , channels , batch_size , height , width );
		backward_bias( layer_outputs[4+3*i] , layer_output_gradients[4+3*i] , layer_input_gradients[4+3*i] , layer_parameters[4+3*i] , channels , channels , batch_size , height , width );
		backward_pointwise_convolution( layer_outputs[3+3*i] , layer_output_gradients[3+3*i] , layer_input_gradients[3+3*i] , layer_parameters[3+3*i] , channels , channels , batch_size , height , width );
	}
	backward_nonlin_leaky_relu( layer_outputs[2] , layer_output_gradients[2] , layer_input_gradients[2] , layer_parameters[2] , channels , channels , batch_size , height , width );
	backward_bias( layer_outputs[1] , layer_output_gradients[1] , layer_input_gradients[1] , layer_parameters[1] , channels , channels , batch_size , height , width );
	backward_pointwise_convolution( layer_outputs[0] , layer_output_gradients[0] , layer_input_gradients[0] , layer_parameters[0] , 2 , channels , batch_size , height , width );
	if( no_param_gradient ) {
		return;
	}

	//compute parameter gradients of each layer of the neural network
	param_gradient_bias( layer_inputs[22] , layer_output_gradients[22] , layer_parameter_gradients[22] , 3 , 3 , batch_size , height , width );
	param_gradient_pointwise_convolution( layer_inputs[21] , layer_output_gradients[21] , layer_parameter_gradients[21] , channels , 3 , batch_size , height , width );
	for( i = 0 ; i < 6 ; i++ ) {
		param_gradient_bias( layer_inputs[4+3*i] , layer_output_gradients[4+3*i] , layer_parameter_gradients[4+3*i] , channels , channels , batch_size , height , width );
		param_gradient_pointwise_convolution( layer_inputs[3+3*i] , layer_output_gradients[3+3*i] , layer_parameter_gradients[3+3*i] , channels , channels , batch_size , height , width );
	}
	param_gradient_bias( layer_inputs[1] , layer_output_gradients[1] , layer_parameter_gradients[1] , channels , channels , batch_size , height , width );
	param_gradient_pointwise_convolution( layer_inputs[0] , layer_output_gradients[0] , layer_parameter_gradients[0] , 2 , channels , batch_size , height , width );
} // }}}

void set_up_buffers( Buffers& layer_inputs , Buffers& layer_outputs , Buffers& layer_input_gradients , Buffers& layer_output_gradients , Buffers& layer_parameters , Buffers& layer_parameter_gradients , Buffers& layer_parameter_gradients2 , Buffers& extra_data , Buffers& layer_parameter_updates , Buffers& layer_1st_moment_estimates , Buffers& layer_2nd_moment_estimates , Buffer& target , Buffer& objective , uint32_t channels , uint32_t batch_size , uint32_t height , uint32_t width , bool forward_only , bool no_param_gradient , bool set_up_thread_independent_buffers ) { // {{{
	uint32_t i;
	std::vector<std::tuple<uint32_t,uint32_t,uint32_t,uint32_t>> buffer_sizes = std::vector<std::tuple<uint32_t,uint32_t,uint32_t,uint32_t>>();
	buffer_sizes.push_back(buffer_sizes_pointwise_convolution( 2 , channels , batch_size , height , width ));
	buffer_sizes.push_back(buffer_sizes_bias( channels , channels , batch_size , height , width ));
	buffer_sizes.push_back(buffer_sizes_nonlin_leaky_relu( channels , channels , batch_size , height , width ));
	for( i = 0 ; i < 6 ; i++ ) {
		buffer_sizes.push_back(buffer_sizes_pointwise_convolution( channels , channels , batch_size , height , width ));
		buffer_sizes.push_back(buffer_sizes_bias( channels , channels , batch_size , height , width ));
		buffer_sizes.push_back(buffer_sizes_nonlin_leaky_relu( channels , channels , batch_size , height , width ));
	}
	buffer_sizes.push_back(buffer_sizes_pointwise_convolution( channels , 3 , batch_size , height , width ));
	buffer_sizes.push_back(buffer_sizes_bias( 3 , 3 , batch_size , height , width ));
	buffer_sizes.push_back(buffer_sizes_nonlin_sigmoid( 3 , 3 , batch_size , height , width ));
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
	//std::cerr << i << " " << (std::get<0>(buffer_sizes.at(i))) << " " << (std::get<1>(buffer_sizes.at(i))) << " " << (std::get<2>(buffer_sizes.at(i))) << " " << (std::get<3>(buffer_sizes.at(i))) << " " << batch_size << " " << height << std::endl;
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

void log_image( Buffers& layer_outputs , Buffer target , uint32_t width , uint32_t height ) { // {{{
	uint32_t row,col;
	float averaged_output[height/8][width/8];
	float averaged_target[height/8][width/8];
	for( row = 0 ; row < (height/8) ; row++ ) {
		for( col = 0 ; col < (width/8) ; col++ ) {
			averaged_output[row][col] = 0.0;
			averaged_target[row][col] = 0.0;
		}
	}
	for( row = 0 ; (row/8) < (height/8) ; row++ ) {
		for( col = 0 ; (col/8) < (width/8) ; col++ ) {
			averaged_output[row/8][col/8] += (	layer_outputs.back()->at(0*width*height+row*width+col)+
								layer_outputs.back()->at(1*width*height+row*width+col)+
								layer_outputs.back()->at(2*width*height+row*width+col))/3.0/64.0;
			averaged_target[row/8][col/8] += (	target->at(0*width*height+row*width+col)+
								target->at(1*width*height+row*width+col)+
								target->at(2*width*height+row*width+col))/3.0/64.0;
		}
	}
	for( row = 0 ; row < (height/8) ; row++ ) {
		for( col = 0 ; col < (width/8) ; col++ ) {
			if( 0.8 < averaged_output[row][col] ) {
				std::cerr << "#";
			} else if( 0.6 < averaged_output[row][col] ) {
				std::cerr << "+";
			} else if( 0.4 < averaged_output[row][col] ) {
				std::cerr << "-";
			} else if( 0.2 < averaged_output[row][col] ) {
				std::cerr << ".";
			} else {
				std::cerr << " ";
			}
		}
		std::cerr << "  ";
		for( col = 0 ; col < (width/8) ; col++ ) {
			if( 0.8 < averaged_target[row][col] ) {
				std::cerr << "#";
			} else if( 0.6 < averaged_target[row][col] ) {
				std::cerr << "+";
			} else if( 0.4 < averaged_target[row][col] ) {
				std::cerr << "-";
			} else if( 0.2 < averaged_target[row][col] ) {
				std::cerr << ".";
			} else {
				std::cerr << " ";
			}
		}
		std::cerr << std::endl;
	}
} // }}}

float learning_rate_schedule( uint32_t iteration , uint32_t max_iteration ) {
	float warmup_end = 0.05 * max_iteration;
	if( iteration < warmup_end ) {
		return (0.02*(iteration/warmup_end)); //low learning rate during warmup
	}
	//return ( 0.02 * exp( -6.91 * ( iteration - warmup_end ) / ( max_iteration - warmup_end ) ) );
	return ( 0.02 * cos( ( 3.1415926535897 * 0.5 * ( iteration - warmup_end ) ) / ( max_iteration - warmup_end ) ) );
}

void worker_thread_function( Buffers* layer_inputs , Buffers* layer_outputs , Buffers* layer_input_gradients , Buffers* layer_output_gradients , const Buffers* layer_parameters , Buffers* layer_parameter_gradients , Buffers* extra_data , Buffer target , Buffer objective , uint32_t channels , uint32_t height , uint32_t width , bool forward_only , bool no_param_gradient , uint32_t thread_nr ) {
	forward_and_backward( *layer_inputs , *layer_outputs , *layer_input_gradients , *layer_output_gradients , *layer_parameters , *layer_parameter_gradients , *extra_data , target , objective , channels , 1 , height , width , forward_only , no_param_gradient );
}

void multithreaded_training_iteration( std::vector<Buffers>& thread_layer_inputs , std::vector<Buffers>& thread_layer_outputs , std::vector<Buffers>& thread_layer_input_gradients , std::vector<Buffers>& thread_layer_output_gradients , Buffers& layer_parameters , std::vector<Buffers>& thread_layer_parameter_gradients , Buffers& layer_parameter_gradients2 , std::vector<Buffers>& thread_extra_data , Buffers& layer_parameter_updates , Buffers& layer_1st_moment_estimates , Buffers& layer_2nd_moment_estimates , Buffers& thread_target , Buffers& thread_objective , uint32_t& iteration , uint32_t iterations , const Training_Data training_data , std::mt19937& rng , bool test_set , uint32_t channels , uint32_t width , uint32_t height , float learning_rate ) { // {{{
	std::vector<std::unique_ptr<std::thread>> worker_threads = std::vector<std::unique_ptr<std::thread>>();
	uint32_t thread_nr,nr_threads;
	uint64_t sum_microseconds;
	float average_obj = 0.0;
	float maximum_output = std::numeric_limits<float>::lowest();
	nr_threads = thread_layer_parameter_gradients.size();
	worker_threads.resize(nr_threads);
	for( thread_nr = 0 ; thread_nr < nr_threads ; thread_nr++ ) {
		zero_data(thread_layer_parameter_gradients.at(thread_nr)); //actual batch of gradients. smaller than effective batch size to save RAM
	}
	zero_data(layer_parameter_gradients2); //sum many of those batches into a larger effective batch
	sum_microseconds = 0;
	for( thread_nr = 0 ; thread_nr < nr_threads ; thread_nr++ ) {
		zero_data(thread_layer_inputs.at(thread_nr));
		generate_data_and_targets( training_data , thread_layer_inputs.at(thread_nr).front() , thread_target.at(thread_nr) , nr_threads , height , width , thread_nr );
		zero_data( thread_layer_parameter_gradients.at(thread_nr) );
		zero_data(thread_layer_outputs.at(thread_nr));
		zero_data(thread_layer_input_gradients.at(thread_nr));
		zero_data(thread_layer_output_gradients.at(thread_nr));
	}
	auto start = std::chrono::high_resolution_clock::now();
	for( thread_nr = 0 ; thread_nr < nr_threads ; thread_nr++ ) {
		std::cerr << ".";
		worker_threads.at(thread_nr) = std::unique_ptr<std::thread>( new std::thread( worker_thread_function ,  &(thread_layer_inputs.at(thread_nr)) , &(thread_layer_outputs.at(thread_nr)) , &(thread_layer_input_gradients.at(thread_nr)) , &(thread_layer_output_gradients.at(thread_nr)) , &layer_parameters , &(thread_layer_parameter_gradients.at(thread_nr)) , &(thread_extra_data.at(thread_nr)) , thread_target.at(thread_nr) , thread_objective.at(thread_nr) , channels , height , width , false , false , thread_nr  ) );
	}
	for( std::unique_ptr<std::thread>& t : worker_threads ) {
		t->join();
	}
	for( thread_nr = 0 ; thread_nr < nr_threads ; thread_nr++ ) {
		add( layer_parameter_gradients2 , thread_layer_parameter_gradients.at(thread_nr) );
	}
	auto elapsed = std::chrono::high_resolution_clock::now() - start;
	sum_microseconds += std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	if( 0 == iteration % 10 ) {
		for( thread_nr = 0 ; thread_nr < nr_threads ; thread_nr++ ) {
			maximum_output = foldl( thread_layer_outputs.at(thread_nr).back() , maximum_output , [](uint32_t i,float a,float x){return (MAX(a,x));} );
		}
		for( thread_nr = 0 ; thread_nr < nr_threads ; thread_nr++ ) {
			average_obj += foldl( thread_objective.at(thread_nr) , 0.0 , [](uint32_t i,float a,float x){return (a+x);} );
		}
	}
	//sgd( layer_parameter_gradients2 , layer_parameter_updates , layer_1st_moment_estimates , layer_2nd_moment_estimates , iteration , learning_rate );
	//the standard momentum coefficients are excessive for this task, which is essentially non-random, so i lowered them for faster convergence
	adam( layer_parameter_gradients2 , layer_parameter_updates , layer_1st_moment_estimates , layer_2nd_moment_estimates , iteration , learning_rate , 0.5 , 0.9 );
	add( layer_parameters , layer_parameter_updates );
	//parameter_norm_clipping( layer_parameters , 0.13 );
	if( 0 == iteration % 10 ) {
		for( thread_nr = 0 ; thread_nr < nr_threads ; thread_nr++ ) {
			log_image( thread_layer_outputs.at(thread_nr) , thread_target.at(thread_nr) , width , height );
		}
		std::cerr << std::endl << "iteration " << iteration << " (" << (static_cast<int>(round((1000.0*iteration)/iterations))/10.0) << " %):" << std::endl;
		std::cerr << "average objective of current batch = " << average_obj << " (this should go down during training)" << std::endl;
		std::cerr << "gradient norm = " << (norm(layer_parameter_gradients2)) << " (if this is 0 or explodes, you have a problem)" << std::endl;
		std::cerr << "update norm = " << (norm(layer_parameter_updates)) << " (ditto)" << std::endl;
		std::cerr << "parameter norm = " << (norm(layer_parameters)) << " (ditto)" << std::endl;
		std::cerr << "learning rate = " << learning_rate << std::endl;
		std::cerr << "maximum output = " << maximum_output << std::endl;
		std::cerr << "number parameters = " << (length(layer_parameters)) << std::endl;
		std::cerr << "FLOP/s is approximately " << ((1000000.0*6.0*(((float)width)*((float)width)*length(layer_parameters)))/sum_microseconds) << " (the bigger the better)" << std::endl;
		std::cerr << "compressed norm of parameter update for all layers: ";
		log_norms(layer_parameter_updates);
	}
} // }}}

void training_iteration( Buffers& layer_inputs , Buffers& layer_outputs , Buffers& layer_input_gradients , Buffers& layer_output_gradients , Buffers& layer_parameters , Buffers& layer_parameter_gradients , Buffers& extra_data , Buffers& layer_parameter_updates , Buffers& layer_1st_moment_estimates , Buffers layer_2nd_moment_estimates , Buffer target , Buffer objective , uint32_t& iteration , uint32_t iterations , const Training_Data training_data , std::mt19937& rng , bool test_set , uint32_t channels , uint32_t batch_size , uint32_t width , uint32_t height , float learning_rate ) { // {{{
	uint64_t sum_microseconds;
	float average_obj = 0.0;
	float maximum_output;
	zero_data(layer_parameter_gradients); //actual batch of gradients. smaller than effective batch size to save RAM
	sum_microseconds = 0;
	zero_data(layer_inputs);
	generate_data_and_targets( training_data , layer_inputs.front() , target , 1 , height , width , 0 );
	std::cerr << ".";
	zero_data(layer_outputs);
	zero_data(layer_input_gradients);
	zero_data(layer_output_gradients);
	auto start = std::chrono::high_resolution_clock::now();
	forward_and_backward( layer_inputs , layer_outputs , layer_input_gradients , layer_output_gradients , layer_parameters , layer_parameter_gradients , extra_data , target , objective , channels , batch_size , height , width , false , false );
	auto elapsed = std::chrono::high_resolution_clock::now() - start;
	sum_microseconds += std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	if( 0 == iteration % 10 ) {
		maximum_output = foldl( layer_outputs.back() , std::numeric_limits<float>::lowest() , [](uint32_t i,float a,float x){return (MAX(a,x));} );
		average_obj += foldl( objective , 0.0 , [](uint32_t i,float a,float x){return (a+x);} );
	}
	//sgd( layer_parameter_gradients , layer_parameter_updates , layer_1st_moment_estimates , layer_2nd_moment_estimates , iteration , learning_rate );
	//the standard momentum coefficients are excessive for this task, which is essentially non-random, so i lowered them for faster convergence
	adam( layer_parameter_gradients , layer_parameter_updates , layer_1st_moment_estimates , layer_2nd_moment_estimates , iteration , learning_rate , 0.5 , 0.9 );
	add( layer_parameters , layer_parameter_updates );
	//parameter_norm_clipping( layer_parameters , 0.13 );
	if( 0 == iteration % 10 ) {
		log_image( layer_outputs , target , width , height );
		std::cerr << std::endl << "iteration " << iteration << " (" << (static_cast<int>(round((1000.0*iteration)/iterations))/10.0) << " %):" << std::endl;
		std::cerr << "average objective of current batch = " << average_obj << " (this should go down during training)" << std::endl;
		std::cerr << "gradient norm = " << (norm(layer_parameter_gradients)) << " (if this is 0 or explodes, you have a problem)" << std::endl;
		std::cerr << "update norm = " << (norm(layer_parameter_updates)) << " (ditto)" << std::endl;
		std::cerr << "parameter norm = " << (norm(layer_parameters)) << " (ditto)" << std::endl;
		std::cerr << "learning rate = " << learning_rate << std::endl;
		std::cerr << "maximum output = " << maximum_output << std::endl;
		std::cerr << "number parameters = " << (length(layer_parameters)) << std::endl;
		std::cerr << "FLOP/s is approximately " << ((1000000.0*6.0*(((float)width)*((float)width)*length(layer_parameters)))/sum_microseconds) << " (the bigger the better)" << std::endl;
		std::cerr << "compressed norm of parameter update for all layers: ";
		log_norms(layer_parameter_updates);
	}
} // }}}

void train( Buffers& layer_parameters , const std::string& filename , Buffers& layer_parameters_copy ) { // {{{
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
	Training_Data training_data;
	uint32_t channels = 64;
	uint32_t batch_size = 1;
	uint32_t iterations = 12000;
	rng.seed( (uint_fast32_t) time( NULL ) );
	training_data = load_image_compression_data(filename);
	uint32_t height = training_data.height;
	uint32_t width = training_data.width;
	set_up_buffers( layer_inputs , layer_outputs , layer_input_gradients , layer_output_gradients , layer_parameters , layer_parameter_gradients , layer_parameter_gradients2 , extra_data , layer_parameter_updates , layer_1st_moment_estimates , layer_2nd_moment_estimates , target , objective , channels , batch_size , height , width , false , false , true );
	gaussian_init_parameters( layer_parameters , rng , 0.13 );
	layer_parameters_copy.resize(layer_parameters.size());
	int i,j;
	for( i = 0 ; i < layer_parameters_copy.size() ; i++ ) {
		layer_parameters_copy.at(i) = Buffer( new std::vector<float>() );
		layer_parameters_copy.at(i)->resize(layer_parameters.at(i)->size());
		for( j = 0 ; j < layer_parameters.at(i)->size() ; j++ ) {
			layer_parameters_copy.at(i)->at(j) = layer_parameters.at(i)->at(j);
		}
	}
	for( iteration = 0 ; iteration < iterations ; iteration++ ) {
		training_iteration( layer_inputs , layer_outputs , layer_input_gradients , layer_output_gradients , layer_parameters , layer_parameter_gradients , extra_data , layer_parameter_updates , layer_1st_moment_estimates , layer_2nd_moment_estimates , target , objective , iteration , iterations , training_data , rng , false , channels , batch_size , width , height , learning_rate_schedule( iteration , iterations ) );
	}
} // }}}

void train_multithreaded( Buffers& layer_parameters , const std::string& filename , uint32_t nr_threads ) { // {{{
	uint32_t iteration;
	std::mt19937 rng;
	std::vector<Buffers> thread_layer_inputs;
	std::vector<Buffers> thread_layer_outputs;
	std::vector<Buffers> thread_layer_input_gradients;
	std::vector<Buffers> thread_layer_output_gradients;
	std::vector<Buffers> thread_layer_parameter_gradients;
	Buffers layer_parameter_gradients2;
	std::vector<Buffers> thread_extra_data;
	Buffers layer_parameter_updates;
	Buffers layer_1st_moment_estimates;
	Buffers layer_2nd_moment_estimates;
	Buffers thread_target;
	Buffers thread_objective;
	Training_Data training_data;
	uint32_t channels = 64;
	uint32_t batch_size = 1;
	uint32_t iterations = 32000;
	rng.seed( (uint_fast32_t) time( NULL ) );
	thread_target.resize(nr_threads);
	thread_objective.resize(nr_threads);
	thread_layer_inputs.resize(nr_threads);
	thread_layer_outputs.resize(nr_threads);
	thread_layer_input_gradients.resize(nr_threads);
	thread_layer_output_gradients.resize(nr_threads);
	thread_layer_parameter_gradients.resize(nr_threads);
	thread_extra_data.resize(nr_threads);
	training_data = load_image_compression_data(filename);
	uint32_t height = training_data.height / nr_threads;
	uint32_t width = training_data.width;
	for( uint32_t i = 0 ; i < nr_threads ; i++ ) {
		set_up_buffers( thread_layer_inputs.at(i) , thread_layer_outputs.at(i) , thread_layer_input_gradients.at(i) , thread_layer_output_gradients.at(i) , layer_parameters , thread_layer_parameter_gradients.at(i) , layer_parameter_gradients2 , thread_extra_data.at(i) , layer_parameter_updates , layer_1st_moment_estimates , layer_2nd_moment_estimates , thread_target.at(i) , thread_objective.at(i) , channels , batch_size , height , width , false , false , ( 0 == i ) );
	}
	gaussian_init_parameters( layer_parameters , rng , 0.13 );
	for( iteration = 0 ; iteration < iterations ; iteration++ ) {
		multithreaded_training_iteration( thread_layer_inputs , thread_layer_outputs , thread_layer_input_gradients , thread_layer_output_gradients , layer_parameters , thread_layer_parameter_gradients , layer_parameter_gradients2 , thread_extra_data , layer_parameter_updates , layer_1st_moment_estimates , layer_2nd_moment_estimates , thread_target , thread_objective , iteration , iterations , training_data , rng , false , channels , width , height , learning_rate_schedule( iteration , iterations ) );
	}
} // }}}

float clamp( float x ) {
	return (MIN(1.0,MAX(0.0,x)));
}

int main( int argc, char** argv ) {
	if( 2 != argc ) {
		std::cerr << "please gimme a PNG image" << std::endl;
		return 1;
	}
	Buffers layer_parameters;
	Buffers layer_parameters_copy;
	//train_multithreaded( layer_parameters , std::string(argv[1]) , 4 );
	train( layer_parameters , std::string(argv[1]) , layer_parameters_copy );
	std::ofstream parameter_file = std::ofstream( "extra_tiny_image_compression_parameters.out" , std::ofstream::out | std::ofstream::binary );
	if( parameter_file.bad() ) {
		throw std::length_error( std::string( CURRENT_FUNCTION_NAME ) + ": error while trying to open extra_tiny_image_compression_parameters.out" );
	}
	uint32_t temp;
	try {
		temp = layer_parameters.size();
		parameter_file.write( (char*) &temp , sizeof(uint32_t) );
		for( Buffer buffer : layer_parameters ) {
			temp = buffer->size();
			parameter_file.write( (char*) &temp , sizeof(uint32_t) );
			if( 0 < buffer->size() ) {
				parameter_file.write( (char*) (buffer->data()) , sizeof(float) * buffer->size() );
			}
		}
	} catch(...) {
		std::cerr << "error while trying to write to extra_tiny_image_compression_parameters.out because of: " << std::endl;
		throw;
	}
	return 0;
}

//int main( int argc, char** argv ) {
//	if( 2 != argc ) {
//		std::cerr << "please gimme a PNG image" << std::endl;
//		return 1;
//	}
//	Buffers layer_inputs;
//	Buffers layer_outputs;
//	Buffers layer_input_gradients;
//	Buffers layer_output_gradients;
//	Buffers layer_parameter_gradients;
//	Buffers layer_parameter_gradients2;
//	Buffers extra_data;
//	Buffers layer_parameter_updates;
//	Buffers layer_1st_moment_estimates;
//	Buffers layer_2nd_moment_estimates;
//	Buffer target;
//	Buffer objective;
//	Buffers layer_parameters;
//	Training_Data training_data;
//	uint32_t channels = 64;
//	uint32_t batch_size = 1;
//	uint32_t iterations = 8000;
//	training_data = load_image_compression_data(std::string(argv[1]));
//	uint32_t height = training_data.height;
//	uint32_t width = training_data.width;
//	set_up_buffers( layer_inputs , layer_outputs , layer_input_gradients , layer_output_gradients , layer_parameters , layer_parameter_gradients , layer_parameter_gradients2 , extra_data , layer_parameter_updates , layer_1st_moment_estimates , layer_2nd_moment_estimates , target , objective , channels , batch_size , height , width , true , true , true );
//	//std::ifstream parameter_file = std::ifstream( "p_art_7_64_leaky_relu.out" , std::ifstream::in | std::ifstream::binary );
//	std::ifstream parameter_file = std::ifstream( "extra_tiny_image_compression_parameters.out" , std::ifstream::in | std::ifstream::binary );
//	if( parameter_file.bad() ) {
//		//throw std::length_error( std::string( CURRENT_FUNCTION_NAME ) + ": error while trying to open p_art_7_64_leaky_relu.out" );
//		throw std::length_error( std::string( CURRENT_FUNCTION_NAME ) + ": error while trying to open extra_tiny_image_compression_parameters.out" );
//	}
//	uint32_t temp;
//	try {
//		parameter_file.read( (char*) &temp , sizeof(uint32_t) );
//		layer_parameters.resize(temp);
//		for( Buffer buffer : layer_parameters ) {
//			parameter_file.read( (char*) &temp , sizeof(uint32_t) );
//			buffer->resize(temp);
//			if( 0 < buffer->size() ) {
//				parameter_file.read( (char*) (buffer->data()) , sizeof(float) * buffer->size() );
//			}
//		}
//	} catch(...) {
//		std::cerr << "error while trying to read from extra_tiny_image_compression_parameters.out because of: " << std::endl;
//		throw;
//	}
//	//try {
//	//	parameter_file.read( (char*) &temp , sizeof(uint64_t) );
//	//	parameter_file.read( (char*) &temp , sizeof(uint64_t) );
//	//	for( Buffer buffer : layer_parameters ) {
//	//		if( 0 < buffer->size() ) {
//	//			parameter_file.read( (char*) (buffer->data()) , sizeof(float) * buffer->size() );
//	//		}
//	//	}
//	//	parameter_file.close()
//	//} catch(...) {
//	//	std::cerr << "error while trying to read from p_art_7_64_leaky_relu.out because of: " << std::endl;
//	//	throw;
//	//}
//	generate_data_and_targets( training_data , layer_inputs.front() , target , 1 , height , width , 0 );
//	forward_and_backward( layer_inputs , layer_outputs , layer_input_gradients , layer_output_gradients , layer_parameters , layer_parameter_gradients , extra_data , target , objective , channels , batch_size , height , width , true , true );
//	uint32_t row,col,channel;
//	png_bytep rowp = NULL;
//	png_bytep px = NULL;
//	for( row = 0 ; row < static_cast<uint32_t>(training_data.height) ; row++ ) {
//		rowp = training_data.row_pointers[row];
//		for( col = 0 ; col < static_cast<uint32_t>(training_data.width) ; col++ ) {
//			px = &(rowp[col * 4]);
//			for( channel = 0 ; channel < 3 ; channel++ ) {
//				px[channel] = static_cast<png_byte>( round( clamp( layer_outputs.back()->at( channel * batch_size * height * width + row * width + col ) ) * 255.0 ) );
//			}
//		}
//	}
//	write_png_file( training_data , std::string(argv[1])+"_compressed.png" );
//	std::cerr << "result written to " << (std::string(argv[1])) << "_compressed.png" << std::endl;
//	return 0;
//}
