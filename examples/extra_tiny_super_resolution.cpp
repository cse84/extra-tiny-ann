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
//- use better down-/upsampling method (currently box filtering, the second-worst of all).
//   https://www.imagemagick.org/Usage/filter/ has a good overview of alternatives, bilinear is probably the simplest after box filtering.
//- sample data not just from the full-size images, but also from downsampled versions.
//   can be handled during loading and adds only 33% of extra RAM usage, like MIP mapping.
//- multithreaded testing
//- 3x or 4x superresolution

typedef std::shared_ptr<std::vector<struct png_image_t>> SR_Data_Buffer;

SR_Data_Buffer load_super_resolution_data( const std::string& dirname ) { // {{{
	int32_t i;
	char old_dir[FILENAME_MAX];
	if( NULL == getcwd(old_dir,sizeof(old_dir)) ) {
		throw std::length_error( std::string( CURRENT_FUNCTION_NAME ) + ": cannot get current directory" );
	}
	std::shared_ptr<std::vector<std::string>> entries = get_dir_entries( dirname );
	SR_Data_Buffer result = SR_Data_Buffer(new std::vector<struct png_image_t>());
	if( 0 > chdir(dirname.c_str()) ) {
		throw std::length_error( std::string( CURRENT_FUNCTION_NAME ) + ": cannot change to directory '" + dirname + "' because of '" + std::string(strerror(errno)) + "'" );
	}
	for( const std::string& filename : (*entries) ) {
		try {
			result->push_back(read_png_file(filename));
			if( ( result->back().width < 64 ) && ( result->back().height < 64 ) ) {
				for( i = 0 ; i < result->back().height ; i++ ) {
					free(result->back().row_pointers[i]);
				}
				free(result->back().row_pointers);
				result->pop_back();
				std::cerr << "skipped '" << filename << "' because it is smaller than 64x64." << std::endl;
			}
		} catch(...) {
			std::cerr << "error while trying to read '" << filename << "' because of: " << std::endl;
			if( 0 > chdir(old_dir) ) {
				;//one error is enough
			}
			throw;
		}
	}
	if( 0 > chdir(old_dir) ) {
		throw std::length_error( std::string( CURRENT_FUNCTION_NAME ) + ": cannot change to directory '" + std::string(old_dir) + "' because of '" + std::string(strerror(errno)) + "'" );
	}
	return result;
} // }}}

void generate_data_and_targets( const SR_Data_Buffer sr_images , std::mt19937& rng , Buffer data , Buffer target , uint32_t batch_size ) { // {{{
	uint_fast32_t i,row,col,channel,r,row_offset,col_offset;
	png_bytep rowp = NULL;
	png_bytep px = NULL;
	float averages[32][32][3];
	for( i = 0 ; i < batch_size ; i++ ) {
		r = rng() % (sr_images->size());
		row_offset = rng() % (sr_images->at(r).height-64);
		col_offset = rng() % (sr_images->at(r).width-64);
		for( row = 0 ; row < 32 ; row++ ) {
			for( col = 0 ; col < 32 ; col++ ) {
				for( channel = 0 ; channel < 3 ; channel++ ) {
					averages[row][col][channel] = 0.0;
				}
			}
		}
		for( row = 0 ; row < 64 ; row++ ) {
			rowp = sr_images->at(r).row_pointers[row_offset+row];
			for( col = 0 ; col < 64 ; col++ ) {
				px = &(rowp[(col_offset+col) * 4]);
				for( channel = 0 ; channel < 3 ; channel++ ) {
					data->at( channel * batch_size * 64 * 64 + i * 64 * 64 + row * 64 + col ) = px[channel] / 255.0;
					averages[row/2][col/2][channel] += px[channel] / 255.0 / 4.0;
				}
			}
		}
		//target is the residual of the input relative to a down- and then upsampled input.
		for( row = 0 ; row < 64 ; row++ ) {
			for( col = 0 ; col < 64 ; col++ ) {
				for( channel = 0 ; channel < 3 ; channel++ ) {
					target->at( channel * batch_size * 64 * 64 + i * 64 * 64 + row * 64 + col ) =
						data->at( channel * batch_size * 64 * 64 + i * 64 * 64 + row * 64 + col ) - averages[row/2][col/2][channel];
					data->at( channel * batch_size * 64 * 64 + i * 64 * 64 + row * 64 + col ) = averages[row/2][col/2][channel];
				}
			}
		}
	}
} // }}}

void forward_and_backward( Buffers& layer_inputs , Buffers& layer_outputs , Buffers& layer_input_gradients , Buffers& layer_output_gradients , const Buffers& layer_parameters , Buffers& layer_parameter_gradients , Buffers& extra_data , Buffer target , Buffer objective , uint32_t channels , uint32_t batch_size , uint32_t height , uint32_t width , bool forward_only , bool no_param_gradient ) { // {{{
	int32_t i;
	//neural network forward pass (compute layer outputs)
	forward_reshape( layer_inputs[0] , layer_outputs[0] , 3 , 6 , batch_size , height , height / 2 , width , width );
	//given that the input to this function has been produced by first downsampling by 2x (and then upsampling again), processing the input at full resolution
	//is pointless (the high frequency information is gone and is exactly what this network is supposed to recreate), so just downsample the input by 2x again,
	//but keep all the information (i.e. reshape the input to 4x the channels and half the height and width). just downsampling (not reshaping) would also be
	//possible, but requires a new layer type, namely a pooling layer (but not the global pooling layer, because pools the entire image into 1 pixel).
	forward_reshape( layer_inputs[1] , layer_outputs[1] , 6 , 12 , batch_size , height / 2 , height / 2 , width , width / 2 );
	forward_pointwise_convolution( layer_inputs[2] , layer_outputs[2] , layer_parameters[2] , 12 , channels , batch_size , height / 2 , width / 2 );
	forward_bias( layer_inputs[3] , layer_outputs[3] , layer_parameters[3] , channels , channels , batch_size , height / 2 , width / 2 );
	forward_nonlin_leaky_relu( layer_inputs[4] , layer_outputs[4] , channels , channels , batch_size , height / 2 , width / 2 );
	for( i = 0 ; i < 6 ; i++ ) {
		forward_depthwise_convolution( layer_inputs[5+4*i] , layer_outputs[5+4*i] , layer_parameters[5+4*i] , channels , channels , batch_size , height / 2 , width / 2 , 3 , 3 );
		forward_pointwise_convolution( layer_inputs[6+4*i] , layer_outputs[6+4*i] , layer_parameters[6+4*i] , channels , channels , batch_size , height / 2 , width / 2 );
		forward_bias( layer_inputs[7+4*i] , layer_outputs[7+4*i] , layer_parameters[7+4*i] , channels , channels , batch_size , height / 2 , width / 2 );
		forward_nonlin_leaky_relu( layer_inputs[8+4*i] , layer_outputs[8+4*i] , channels , channels , batch_size , height / 2 , width / 2 );
	}
	forward_reshape( layer_inputs[29] , layer_outputs[29] , channels , channels / 2 , batch_size , height / 2 , height / 2 , width / 2 , width );
	forward_reshape( layer_inputs[30] , layer_outputs[30] , channels / 2 , channels / 4 , batch_size , height / 2 , height , width , width );
	forward_depthwise_convolution( layer_inputs[31] , layer_outputs[31] , layer_parameters[31] , channels / 4 , channels / 4 , batch_size , height , width , 3 , 3 );
	forward_pointwise_convolution( layer_inputs[32] , layer_outputs[32] , layer_parameters[32] , channels / 4 , 3 , batch_size , height , width );
	forward_bias( layer_inputs[33] , layer_outputs[33] , layer_parameters[33] , 3 , 3 , batch_size , height , width );

	l2_forward( layer_outputs[33] , target , objective , 3 , batch_size , height , width ); //loss function
	if( forward_only ) {
		return;
	}
	l2_backward( layer_outputs[33] , target , layer_output_gradients[33] , 3 , batch_size , height , width ); //loss gradient

	//neural network backward pass (compute layer input gradients)
	backward_bias( layer_outputs[33] , layer_output_gradients[33] , layer_input_gradients[33] , layer_parameters[33] , 3 , 3 , batch_size , height , width );
	backward_pointwise_convolution( layer_outputs[32] , layer_output_gradients[32] , layer_input_gradients[32] , layer_parameters[32] , channels / 4 , 3 , batch_size , height , width );
	backward_depthwise_convolution( layer_outputs[31] , layer_output_gradients[31] , layer_input_gradients[31] , layer_parameters[31] , channels / 4 , channels / 4 , batch_size , height , width , 3 , 3 );
	backward_reshape( layer_outputs[30] , layer_output_gradients[30] , layer_input_gradients[30] , layer_parameters[30] , channels / 2 , channels / 4 , batch_size , height / 2 , height , width , width );
	backward_reshape( layer_outputs[29] , layer_output_gradients[29] , layer_input_gradients[29] , layer_parameters[29] , channels , channels / 2 , batch_size , height / 2 , height / 2 , width / 2 , width );
	for( i = 5 ; 0 <= i ; i-- ) {
		backward_nonlin_leaky_relu( layer_outputs[8+4*i] , layer_output_gradients[8+4*i] , layer_input_gradients[8+4*i] , layer_parameters[8+4*i] , channels , channels , batch_size , height / 2 , width / 2 );
		backward_bias( layer_outputs[7+4*i] , layer_output_gradients[7+4*i] , layer_input_gradients[7+4*i] , layer_parameters[7+4*i] , channels , channels , batch_size , height / 2 , width / 2 );
		backward_pointwise_convolution( layer_outputs[6+4*i] , layer_output_gradients[6+4*i] , layer_input_gradients[6+4*i] , layer_parameters[6+4*i] , channels , channels , batch_size , height / 2 , width / 2 );
		backward_depthwise_convolution( layer_outputs[5+4*i] , layer_output_gradients[5+4*i] , layer_input_gradients[5+4*i] , layer_parameters[5+4*i] , channels , channels , batch_size , height / 2 , width / 2  , 3 , 3 );
	}
	backward_nonlin_leaky_relu( layer_outputs[4] , layer_output_gradients[4] , layer_input_gradients[4] , layer_parameters[4] , channels , channels , batch_size , height / 2 , width / 2 );
	backward_bias( layer_outputs[3] , layer_output_gradients[3] , layer_input_gradients[3] , layer_parameters[3] , channels , channels , batch_size , height / 2 , width / 2 );
	backward_pointwise_convolution( layer_outputs[2] , layer_output_gradients[2] , layer_input_gradients[2] , layer_parameters[2] , 12 , channels , batch_size , height / 2 , width / 2 );
	backward_reshape( layer_outputs[1] , layer_output_gradients[1] , layer_input_gradients[1] , layer_parameters[1] , 6 , 12 , batch_size , height / 2 , height / 2 , width , width / 2 );
	backward_reshape( layer_outputs[0] , layer_output_gradients[0] , layer_input_gradients[0] , layer_parameters[0] , 3 , 6 , batch_size , height , height / 2 , width , width );
	if( no_param_gradient ) {
		return;
	}

	//compute parameter gradients of each layer of the neural network
	param_gradient_bias( layer_inputs[33] , layer_output_gradients[33] , layer_parameter_gradients[33] , 3 , 3 , batch_size , height , width );
	param_gradient_pointwise_convolution( layer_inputs[32] , layer_output_gradients[32] , layer_parameter_gradients[32] , channels / 4 , 3 , batch_size , height , width );
	param_gradient_depthwise_convolution( layer_inputs[31] , layer_output_gradients[31] , layer_parameter_gradients[31] , channels / 4 , channels / 4 , batch_size , height , width , 3 , 3 );
	for( i = 0 ; i < 6 ; i++ ) {
		param_gradient_bias( layer_inputs[7+4*i] , layer_output_gradients[7+4*i] , layer_parameter_gradients[7+4*i] , channels , channels , batch_size , height / 2 , width / 2 );
		param_gradient_pointwise_convolution( layer_inputs[6+4*i] , layer_output_gradients[6+4*i] , layer_parameter_gradients[6+4*i] , channels , channels , batch_size , height / 2 , width / 2 );
		param_gradient_depthwise_convolution( layer_inputs[5+4*i] , layer_output_gradients[5+4*i] , layer_parameter_gradients[5+4*i] , channels , channels , batch_size , height / 2 , width / 2 , 3 , 3 );
	}
	param_gradient_bias( layer_inputs[3] , layer_output_gradients[3] , layer_parameter_gradients[3] , channels , channels , batch_size , height / 2 , width / 2 );
	param_gradient_pointwise_convolution( layer_inputs[2] , layer_output_gradients[2] , layer_parameter_gradients[2] , 12 , channels , batch_size , height / 2 , width / 2 );
} // }}}

void set_up_buffers( Buffers& layer_inputs , Buffers& layer_outputs , Buffers& layer_input_gradients , Buffers& layer_output_gradients , Buffers& layer_parameters , Buffers& layer_parameter_gradients , Buffers& layer_parameter_gradients2 , Buffers& extra_data , Buffers& layer_parameter_updates , Buffers& layer_1st_moment_estimates , Buffers& layer_2nd_moment_estimates , Buffer& target , Buffer& objective , uint32_t channels , uint32_t batch_size , uint32_t height , uint32_t width , bool forward_only , bool no_param_gradient , bool set_up_thread_independent_buffers ) { // {{{
	uint32_t i;
	std::vector<std::tuple<uint32_t,uint32_t,uint32_t,uint32_t>> buffer_sizes = std::vector<std::tuple<uint32_t,uint32_t,uint32_t,uint32_t>>();
	buffer_sizes.push_back(buffer_sizes_reshape( 3 , 6 , batch_size , height , height / 2 , width , width ));
	buffer_sizes.push_back(buffer_sizes_reshape( 6 , 12 , batch_size , height / 2 , height / 2 , width , width / 2 ));
	buffer_sizes.push_back(buffer_sizes_pointwise_convolution( 12 , channels , batch_size , height / 2 , width / 2 ));
	buffer_sizes.push_back(buffer_sizes_bias( channels , channels , batch_size , height / 2 , width / 2 ));
	buffer_sizes.push_back(buffer_sizes_nonlin_leaky_relu( channels , channels , batch_size , height / 2 , width / 2 ));
	for( i = 0 ; i < 6 ; i++ ) {
		buffer_sizes.push_back(buffer_sizes_depthwise_convolution( channels , channels , batch_size , height / 2 , width / 2 , 3 , 3 ));
		buffer_sizes.push_back(buffer_sizes_pointwise_convolution(channels , channels , batch_size , height / 2 , width / 2));
		buffer_sizes.push_back(buffer_sizes_bias(channels , channels , batch_size , height / 2 , width / 2));
		buffer_sizes.push_back(buffer_sizes_nonlin_leaky_relu(channels , channels , batch_size , height / 2 , width / 2));
	}
	buffer_sizes.push_back(buffer_sizes_reshape( channels , channels / 2 , batch_size , height / 2 , height / 2 , width / 2 , width ));
	buffer_sizes.push_back(buffer_sizes_reshape( channels / 2 , channels / 4 , batch_size , height / 2 , height , width , width ));
	buffer_sizes.push_back(buffer_sizes_depthwise_convolution( channels / 4 , channels / 4 , batch_size , height , width , 3 , 3 ));
	buffer_sizes.push_back(buffer_sizes_pointwise_convolution( channels / 4 , 3 , batch_size , height , width ));
	buffer_sizes.push_back(buffer_sizes_bias( 3 , 3 , batch_size , height , width ));
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

void worker_thread_function( Buffers* layer_inputs , Buffers* layer_outputs , Buffers* layer_input_gradients , Buffers* layer_output_gradients , const Buffers* layer_parameters , Buffers* layer_parameter_gradients , Buffers* extra_data , Buffer target , Buffer objective , uint32_t channels , uint32_t batch_size , uint32_t height , uint32_t width , bool forward_only , bool no_param_gradient , uint32_t thread_nr ) {
	forward_and_backward( *layer_inputs , *layer_outputs , *layer_input_gradients , *layer_output_gradients , *layer_parameters , *layer_parameter_gradients , *extra_data , target , objective , channels , batch_size , height , width , forward_only , no_param_gradient );
}

void multithreaded_training_iteration( std::vector<Buffers>& thread_layer_inputs , std::vector<Buffers>& thread_layer_outputs , std::vector<Buffers>& thread_layer_input_gradients , std::vector<Buffers>& thread_layer_output_gradients , Buffers& layer_parameters , std::vector<Buffers>& thread_layer_parameter_gradients , Buffers& layer_parameter_gradients2 , std::vector<Buffers>& thread_extra_data , Buffers& layer_parameter_updates , Buffers& layer_1st_moment_estimates , Buffers& layer_2nd_moment_estimates , Buffers& thread_target , Buffers& thread_objective , uint32_t& iteration , uint32_t iterations , const SR_Data_Buffer sr_data_buffer , std::mt19937& rng , bool test_set , uint32_t channels , uint32_t batch_size , uint32_t effective_batch_size , uint32_t width , uint32_t height , float learning_rate ) { // {{{
	std::vector<std::unique_ptr<std::thread>> worker_threads = std::vector<std::unique_ptr<std::thread>>();
	uint32_t k,thread_nr,nr_threads;
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
	for( k = 0 ; k < effective_batch_size ; k += batch_size * nr_threads ) {
		for( thread_nr = 0 ; thread_nr < nr_threads ; thread_nr++ ) {
			zero_data(thread_layer_inputs.at(thread_nr));
			generate_data_and_targets( sr_data_buffer , rng , thread_layer_inputs.at(thread_nr).front() , thread_target.at(thread_nr) , batch_size );
			add( layer_parameter_gradients2 , thread_layer_parameter_gradients.at(thread_nr) );
			zero_data( thread_layer_parameter_gradients.at(thread_nr) );
			zero_data(thread_layer_outputs.at(thread_nr));
			zero_data(thread_layer_input_gradients.at(thread_nr));
			zero_data(thread_layer_output_gradients.at(thread_nr));
		}
		auto start = std::chrono::high_resolution_clock::now();
		for( thread_nr = 0 ; thread_nr < nr_threads ; thread_nr++ ) {
			std::cerr << ".";
			worker_threads.at(thread_nr) = std::unique_ptr<std::thread>( new std::thread( worker_thread_function ,  &(thread_layer_inputs.at(thread_nr)) , &(thread_layer_outputs.at(thread_nr)) , &(thread_layer_input_gradients.at(thread_nr)) , &(thread_layer_output_gradients.at(thread_nr)) , &layer_parameters , &(thread_layer_parameter_gradients.at(thread_nr)) , &(thread_extra_data.at(thread_nr)) , thread_target.at(thread_nr) , thread_objective.at(thread_nr) , channels , batch_size , height , width , false , false , thread_nr  ) );
		}
		for( std::unique_ptr<std::thread>& t : worker_threads ) {
			t->join();
		}
		auto elapsed = std::chrono::high_resolution_clock::now() - start;
		sum_microseconds += std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
		if( 0 == iteration % 10 ) {
			for( thread_nr = 0 ; thread_nr < nr_threads ; thread_nr++ ) {
				maximum_output = foldl( thread_layer_outputs.at(thread_nr).back() , maximum_output , [](uint32_t i,float a,float x){return (MAX(a,x));} );
			}
			for( thread_nr = 0 ; thread_nr < nr_threads ; thread_nr++ ) {
				average_obj += foldl( thread_objective.at(thread_nr) , 0.0 , [](uint32_t i,float a,float x){return (a+x);} ) / effective_batch_size;
			}
		}
	}
	scale( layer_parameter_gradients2 , 1.0/effective_batch_size );
	//sgd( layer_parameter_gradients2 , layer_parameter_updates , layer_1st_moment_estimates , layer_2nd_moment_estimates , iteration , learning_rate );
	adam( layer_parameter_gradients2 , layer_parameter_updates , layer_1st_moment_estimates , layer_2nd_moment_estimates , iteration , learning_rate );
	add( layer_parameters , layer_parameter_updates );
	parameter_norm_clipping( layer_parameters , 0.13 );
	if( 0 == iteration % 10 ) {
		std::cerr << std::endl << "iteration " << iteration << " (" << (static_cast<int>(round((1000.0*iteration)/iterations))/10.0) << " %):" << std::endl;
		std::cerr << "average objective of current batch = " << average_obj << " (this should go down during training)" << std::endl;
		std::cerr << "gradient norm = " << (norm(layer_parameter_gradients2)) << " (if this is 0 or explodes, you have a problem)" << std::endl;
		std::cerr << "update norm = " << (norm(layer_parameter_updates)) << " (ditto)" << std::endl;
		std::cerr << "parameter norm = " << (norm(layer_parameters)) << " (ditto)" << std::endl;
		std::cerr << "learning rate = " << learning_rate << std::endl;
		std::cerr << "maximum output = " << maximum_output << std::endl;
		std::cerr << "number parameters = " << (length(layer_parameters)) << std::endl;
		std::cerr << "FLOP/s is approximately " << ((1000000.0*6.0*(((float)width)*((float)width)*length(layer_parameters))*effective_batch_size)/sum_microseconds) << " (the bigger the better)" << std::endl;
		std::cerr << "compressed norm of parameter update for all layers: ";
		log_norms(layer_parameter_updates);
	}
} // }}}

void training_iteration( Buffers& layer_inputs , Buffers& layer_outputs , Buffers& layer_input_gradients , Buffers& layer_output_gradients , Buffers& layer_parameters , Buffers& layer_parameter_gradients , Buffers& layer_parameter_gradients2 , Buffers& extra_data , Buffers& layer_parameter_updates , Buffers& layer_1st_moment_estimates , Buffers layer_2nd_moment_estimates , Buffer target , Buffer objective , uint32_t& iteration , uint32_t iterations , const SR_Data_Buffer sr_data_buffer , std::mt19937& rng , bool test_set , uint32_t channels , uint32_t batch_size , uint32_t effective_batch_size , uint32_t width , uint32_t height , float learning_rate ) { // {{{
	uint32_t k;
	uint64_t sum_microseconds;
	float average_obj = 0.0;
	float maximum_output;
	zero_data(layer_parameter_gradients); //actual batch of gradients. smaller than effective batch size to save RAM
	zero_data(layer_parameter_gradients2); //sum many of those batches into a larger effective batch
	sum_microseconds = 0;
	for( k = 0 ; k < effective_batch_size ; k += batch_size ) {
		zero_data(layer_inputs);
		generate_data_and_targets( sr_data_buffer , rng , layer_inputs.front() , target , batch_size );
		//TODO: why did i add this before forward_and_backward? that way i'm losing the gradient of the last loop iteration
		add( layer_parameter_gradients2 , layer_parameter_gradients );
		zero_data( layer_parameter_gradients );
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
			average_obj += foldl( objective , 0.0 , [](uint32_t i,float a,float x){return (a+x);} ) / effective_batch_size;
		}
	}
	scale( layer_parameter_gradients2 , 1.0/effective_batch_size );
	//sgd( layer_parameter_gradients2 , layer_parameter_updates , layer_1st_moment_estimates , layer_2nd_moment_estimates , iteration , learning_rate );
	adam( layer_parameter_gradients2 , layer_parameter_updates , layer_1st_moment_estimates , layer_2nd_moment_estimates , iteration , learning_rate );
	add( layer_parameters , layer_parameter_updates );
	parameter_norm_clipping( layer_parameters , 0.13 );
	if( 0 == iteration % 10 ) {
		std::cerr << std::endl << "iteration " << iteration << " (" << (static_cast<int>(round((1000.0*iteration)/iterations))/10.0) << " %):" << std::endl;
		std::cerr << "average objective of current batch = " << average_obj << " (this should go down during training)" << std::endl;
		std::cerr << "gradient norm = " << (norm(layer_parameter_gradients2)) << " (if this is 0 or explodes, you have a problem)" << std::endl;
		std::cerr << "update norm = " << (norm(layer_parameter_updates)) << " (ditto)" << std::endl;
		std::cerr << "parameter norm = " << (norm(layer_parameters)) << " (ditto)" << std::endl;
		std::cerr << "learning rate = " << learning_rate << std::endl;
		std::cerr << "maximum output = " << maximum_output << std::endl;
		std::cerr << "FLOP/s is approximately " << ((1000000.0*6.0*(((float)width)*((float)width)*length(layer_parameters))*effective_batch_size)/sum_microseconds) << " (the bigger the better)" << std::endl;
		std::cerr << "compressed norm of parameter update for all layers: ";
		log_norms(layer_parameter_updates);
	}
} // }}}

void train( Buffers& layer_parameters , const std::string& dirname ) { // {{{
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
	SR_Data_Buffer sr_data_buffer;
	uint32_t channels = 128;
	uint32_t batch_size = 4;
	uint32_t effective_batch_size = 48;
	uint32_t height = 64;
	uint32_t width = 64;
	uint32_t iterations = 2000;
	rng.seed( (uint_fast32_t) time( NULL ) );
	set_up_buffers( layer_inputs , layer_outputs , layer_input_gradients , layer_output_gradients , layer_parameters , layer_parameter_gradients , layer_parameter_gradients2 , extra_data , layer_parameter_updates , layer_1st_moment_estimates , layer_2nd_moment_estimates , target , objective , channels , batch_size , height , width , false , false , true );
	sr_data_buffer = load_super_resolution_data(dirname);
	gaussian_init_parameters( layer_parameters , rng , 0.13 );
	for( iteration = 0 ; iteration < iterations ; iteration++ ) {
		training_iteration( layer_inputs , layer_outputs , layer_input_gradients , layer_output_gradients , layer_parameters , layer_parameter_gradients , layer_parameter_gradients2 , extra_data , layer_parameter_updates , layer_1st_moment_estimates , layer_2nd_moment_estimates , target , objective , iteration , iterations , sr_data_buffer , rng , false , channels , batch_size , effective_batch_size , width , height , learning_rate_schedule( iteration , iterations ) );
	}
} // }}}

void train_multithreaded( Buffers& layer_parameters , const std::string& dirname , uint32_t nr_threads ) { // {{{
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
	SR_Data_Buffer sr_data_buffer;
	uint32_t channels = 128;
	uint32_t batch_size = 4;
	uint32_t effective_batch_size = 48;
	uint32_t height = 64;
	uint32_t width = 64;
	uint32_t iterations = 2000;
	rng.seed( (uint_fast32_t) time( NULL ) );
	thread_target.resize(nr_threads);
	thread_objective.resize(nr_threads);
	thread_layer_inputs.resize(nr_threads);
	thread_layer_outputs.resize(nr_threads);
	thread_layer_input_gradients.resize(nr_threads);
	thread_layer_output_gradients.resize(nr_threads);
	thread_layer_parameter_gradients.resize(nr_threads);
	thread_extra_data.resize(nr_threads);
	for( uint32_t i = 0 ; i < nr_threads ; i++ ) {
		set_up_buffers( thread_layer_inputs.at(i) , thread_layer_outputs.at(i) , thread_layer_input_gradients.at(i) , thread_layer_output_gradients.at(i) , layer_parameters , thread_layer_parameter_gradients.at(i) , layer_parameter_gradients2 , thread_extra_data.at(i) , layer_parameter_updates , layer_1st_moment_estimates , layer_2nd_moment_estimates , thread_target.at(i) , thread_objective.at(i) , channels , batch_size , height , width , false , false , ( 0 == i ) );
	}
	sr_data_buffer = load_super_resolution_data(dirname);
	gaussian_init_parameters( layer_parameters , rng , 0.13 );
	for( iteration = 0 ; iteration < iterations ; iteration++ ) {
		multithreaded_training_iteration( thread_layer_inputs , thread_layer_outputs , thread_layer_input_gradients , thread_layer_output_gradients , layer_parameters , thread_layer_parameter_gradients , layer_parameter_gradients2 , thread_extra_data , layer_parameter_updates , layer_1st_moment_estimates , layer_2nd_moment_estimates , thread_target , thread_objective , iteration , iterations , sr_data_buffer , rng , false , channels , batch_size , effective_batch_size , width , height , learning_rate_schedule( iteration , iterations ) );
	}
} // }}}

void test( Buffers& layer_parameters , const std::string& dirname ) { // {{{
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
	SR_Data_Buffer sr_data_buffer;
	uint32_t channels = 128;
	uint32_t batch_size = 1;
	uint32_t iterations = 2000;
	uint32_t height = 64;
	uint32_t width = 64;
	float average_obj = 0.0;
	float maximum_output = std::numeric_limits<float>::lowest();
	uint64_t sum_microseconds;
	double objective_2nd_moment; //needs to be double precision to reduce rounding error
	rng.seed( (uint_fast32_t) time( NULL ) );
	set_up_buffers( layer_inputs , layer_outputs , layer_input_gradients , layer_output_gradients , dummy , layer_parameter_gradients , layer_parameter_gradients2 , extra_data , layer_parameter_updates , layer_1st_moment_estimates , layer_2nd_moment_estimates , target , objective , channels , batch_size , height , width , true , true , true );
	sr_data_buffer = load_super_resolution_data(dirname);
	sum_microseconds = 0;
	objective_2nd_moment = 0;
	for( iteration = 0 ; iteration < iterations ; iteration += batch_size ) {
		zero_data(layer_inputs);
		zero_data(layer_outputs);
		generate_data_and_targets( sr_data_buffer , rng , layer_inputs.front() , target , batch_size );
		for( i = 0 ; i < batch_size ; i++ ) {
			std::cerr << ".";
		}
		auto start = std::chrono::high_resolution_clock::now();
		forward_and_backward( layer_inputs , layer_outputs , layer_input_gradients , layer_output_gradients , layer_parameters , layer_parameter_gradients , extra_data , target , objective , channels , batch_size , height , width , true , true );
		auto elapsed = std::chrono::high_resolution_clock::now() - start;
		sum_microseconds += std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
		maximum_output = foldl( layer_outputs.back() , maximum_output , [](uint32_t i,float a,float x){return (MAX(a,x));} );
		average_obj += foldl( objective , 0.0 , [](uint32_t i,float a,float x){return (a+x);} );
		objective_2nd_moment += foldl( objective , 0.0 , [](uint32_t i,float a,float x){return (a+x*x);} );
	}
	average_obj /= iterations;
	objective_2nd_moment /= iterations;
	std::cerr << std::endl;
	std::cerr << "average objective = " << average_obj <<  " standard deviation = " << (sqrt( ( objective_2nd_moment - average_obj * average_obj ) / iterations )) << std::endl;
	std::cerr << "maximum output = " << maximum_output << std::endl;
	std::cerr << "FLOP/s is approximately " << ((1000000.0*2.0*(((float)width)*((float)width)*length(layer_parameters))*batch_size*iterations)/sum_microseconds) << std::endl;
} // }}}

int main( int argc, char** argv ) {
	if( 3 != argc ) {
		std::cerr << "need 2 directories full of PNG images, 1 for training and 1 for testing" << std::endl;
		return 1;
	}
	Buffers layer_parameters;
	//train_multithreaded( layer_parameters , std::string(argv[1]) , 6 );
	train( layer_parameters , std::string(argv[1]) );
	std::ofstream parameter_file = std::ofstream( "extra_tiny_super_resolution_parameters.out" , std::ofstream::out | std::ofstream::binary );
	if( parameter_file.bad() ) {
		throw std::length_error( std::string( CURRENT_FUNCTION_NAME ) + ": error while trying to open extra_tiny_super_resolution_parameters.out" );
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
		std::cerr << "error while trying to write to extra_tiny_super_resolution_parameters.out because of: " << std::endl;
		throw;
	}
	std::cerr << "training set evaluation:" << std::endl;
	test( layer_parameters , std::string(argv[1]) );
	std::cerr << "test set evaluation:" << std::endl;
	test( layer_parameters , std::string(argv[2]) );
	return 0;
}

float clamp( float x ) {
	return (MIN(1.0,MAX(0.0,x)));
}

////apply (a.k.a. inference) neural network stored in extra_tiny_super_resolution_parameters.out to a single PNG image
//int main( int argc, char** argv ) {
//	if( 2 != argc ) {
//		std::cerr << "gimme a PNG file please!" << std::endl;
//		return 1;
//	}
//	Buffers layer_parameters;
//	std::ifstream parameter_file = std::ifstream( "extra_tiny_super_resolution_parameters.out" , std::ifstream::in | std::ifstream::binary );
//	if( parameter_file.bad() ) {
//		throw std::length_error( std::string( CURRENT_FUNCTION_NAME ) + ": error while trying to open extra_tiny_super_resolution_parameters.out" );
//	}
//	uint32_t temp,row,col,row_offset,col_offset;
//	try {
//		parameter_file.read( (char*) &temp , sizeof(uint32_t) );
//		layer_parameters.resize(temp);
//		for( Buffer& buffer : layer_parameters ) {
//			parameter_file.read( (char*) &temp , sizeof(uint32_t) );
//			buffer = Buffer(new std::vector<float>(temp));
//			if( 0 < buffer->size() ) {
//				parameter_file.read( (char*) (buffer->data()) , sizeof(float) * buffer->size() );
//			}
//		}
//	} catch(...) {
//		std::cerr << "error while trying to read from extra_tiny_super_resolution_parameters.out because of: " << std::endl;
//		throw;
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
//	Buffers dummy;
//	Buffer objective;
//	Buffer target;
//	uint32_t channels = 128;
//	uint32_t batch_size = 1;
//	uint32_t height = 64;
//	uint32_t width = 64;
//	set_up_buffers( layer_inputs , layer_outputs , layer_input_gradients , layer_output_gradients , dummy , layer_parameter_gradients , layer_parameter_gradients2 , extra_data , layer_parameter_updates , layer_1st_moment_estimates , layer_2nd_moment_estimates , target , objective , channels , batch_size , height , width , true , true , true );
//	struct png_image_t input_image = read_png_file( std::string(argv[1]) );
//	struct png_image_t output_image;
//	//this means the output will not be exactly twice the input size. some pixels on the bottom & right will be cut off.
//	//it makes for an easier implementation (no overlapping patches, no non-32x32-pixel-patches).
//	output_image.width = 64 * ( input_image.width / 32 );
//	output_image.height = 64 * ( input_image.height / 32 );
//	output_image.color_type = input_image.color_type;
//	output_image.bit_depth = input_image.bit_depth;
//	output_image.row_pointers = static_cast<png_bytep*>(malloc(sizeof(png_bytep) * output_image.height));
//	for( row = 0 ; row < static_cast<uint32_t>(output_image.height) ; row++ ) {
//		//NB: this is a hack. you actually need png_get_rowbytes() to tell you how many bytes are in a row -_- . i'm gonna guess this. don't do this at home.
//		output_image.row_pointers[row] = static_cast<png_bytep>(malloc( 4 * sizeof(png_byte) * output_image.width ));
//	}
//	png_bytep rowp = NULL;
//	png_bytep px = NULL;
//	//this goes through the input image in 32x32-pixel-patches (64x64 on the output side). processing the entire input image at once
//	//is possible, but requires unreasonable amounts of RAM (approximately number of pixels times number of channels times number of layers time sizeof(float)).
//	for( row_offset = 0 ; row_offset < static_cast<uint32_t>(input_image.height/32) ; row_offset++ ) {
//		for( col_offset = 0 ; col_offset < static_cast<uint32_t>(input_image.width/32) ; col_offset++ ) {
//			//NB: if the down-/upsampling method is changed, the entire body of this loop needs to be replaced.
//			for( row = 0 ; row < 64 ; row++ ) {
//				rowp = input_image.row_pointers[(64*row_offset+row)/2]; //every 2nd input row will be the same
//				for( col = 0 ; col < 64 ; col++ ) {
//					px = &(rowp[((64*col_offset+col)/2) * 4]); //ditto for every 2nd input column
//					layer_inputs[0]->at( 0 * batch_size * 64 * 64 + row * 64 + col ) = px[0] / 255.0;
//					layer_inputs[0]->at( 1 * batch_size * 64 * 64 + row * 64 + col ) = px[1] / 255.0;
//					layer_inputs[0]->at( 2 * batch_size * 64 * 64 + row * 64 + col ) = px[2] / 255.0;
//				}
//			}
//			forward_and_backward( layer_inputs , layer_outputs , layer_input_gradients , layer_output_gradients , layer_parameters , layer_parameter_gradients , extra_data , target , objective , channels , batch_size , height , width , true , true );
//			for( row = 0 ; row < 64 ; row++ ) {
//				rowp = output_image.row_pointers[64*row_offset+row];
//				for( col = 0 ; col < 64 ; col++ ) {
//					px = &(rowp[(64*col_offset+col) * 4]);
//					//the target for the network during training was the residual, so now we have to add the output of the network to its
//					//input to get the final upsampled image.
//					px[0] = static_cast<png_byte>( round( 255.0 * ( clamp(	layer_inputs[0]->at( 0 * batch_size * 64 * 64 + row * 64 + col ) +
//												layer_outputs.back()->at( 0 * batch_size * 64 * 64 + row * 64 + col ) ) ) ) );
//					px[1] = static_cast<png_byte>( round( 255.0 * ( clamp(	layer_inputs[0]->at( 1 * batch_size * 64 * 64 + row * 64 + col ) +
//												layer_outputs.back()->at( 1 * batch_size * 64 * 64 + row * 64 + col ) ) ) ) );
//					px[2] = static_cast<png_byte>( round( 255.0 * ( clamp(	layer_inputs[0]->at( 2 * batch_size * 64 * 64 + row * 64 + col ) +
//												layer_outputs.back()->at( 2 * batch_size * 64 * 64 + row * 64 + col ) ) ) ) );
//				}
//			}
//			std::cerr << ".";
//		}
//	}
//	std::cerr << std::endl;
//	write_png_file( output_image , std::string(argv[1])+"_superresolved.png" );
//	std::cerr << "result written to " << (std::string(argv[1])) << "_superresolved.png" << std::endl;
//	return 0;
//}

// vim: ts=8 : autoindent : textwidth=0 : foldmethod=marker
