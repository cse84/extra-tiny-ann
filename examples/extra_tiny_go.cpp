// Copyright 2024 Daniel Müller
//
// Licensed under the terms given in the file named "LICENSE"
// in the root folder of the project

#include <fstream>
#include <sstream>
#include <thread>
#include "../extra_tiny_ann.hpp"

//NB: to be able to run this, you need to download the training data from
//https://github.com/cse84/extra-tiny-go-dataset (download
//extra-tiny-go-dataset.tar.gz to this directory
//and execute "tar -xzf extra-tiny-go-dataset.tar.gz").

//TODO: when this was lower, training loss exploded, which implies problems in the training data (which would be unsurprising
//       if training data were to be sampled from unitialized memory). but my code is written such that a lower number here should
//       just lead to data being sampled from a smaller subset, which is not a problem (apart from the risk of overfitting),
//       so there must be a bug.
#define NR_GO_TRAIN_IMAGES 1000000
#define NR_GO_TEST_IMAGES 100000

#pragma pack(push, 1)
struct go_t {
	uint8_t target_row;
	uint8_t target_col;
	uint8_t pixels[19][19];
};
#pragma pack(pop)

typedef std::shared_ptr<std::vector<struct go_t>> GO_Buffer;

GO_Buffer load_go_data( bool test_set ) { // {{{
	uint_fast32_t nr_images = test_set?NR_GO_TEST_IMAGES:NR_GO_TRAIN_IMAGES;
	std::string filename = test_set?"extra-tiny-go-dataset.test":"extra-tiny-go-dataset.train";
	GO_Buffer go_images = GO_Buffer( new std::vector<struct go_t>( nr_images ) );
	std::ifstream go_file = std::ifstream( filename , std::ifstream::in | std::ifstream::binary );
	if( go_file.bad() ) {
		throw std::length_error( std::string( CURRENT_FUNCTION_NAME ) + ": error while trying to open " + filename );
	}
	try {
		go_file.read( (char*) (go_images->data()) , sizeof(struct go_t) * nr_images );
	} catch(...) {
		std::cerr << "error while trying to read " << filename << " because of: " << std::endl;
		throw;
	}
	return go_images;
} // }}}

void generate_data_and_targets( const GO_Buffer go_images , std::mt19937& rng , bool test_set , Buffer data , Buffer target , uint32_t batch_size , int32_t index = -1 ) { // {{{
	uint_fast32_t i,j,k,r;
	zero_data( target );
	for( i = 0 ; i < batch_size ; i++ ) {
		if( -1 == index ) {
			r = rng() % (test_set?NR_GO_TEST_IMAGES:NR_GO_TRAIN_IMAGES);
		} else {
			r = index;
		}
		for( j = 0 ; j < 19 ; j++ ) {
			for( k = 0 ; k < 19 ; k++ ) {
				data->at( 0 * batch_size * 19 * 19 + i * 19 * 19 + j * 19 + k ) = 0.0;
				data->at( 1 * batch_size * 19 * 19 + i * 19 * 19 + j * 19 + k ) = 0.0;
				data->at( 2 * batch_size * 19 * 19 + i * 19 * 19 + j * 19 + k ) = 0.0;
				data->at( (go_images->at(r).pixels[j][k]) * batch_size * 19 * 19 + i * 19 * 19 + j * 19 + k ) = 1.0;
			}
		}
		target->at( i * 19 * 19 + (go_images->at(r).target_row) * 19 + (go_images->at(r).target_col) ) = 1;
	}
} // }}}

void forward_and_backward( Buffers& layer_inputs , Buffers& layer_outputs , Buffers& layer_input_gradients , Buffers& layer_output_gradients , const Buffers& layer_parameters , Buffers& layer_parameter_gradients , Buffers& extra_data , Buffer target , Buffer objective , uint32_t channels , uint32_t batch_size , uint32_t height , uint32_t width , bool forward_only , bool no_param_gradient ) { // {{{
	int32_t i;
	//neural network forward pass (compute layer outputs)
	forward_pointwise_convolution( layer_inputs.at(0) , layer_outputs.at(0) , layer_parameters.at(0) , 3 , channels , batch_size , height , width );
	for( i = 0 ; i < 14 ; i++ ) {
		forward_depthwise_convolution( layer_inputs.at(1+4*i) , layer_outputs.at(1+4*i) , layer_parameters.at(1+4*i) , channels , channels , batch_size , height , width , 3 , 3 );
		forward_pointwise_convolution( layer_inputs.at(2+4*i) , layer_outputs.at(2+4*i) , layer_parameters.at(2+4*i) , channels , channels , batch_size , height , width );
		forward_bias( layer_inputs.at(3+4*i) , layer_outputs.at(3+4*i) , layer_parameters.at(3+4*i) , channels , channels , batch_size , height , width );
		forward_nonlin_leaky_relu( layer_inputs.at(4+4*i) , layer_outputs.at(4+4*i) , channels , channels , batch_size , height , width );
	}
	forward_pointwise_convolution( layer_inputs.at(57) , layer_outputs.at(57) , layer_parameters.at(57) , channels , 1 , batch_size , height , width );
	forward_nonlin_poly_exp( layer_inputs.at(58) , layer_outputs.at(58) , 1 , 1 , batch_size , height , width );
	//forward_nonlin_exp( layer_inputs.at(58) , layer_outputs.at(58) , 1 , 1 , batch_size , height , width );
	forward_channel_normalization( layer_inputs.at(59) , layer_outputs.at(59) , 1 , 1 , batch_size , height , width , extra_data.at(59) );

	cross_entropy_forward( layer_outputs.at(59) , target , objective , 1 , batch_size , height , width ); //loss function
	if( forward_only ) {
		return;
	}
	cross_entropy_backward( layer_outputs.at(59) , target , layer_output_gradients.at(59) , 1 , batch_size , height , width ); //loss gradient

	//neural network backward pass (compute layer input gradients)
	backward_channel_normalization( layer_outputs.at(59) , layer_output_gradients.at(59) , layer_input_gradients.at(59) , layer_parameters.at(59) , 1 , 1 , batch_size , height , width , extra_data.at(59) );
	backward_nonlin_poly_exp( layer_outputs.at(58) , layer_output_gradients.at(58) , layer_input_gradients.at(58) , layer_parameters.at(58) , 1 , 1 , batch_size , height , width );
	//backward_nonlin_exp( layer_outputs.at(58) , layer_output_gradients.at(58) , layer_input_gradients.at(58) , layer_parameters.at(58) , 1 , 1 , batch_size , height , width );
	backward_pointwise_convolution( layer_outputs.at(57) , layer_output_gradients.at(57) , layer_input_gradients.at(57) , layer_parameters.at(57) , channels , 1 , batch_size , height , width );
	for( i = 13 ; 0 <= i ; i-- ) {
		backward_nonlin_leaky_relu( layer_outputs.at(4+4*i) , layer_output_gradients.at(4+4*i) , layer_input_gradients.at(4+4*i) , layer_parameters.at(4+4*i) , channels , channels , batch_size , height , width );
		backward_bias( layer_outputs.at(3+4*i) , layer_output_gradients.at(3+4*i) , layer_input_gradients.at(3+4*i) , layer_parameters.at(3+4*i) , channels , channels , batch_size , height , width );
		backward_pointwise_convolution( layer_outputs.at(2+4*i) , layer_output_gradients.at(2+4*i) , layer_input_gradients.at(2+4*i) , layer_parameters.at(2+4*i) , channels , channels , batch_size , height , width );
		backward_depthwise_convolution( layer_outputs.at(1+4*i) , layer_output_gradients.at(1+4*i) , layer_input_gradients.at(1+4*i) , layer_parameters.at(1+4*i) , channels , channels , batch_size , height , width  , 3 , 3);
	}
	backward_pointwise_convolution( layer_outputs.at(0) , layer_output_gradients.at(0) , layer_input_gradients.at(0) , layer_parameters.at(0) , 3 , channels , batch_size , height , width );
	if( no_param_gradient ) {
		return;
	}

	//compute parameter gradients of each layer of the neural network
	param_gradient_pointwise_convolution( layer_inputs.at(57) , layer_output_gradients.at(57) , layer_parameter_gradients.at(57) , channels , 1 , batch_size , height , width );
	for( i = 0 ; i < 14 ; i++ ) {
		param_gradient_bias( layer_inputs.at(3+4*i) , layer_output_gradients.at(3+4*i) , layer_parameter_gradients.at(3+4*i) , channels , channels , batch_size , height , width );
		param_gradient_pointwise_convolution( layer_inputs.at(2+4*i) , layer_output_gradients.at(2+4*i) , layer_parameter_gradients.at(2+4*i) , channels , channels , batch_size , height , width );
		param_gradient_depthwise_convolution( layer_inputs.at(1+4*i) , layer_output_gradients.at(1+4*i) , layer_parameter_gradients.at(1+4*i) , channels , channels , batch_size , height , width , 3 , 3 );
	}
	param_gradient_pointwise_convolution( layer_inputs.at(0) , layer_output_gradients.at(0) , layer_parameter_gradients.at(0) , 3 , channels , batch_size , height , width );
} // }}}

void set_up_buffers( Buffers& layer_inputs , Buffers& layer_outputs , Buffers& layer_input_gradients , Buffers& layer_output_gradients , Buffers& layer_parameters , Buffers& layer_parameter_gradients , Buffers& layer_parameter_gradients2 , Buffers& extra_data , Buffers& layer_parameter_updates , Buffers& layer_1st_moment_estimates , Buffers& layer_2nd_moment_estimates , Buffer& target , Buffer& objective , uint32_t channels , uint32_t batch_size , uint32_t height , uint32_t width , bool forward_only , bool no_param_gradient , bool set_up_thread_independent_buffers ) { // {{{
	uint32_t i;
	std::vector<std::tuple<uint32_t,uint32_t,uint32_t,uint32_t>> buffer_sizes = std::vector<std::tuple<uint32_t,uint32_t,uint32_t,uint32_t>>();
	buffer_sizes.push_back(buffer_sizes_pointwise_convolution( 3 , channels , batch_size , height , width ));
	for( i = 0 ; i < 14 ; i++ ) {
		buffer_sizes.push_back(buffer_sizes_depthwise_convolution( channels , channels , batch_size , height , width , 3 , 3 ));
		buffer_sizes.push_back(buffer_sizes_pointwise_convolution(channels , channels , batch_size , height , width));
		buffer_sizes.push_back(buffer_sizes_bias(channels , channels , batch_size , height , width));
		buffer_sizes.push_back(buffer_sizes_nonlin_leaky_relu(channels , channels , batch_size , height , width));
	}
	buffer_sizes.push_back(buffer_sizes_pointwise_convolution(channels , 1 , batch_size , height , width));
	buffer_sizes.push_back(buffer_sizes_nonlin_poly_exp(1 , 1 , batch_size , height , width));
	//buffer_sizes.push_back(buffer_sizes_nonlin_exp(1 , 1 , batch_size , height , width));
	buffer_sizes.push_back(buffer_sizes_channel_normalization(1 , 1 , batch_size , height , width));
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

void multithreaded_training_iteration( std::vector<Buffers>& thread_layer_inputs , std::vector<Buffers>& thread_layer_outputs , std::vector<Buffers>& thread_layer_input_gradients , std::vector<Buffers>& thread_layer_output_gradients , Buffers& layer_parameters , std::vector<Buffers>& thread_layer_parameter_gradients , Buffers& layer_parameter_gradients2 , std::vector<Buffers>& thread_extra_data , Buffers& layer_parameter_updates , Buffers& layer_1st_moment_estimates , Buffers& layer_2nd_moment_estimates , Buffers& thread_target , Buffers& thread_objective , uint32_t& iteration , uint32_t iterations , const GO_Buffer go_images , std::mt19937& rng , bool test_set , uint32_t channels , uint32_t batch_size , uint32_t effective_batch_size , uint32_t width , uint32_t height , float learning_rate ) { // {{{
	std::vector<std::unique_ptr<std::thread>> worker_threads = std::vector<std::unique_ptr<std::thread>>();
	uint32_t i,j,k,thread_nr,nr_threads;
	uint64_t sum_microseconds;
	float average_obj = 0.0;
	float maximum_output = std::numeric_limits<float>::lowest();
	uint32_t output_max_index,target_max_index;
	float temp;
	float accuracy = 0;
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
			generate_data_and_targets( go_images , rng , test_set , thread_layer_inputs.at(thread_nr).front() , thread_target.at(thread_nr) , batch_size );
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
			for( thread_nr = 0 ; thread_nr < nr_threads ; thread_nr++ ) {
				for( j = 0 ; j < batch_size ; j++ ) {
					temp = std::numeric_limits<float>::lowest();
					output_max_index = height * width;
					for( i = 0 ; i < height * width ; i++ ) {
						if( temp < thread_layer_outputs.at(thread_nr).back()->at( j * height * width + i ) ) {
							temp = thread_layer_outputs.at(thread_nr).back()->at( j * height * width + i );
							output_max_index = i;
						}
					}
					temp = std::numeric_limits<float>::lowest();
					target_max_index = height * width;
					for( i = 0 ; i < height * width ; i++ ) {
						if( temp < thread_target.at(thread_nr)->at( j * height * width + i ) ) {
							temp = thread_target.at(thread_nr)->at( j * height * width + i );
							target_max_index = i;
						}
					}
					accuracy += (output_max_index==target_max_index)?1:0;
				}
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
		std::cerr << "accuracy of current batch = " << (accuracy/effective_batch_size) << " (this should go up during training)" << std::endl;
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

void training_iteration( Buffers& layer_inputs , Buffers& layer_outputs , Buffers& layer_input_gradients , Buffers& layer_output_gradients , Buffers& layer_parameters , Buffers& layer_parameter_gradients , Buffers& layer_parameter_gradients2 , Buffers& extra_data , Buffers& layer_parameter_updates , Buffers& layer_1st_moment_estimates , Buffers layer_2nd_moment_estimates , Buffer target , Buffer objective , uint32_t& iteration , uint32_t iterations , const GO_Buffer go_images , std::mt19937& rng , bool test_set , uint32_t channels , uint32_t batch_size , uint32_t effective_batch_size , uint32_t width , uint32_t height , float learning_rate ) { // {{{
	uint32_t i,j,k;
	uint64_t sum_microseconds;
	float average_obj = 0.0;
	float maximum_output;
	uint32_t output_max_index,target_max_index;
	float temp;
	float accuracy = 0;
	zero_data(layer_parameter_gradients); //actual batch of gradients. smaller than effective batch size to save RAM
	zero_data(layer_parameter_gradients2); //sum many of those batches into a larger effective batch
	sum_microseconds = 0;
	for( k = 0 ; k < effective_batch_size ; k += batch_size ) {
		zero_data(layer_inputs);
		generate_data_and_targets( go_images , rng , test_set , layer_inputs.front() , target , batch_size );
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
			for( j = 0 ; j < batch_size ; j++ ) {
				temp = std::numeric_limits<float>::lowest();
				output_max_index = height * width;
				for( i = 0 ; i < height * width ; i++ ) {
					if( temp < layer_outputs.back()->at( j * height * width + i ) ) {
						temp = layer_outputs.back()->at( j * height * width + i );
						output_max_index = i;
					}
				}
				temp = std::numeric_limits<float>::lowest();
				target_max_index = height * width;
				for( i = 0 ; i < height * width ; i++ ) {
					if( temp < target->at( j * height * width + i ) ) {
						temp = target->at( j * height * width + i );
						target_max_index = i;
					}
				}
				accuracy += (output_max_index==target_max_index)?1:0;
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
		std::cerr << "accuracy of current batch = " << (accuracy/effective_batch_size) << " (this should go up during training)" << std::endl;
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
	GO_Buffer go_images;
	uint32_t channels = 64;
	uint32_t batch_size = 4;
	uint32_t effective_batch_size = 240;
	uint32_t height = 19;
	uint32_t width = 19;
	uint32_t iterations = 25000;
	rng.seed( (uint_fast32_t) time( NULL ) );
	set_up_buffers( layer_inputs , layer_outputs , layer_input_gradients , layer_output_gradients , layer_parameters , layer_parameter_gradients , layer_parameter_gradients2 , extra_data , layer_parameter_updates , layer_1st_moment_estimates , layer_2nd_moment_estimates , target , objective , channels , batch_size , height , width , false , false , true );
	go_images = load_go_data( false );
	gaussian_init_parameters( layer_parameters , rng , 0.13 );
	for( iteration = 0 ; iteration < iterations ; iteration++ ) {
		training_iteration( layer_inputs , layer_outputs , layer_input_gradients , layer_output_gradients , layer_parameters , layer_parameter_gradients , layer_parameter_gradients2 , extra_data , layer_parameter_updates , layer_1st_moment_estimates , layer_2nd_moment_estimates , target , objective , iteration , iterations , go_images , rng , false , channels , batch_size , effective_batch_size , width , height , learning_rate_schedule( iteration , iterations ) );
	}
} // }}}

void train_multithreaded( Buffers& layer_parameters , uint32_t nr_threads ) { // {{{
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
	GO_Buffer go_images;
	uint32_t channels = 64;
	uint32_t batch_size = 4;
	uint32_t effective_batch_size = 240;
	uint32_t height = 19;
	uint32_t width = 19;
	uint32_t iterations = 25000;
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
	go_images = load_go_data( false );
	gaussian_init_parameters( layer_parameters , rng , 0.13 );
	for( iteration = 0 ; iteration < iterations ; iteration++ ) {
		multithreaded_training_iteration( thread_layer_inputs , thread_layer_outputs , thread_layer_input_gradients , thread_layer_output_gradients , layer_parameters , thread_layer_parameter_gradients , layer_parameter_gradients2 , thread_extra_data , layer_parameter_updates , layer_1st_moment_estimates , layer_2nd_moment_estimates , thread_target , thread_objective , iteration , iterations , go_images , rng , false , channels , batch_size , effective_batch_size , width , height , learning_rate_schedule( iteration , iterations ) );
	}
} // }}}

void test( Buffers& layer_parameters , bool test_set ) { // {{{
	uint32_t i,j,iteration;
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
	GO_Buffer go_images;
	uint32_t channels = 64;
	uint32_t batch_size = 4;
	uint32_t iterations = 10000;
	uint32_t height = 19;
	uint32_t width = 19;
	float average_obj = 0.0;
	float accuracy = 0;
	float maximum_output = std::numeric_limits<float>::lowest();
	uint64_t sum_microseconds;
	uint32_t output_max_index,target_max_index;
	float temp;
	double objective_2nd_moment; //needs to be double precision to reduce rounding error
	rng.seed( (uint_fast32_t) time( NULL ) );
	set_up_buffers( layer_inputs , layer_outputs , layer_input_gradients , layer_output_gradients , dummy , layer_parameter_gradients , layer_parameter_gradients2 , extra_data , layer_parameter_updates , layer_1st_moment_estimates , layer_2nd_moment_estimates , target , objective , channels , batch_size , height , width , true , true , true );
	go_images = load_go_data( test_set );
	sum_microseconds = 0;
	objective_2nd_moment = 0;
	for( iteration = 0 ; iteration < iterations ; iteration += batch_size ) {
		zero_data(layer_inputs);
		zero_data(layer_outputs);
		generate_data_and_targets( go_images , rng , test_set , layer_inputs.front() , target , batch_size , iteration );
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
		for( j = 0 ; j < batch_size ; j++ ) {
			temp = std::numeric_limits<float>::lowest();
			output_max_index = height * width;
			for( i = 0 ; i < height * width ; i++ ) {
				if( temp < layer_outputs.back()->at( j * height * width + i ) ) {
					temp = layer_outputs.back()->at( j * height * width + i );
					output_max_index = i;
				}
			}
			temp = std::numeric_limits<float>::lowest();
			target_max_index = height * width;
			for( i = 0 ; i < height * width ; i++ ) {
				if( temp < target->at( j * height * width + i ) ) {
					temp = target->at( j * height * width + i );
					target_max_index = i;
				}
			}
			accuracy += (output_max_index==target_max_index)?1:0;
		}
	}
	average_obj /= iterations;
	accuracy /= iterations;
	objective_2nd_moment /= iterations;
	//how and why are these standard deviations calculated? while the same data samples are used every time for easier comparability, the point of this function
	//is to measure the quality of the neural network on a data set that is representative of all data, even unseen data. that means we must assume each of these iterations
	//to be an independent experiment on a randomly drawn sample, i.e. this function is like a series of coin flips. i say that because there are scientific works out there
	//that incorrectly assume that different solutions can be compared without randomness and congratulate themselves for a solution that achieves 99.8% accuracy compared to
	//99.77% of the best competitor on the same data set. on a data set of size 10000, the standard deviation for these accuracies is about 0.04%, and thus the difference
	//between 99.8% and 99.77% is not statistically significant and consequently might not be worthy of publication. the standard deviations below can be used to make sure
	//your new solution is actually statistically significantly better (the usualy cutoff for that is 2 or more standard deviations different) than your old solution.
	std::cerr << std::endl;
	std::cerr << "average objective = " << average_obj <<  " standard deviation = " << (sqrt( ( objective_2nd_moment - average_obj * average_obj ) / iterations )) << std::endl;
	std::cerr << "accuracy = " << accuracy << " standard deviation = " << (sqrt( accuracy * ( 1.0 - accuracy ) / iterations )) << std::endl;
	std::cerr << "maximum output = " << maximum_output << std::endl;
	std::cerr << "FLOP/s is approximately " << ((1000000.0*2.0*(((float)width)*((float)width)*length(layer_parameters))*batch_size*iterations)/sum_microseconds) << std::endl;
} // }}}

//int main() {
//	Buffers layer_parameters;
//	train_multithreaded( layer_parameters , 4 );
//	//train( layer_parameters );
//	std::ofstream parameter_file = std::ofstream( "extra_tiny_go_parameters.out" , std::ofstream::out | std::ofstream::binary );
//	if( parameter_file.bad() ) {
//		throw std::length_error( std::string( CURRENT_FUNCTION_NAME ) + ": error while trying to open extra_tiny_go_parameters.out" );
//	}
//	uint32_t temp;
//	try {
//		temp = layer_parameters.size();
//		parameter_file.write( (char*) &temp , sizeof(uint32_t) );
//		for( Buffer buffer : layer_parameters ) {
//			temp = buffer->size();
//			parameter_file.write( (char*) &temp , sizeof(uint32_t) );
//			if( 0 < buffer->size() ) {
//				parameter_file.write( (char*) (buffer->data()) , sizeof(float) * buffer->size() );
//			}
//		}
//	} catch(...) {
//		std::cerr << "error while trying to write to extra_tiny_go_parameters.out because of: " << std::endl;
//		throw;
//	}
//	std::cerr << "training set evaluation:" << std::endl;
//	test( layer_parameters , false );
//	std::cerr << "test set evaluation:" << std::endl;
//	test( layer_parameters , true );
//	return 0;
//}

uint32_t libertiesFloodFill( Buffer position , Buffer flags , uint32_t row , uint32_t col , uint32_t color , uint32_t height , uint32_t width ) { // {{{
	uint32_t result;
	std::vector<std::pair<uint32_t,uint32_t>> neighbors = std::vector<std::pair<uint32_t,uint32_t>>();
	if( 0 < row ) {
		neighbors.push_back(std::pair<uint32_t,uint32_t>(row-1,col));
	}
	if( row < height - 1 ) {
		neighbors.push_back(std::pair<uint32_t,uint32_t>(row+1,col));
	}
	if( 0 < col ) {
		neighbors.push_back(std::pair<uint32_t,uint32_t>(row,col-1));
	}
	if( col < width - 1 ) {
		neighbors.push_back(std::pair<uint32_t,uint32_t>(row,col+1));
	}
	result = 0;
	for( std::pair<uint32_t,uint32_t> p : neighbors ) {
		if( 0.0 == (*flags)[ p.first * width + p.second ] ) { // not yet seen
			if( 0.0 < (*position)[ color * height * width + p.first * width + p.second ] ) { // same-coloured stone found -> recurse
				(*flags)[ p.first * width + p.second ] = 1.0; // don't want to see it twice
				result += libertiesFloodFill( position , flags , p.first , p.second , color , height , width );
			} else if( 0.0 < (*position)[ 2 * height * width + p.first * width + p.second ] ) {
				(*flags)[ p.first * width + p.second ] = 1.0; // don't want to see it twice
				result += 1; // empty space found -> add 1 liberty
			}
		}
	}
	return result;
} // }}}

std::pair<uint32_t,Buffer> liberties( Buffer position , uint32_t row , uint32_t col , uint32_t height , uint32_t width ) { // {{{
	uint32_t color;
	Buffer flags = Buffer(new std::vector<float>((position->size())/3));
	zero_data(flags);
	if( 0.0 < position->at( 0 * height * width + row * width + col ) ) {
		color = 0;
	} else if( 0.0 < position->at( 1 * height * width + row * width + col ) ) {
		color = 1;
	} else if( 0.0 < position->at( 2 * height * width + row * width + col ) ) {
		color = 2;
	} else {
		throw std::domain_error( std::string( CURRENT_FUNCTION_NAME ) + ": internal error" );
	}
	(*flags)[ row * width + col ] = 1.0; // don't want to see it twice
	return ( std::pair<uint32_t,Buffer>( libertiesFloodFill( position , flags , row , col , color , height , width ) , flags ) );
} // }}}

void display_position( Buffer position , uint32_t height , uint32_t width ) { // {{{
	uint32_t row,col,black_stones,white_stones;
	std::cout << "current position:" << std::endl << "   ";
	for( col = 0 ; col < width ; col++ ) {
		std::cout << " " << (col%10);
	}
	std::cout << std::endl;
	black_stones = 0;
	white_stones = 0;
	for( row = 0 ; row < height ; row++ ) {
		if( 9 < row ) {
			std::cout << row << "  ";
		} else {
			std::cout << " " << row << "  ";
		}
		for( col = 0 ; col < width ; col++ ) {
			black_stones += ( 0.0 < (*position)[ 0 * height * width + row * width + col ])?1:0;
			white_stones += ( 0.0 < (*position)[ 1 * height * width + row * width + col ])?1:0;
			if( 0.0 < (*position)[ 0 * height * width + row * width + col ]) {
				std::cout << "X ";
			} else if( 0.0 < (*position)[ 1 * height * width + row * width + col ]) {
				std::cout << "O ";
			} else {
				std::cout << ". ";
			}
		}
		std::cout << std::endl;
	}
	std::cout << "black has " << black_stones << " stones, white has " << white_stones << " stones" << std::endl;
} // }}}

void play_move( Buffer position , uint32_t active_player , uint32_t move_row , uint32_t move_col , uint32_t height , uint32_t width , uint32_t& kou_row , uint32_t& kou_col , uint32_t& kou_row2 , uint32_t& kou_col2 ) { // {{{
	uint32_t row,col,nr_captives;
	std::vector<std::pair<uint32_t,uint32_t>> neighbors = std::vector<std::pair<uint32_t,uint32_t>>();
	if( 0 < move_row ) {
		neighbors.push_back(std::pair<uint32_t,uint32_t>(move_row-1,move_col));
	}
	if( static_cast<uint32_t>(move_row) < height - 1 ) {
		neighbors.push_back(std::pair<uint32_t,uint32_t>(move_row+1,move_col));
	}
	if( 0 < move_col ) {
		neighbors.push_back(std::pair<uint32_t,uint32_t>(move_row,move_col-1));
	}
	if( static_cast<uint32_t>(move_col) < width - 1 ) {
		neighbors.push_back(std::pair<uint32_t,uint32_t>(move_row,move_col+1));
	}
	(*position)[ active_player * height * width + move_row * width + move_col ] = 1.0;
	(*position)[ 2 * height * width + move_row * width + move_col ] = 0.0;
	std::pair<uint32_t,Buffer> libs;
	nr_captives = 0;
	for( std::pair<uint32_t,uint32_t> p : neighbors ) {
		libs = liberties( position , p.first , p.second , height , width );
		//opposing stones without liberties
		if( ( 0.0 < (*position)[ ( 1 - active_player ) * height * width + p.first * width + p.second ] ) && ( 0 == libs.first ) ) {
			for( row = 0 ; row < height ; row++ ) {
				for( col = 0 ; col < width ; col++ ) {
					if( ( 0.0 < (*position)[ ( 1 - active_player ) * height * width + row * width + col ])
							&& ( 0.0 < (*(libs.second))[ row * width + col ]) ) {
						(*position)[ ( 1 - active_player ) * height * width + row * width + col ] = 0.0;
						(*position)[ 2 * height * width + row * width + col ] = 1.0;
						nr_captives += 1;
						kou_row2 = row;
						kou_col2 = col;
					}
				}
			}
		}
	}
	if( 0 == nr_captives ) {
		libs = liberties( position , move_row , move_col , height , width );
		//own stones without liberties
		if( ( 0.0 < (*position)[ active_player * height * width + move_row * width + move_col ] ) && ( 0 == libs.first ) ) {
			for( row = 0 ; row < height ; row++ ) {
				for( col = 0 ; col < width ; col++ ) {
					if( ( 0.0 < (*position)[ active_player * height * width + row * width + col ])
							&& ( 0.0 < (*(libs.second))[ row * width + col ]) ) {
						(*position)[ active_player * height * width + row * width + col ] = 0.0;
						(*position)[ 2 * height * width + row * width + col ] = 1.0;
					}
				}
			}
		}
		kou_row = height;
		kou_col = width;
	} else {
		if( 1 == nr_captives ) {
			kou_row = move_row;
			kou_col = move_col;
		} else {
			kou_row = height;
			kou_col = width;
		}
	}
} // }}}

//play against AI
int main( int argc, char** argv ) {
	std::mt19937 rng;
	uint32_t channels = 64;
	uint32_t batch_size = 1;
	uint32_t height = 19;
	uint32_t width = 19;
	uint32_t active_player = 0;
	int32_t move_row,move_col,move_row2,move_col2;
	uint32_t row,col,kou_row,kou_col,kou_row2,kou_col2,temp2;
	float highest,second_highest;
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
	Buffer objective;
	Buffer target;
	Buffers layer_parameters;
	set_up_buffers( layer_inputs , layer_outputs , layer_input_gradients , layer_output_gradients , layer_parameters , layer_parameter_gradients , layer_parameter_gradients2 , extra_data , layer_parameter_updates , layer_1st_moment_estimates , layer_2nd_moment_estimates , target , objective , channels , batch_size , height , width , false , false , true );
	rng.seed( (uint_fast32_t) time( NULL ) );
	//gaussian_init_parameters( layer_parameters , rng , 0.13 );
	std::ifstream parameter_file = std::ifstream( "extra_tiny_go_parameters.out" , std::ifstream::in | std::ifstream::binary );
	if( parameter_file.bad() ) {
		throw std::length_error( std::string( CURRENT_FUNCTION_NAME ) + ": error while trying to open extra_tiny_go_parameters.out" );
	}
	try {
		parameter_file.read( (char*) &temp2 , sizeof(uint32_t) );
		layer_parameters.resize(temp2);
		for( Buffer& buffer : layer_parameters ) {
			parameter_file.read( (char*) &temp2 , sizeof(uint32_t) );
			buffer = Buffer(new std::vector<float>(temp2));
			if( 0 < buffer->size() ) {
				parameter_file.read( (char*) (buffer->data()) , sizeof(float) * buffer->size() );
			}
		}
	} catch(...) {
		std::cerr << "error while trying to read from extra_tiny_go_parameters.out because of: " << std::endl;
		throw;
	}
	parameter_file.close();
	std::string s;
	kou_row2 = height;
	kou_col2 = width;
	kou_row = height;
	kou_col = width;
	for( row = 0 ; row < height ; row++ ) {
		for( col = 0 ; col < width ; col++ ) {
			(*(layer_inputs.front()))[ 2 * height * width + row * width + col ] = 1.0;
		}
	}
	do {
		display_position( layer_inputs.front() , height , width );
		if( 0 == active_player ) { // human
			do {
				std::cout << "row of next move = ";
				getline( std::cin , s );
				move_row = atoi(s.c_str());
				std::cout << "col of next move = ";
				getline( std::cin , s );
				move_col = atoi(s.c_str());
			//don't allow moves outside of the board, on top of existing stones and moves taking back in a kou
			} while( ( move_row < 0 ) || ( height <= static_cast<uint32_t>(move_row) ) || ( move_col < 0 ) || ( width <= static_cast<uint32_t>(move_col) )
				|| ( 0.0 == (*(layer_inputs.front()))[ 2 * height * width + move_row * width + move_col ] )
				|| ( ( static_cast<uint32_t>(move_row) == kou_row2 ) && ( static_cast<uint32_t>(move_col) == kou_col2 )
				&& ( 0.0 < (*(layer_inputs.front()))[ ( 1 - active_player ) * height * width + kou_row * width + kou_col ] )
				&& ( kou_row < height ) && ( kou_col < width )
				&& ( 1 == liberties( layer_inputs.front() , kou_row , kou_col , height , width ).first) ) );
		} else { // AI
			forward_and_backward( layer_inputs , layer_outputs , layer_input_gradients , layer_output_gradients , layer_parameters , layer_parameter_gradients , extra_data , target , objective , channels , batch_size , height , width , true , true );
			highest = 0.0;
			second_highest = 0.0;
			move_row = 0.0;
			move_col = 0.0;
			move_row2 = 0.0;
			move_col2 = 0.0;
			for( row = 0 ; row < height ; row++ ) {
				for( col = 0 ; col < width ; col++ ) {
					if( 0.0 < (*(layer_inputs.front()))[ 2 * height * width + row * width + col ] ) {
						if( highest < (*(layer_outputs.back()))[ row * width + col ] ) {
							second_highest = highest;
							move_row2 = move_row;
							move_col2 = move_col;
							highest = (*(layer_outputs.back()))[ row * width + col ];
							move_row = row;
							move_col = col;
						} else if( second_highest < (*(layer_outputs.back()))[ row * width + col ] ) {
							second_highest = (*(layer_outputs.back()))[ row * width + col ];
							move_row2 = row;
							move_col2 = col;
						}
					}
				}
			}
			if( ( second_highest > 0.5 * highest ) && ( rng() % 1000000 < 1000000.0 * ( second_highest / ( highest + second_highest ) ) ) ) {
				move_row = move_row2;
				move_col = move_col2;
				std::cout << "AI plays " << move_row << " " << move_col << " (its second favorite choice)" << std::endl;
			} else {
				std::cout << "AI plays " << move_row << " " << move_col << std::endl;
			}
			//TODO: stop AI from taking back in a kou
		}
		play_move( layer_inputs.front() , active_player , move_row , move_col , height , width , kou_row , kou_col , kou_row2 , kou_col2 );
		active_player = 1 - active_player;
	} while(true);
	return 0;
}

// vim: ts=8 : autoindent : textwidth=0 : foldmethod=marker
