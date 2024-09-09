#include <fstream>
#include <sstream>
#include <thread>
#include "../extra_tiny_ann.hpp"
#ifdef USE_PNG
#include "png_simplified.hpp"
#endif

#define NR_CIFAR100_CLASSES 100
#define NR_CIFAR100_TRAIN_IMAGES 50000
#define NR_CIFAR100_TEST_IMAGES 10000

#pragma pack(push, 1)
struct cifar100_t {
	uint8_t coarse_label;
	uint8_t label;
	uint8_t pixels[3][32][32];
};
#pragma pack(pop)

typedef std::shared_ptr<std::vector<struct cifar100_t>> CIFAR100_Buffer;

CIFAR100_Buffer load_cifar100_data( bool test_set ) { // {{{
	uint_fast32_t nr_images = test_set?NR_CIFAR100_TEST_IMAGES:NR_CIFAR100_TRAIN_IMAGES;
	std::string filename = test_set?"test.bin":"train.bin";
	CIFAR100_Buffer cifar100_images = CIFAR100_Buffer( new std::vector<struct cifar100_t>( nr_images ) );
	std::ifstream cifar100_file = std::ifstream( filename , std::ifstream::in | std::ifstream::binary );
	if( cifar100_file.bad() ) {
		throw std::length_error( std::string( CURRENT_FUNCTION_NAME ) + ": error while trying to open " + filename );
	}
	try {
		cifar100_file.read( (char*) (cifar100_images->data()) , sizeof(struct cifar100_t) * nr_images );
	} catch(...) {
		std::cerr << "error while trying to read " << filename << " because of: " << std::endl;
		throw;
	}
	return cifar100_images;
} // }}}

void generate_data_and_targets( const CIFAR100_Buffer cifar100_images , std::mt19937& rng , bool test_set , Buffer data , Buffer target , uint32_t batch_size , int32_t index = -1 ) { // {{{
	uint_fast32_t i,j,k,r;
	for( i = 0 ; i < batch_size ; i++ ) {
		if( -1 == index ) {
			r = rng() % (test_set?NR_CIFAR100_TEST_IMAGES:NR_CIFAR100_TRAIN_IMAGES);
		} else {
			r = index;
		}
		for( j = 0 ; j < 32 ; j++ ) {
			for( k = 0 ; k < 32 ; k++ ) {
				data->at( 0 * batch_size * 32 * 32 + i * 32 * 32 + j * 32 + k ) = cifar100_images->at(r).pixels[0][j][k] / 255.0;
				data->at( 1 * batch_size * 32 * 32 + i * 32 * 32 + j * 32 + k ) = cifar100_images->at(r).pixels[1][j][k] / 255.0;
				data->at( 2 * batch_size * 32 * 32 + i * 32 * 32 + j * 32 + k ) = cifar100_images->at(r).pixels[2][j][k] / 255.0;
			}
		}
		for( j = 0 ; j < NR_CIFAR100_CLASSES ; j++ ) {
			target->at( j * batch_size + i ) = 0;
		}
		target->at( cifar100_images->at(r).label * batch_size + i ) = 1;
	}
} // }}}

void forward_and_backward( Buffers& layer_inputs , Buffers& layer_outputs , Buffers& layer_input_gradients , Buffers& layer_output_gradients , const Buffers& layer_parameters , Buffers& layer_parameter_gradients , Buffers& extra_data , Buffer target , Buffer objective , uint32_t channels , uint32_t batch_size , uint32_t height , uint32_t width , bool forward_only , bool no_param_gradient ) { // {{{
	int32_t i;
	//neural network forward pass (compute layer outputs)
	forward_pointwise_convolution( layer_inputs[0] , layer_outputs[0] , layer_parameters[0] , 3 , channels , batch_size , height , width );
	for( i = 0 ; i < 11 ; i++ ) {
		forward_depthwise_convolution( layer_inputs[1+4*i] , layer_outputs[1+4*i] , layer_parameters[1+4*i] , channels , channels , batch_size , height , width , 7 , 7 );
		forward_pointwise_convolution( layer_inputs[2+4*i] , layer_outputs[2+4*i] , layer_parameters[2+4*i] , channels , channels , batch_size , height , width );
		forward_bias( layer_inputs[3+4*i] , layer_outputs[3+4*i] , layer_parameters[3+4*i] , channels , channels , batch_size , height , width );
		forward_nonlin_leaky_relu( layer_inputs[4+4*i] , layer_outputs[4+4*i] , channels , channels , batch_size , height , width );
	}
	forward_global_max_pooling( layer_inputs[45] , layer_outputs[45] , channels , channels , batch_size , height , width , extra_data[45] );
	forward_pointwise_convolution( layer_inputs[46] , layer_outputs[46] , layer_parameters[46] , channels , NR_CIFAR100_CLASSES , batch_size , 1 , 1 );
	forward_nonlin_poly_exp( layer_inputs[47] , layer_outputs[47] , NR_CIFAR100_CLASSES , NR_CIFAR100_CLASSES , batch_size , 1 , 1 );
	//forward_nonlin_exp( layer_inputs[47] , layer_outputs[47] , NR_CIFAR100_CLASSES , NR_CIFAR100_CLASSES , batch_size , 1 , 1 );
	forward_channel_normalization( layer_inputs[48] , layer_outputs[48] , NR_CIFAR100_CLASSES , NR_CIFAR100_CLASSES , batch_size , 1 , 1 , extra_data[48] );

	cross_entropy_forward( layer_outputs[48] , target , objective , NR_CIFAR100_CLASSES , batch_size , 1 , 1 ); //loss function
	if( forward_only ) {
		return;
	}
	cross_entropy_backward( layer_outputs[48] , target , layer_output_gradients[48] , NR_CIFAR100_CLASSES , batch_size , 1 , 1 ); //loss gradient

	//neural network backward pass (compute layer input gradients)
	backward_channel_normalization( layer_outputs[48] , layer_output_gradients[48] , layer_input_gradients[48] , layer_parameters[48] , NR_CIFAR100_CLASSES , NR_CIFAR100_CLASSES , batch_size , 1 , 1 , extra_data[48] );
	backward_nonlin_poly_exp( layer_outputs[47] , layer_output_gradients[47] , layer_input_gradients[47] , layer_parameters[47] , NR_CIFAR100_CLASSES , NR_CIFAR100_CLASSES , batch_size , 1 , 1 );
	//backward_nonlin_exp( layer_outputs[47] , layer_output_gradients[47] , layer_input_gradients[47] , layer_parameters[47] , NR_CIFAR100_CLASSES , NR_CIFAR100_CLASSES , batch_size , 1 , 1 );
	backward_pointwise_convolution( layer_outputs[46] , layer_output_gradients[46] , layer_input_gradients[46] , layer_parameters[46] , channels , NR_CIFAR100_CLASSES , batch_size , 1 , 1 );
	backward_global_max_pooling( layer_outputs[45] , layer_output_gradients[45] , layer_input_gradients[45] , layer_parameters[45] , channels , channels , batch_size , height , width , extra_data[45] );
	for( i = 10 ; 0 <= i ; i-- ) {
		backward_nonlin_leaky_relu( layer_outputs[4+4*i] , layer_output_gradients[4+4*i] , layer_input_gradients[4+4*i] , layer_parameters[4+4*i] , channels , channels , batch_size , height , width );
		backward_bias( layer_outputs[3+4*i] , layer_output_gradients[3+4*i] , layer_input_gradients[3+4*i] , layer_parameters[3+4*i] , channels , channels , batch_size , height , width );
		backward_pointwise_convolution( layer_outputs[2+4*i] , layer_output_gradients[2+4*i] , layer_input_gradients[2+4*i] , layer_parameters[2+4*i] , channels , channels , batch_size , height , width );
		backward_depthwise_convolution( layer_outputs[1+4*i] , layer_output_gradients[1+4*i] , layer_input_gradients[1+4*i] , layer_parameters[1+4*i] , channels , channels , batch_size , height , width  , 7 , 7);
	}
	backward_pointwise_convolution( layer_outputs[0] , layer_output_gradients[0] , layer_input_gradients[0] , layer_parameters[0] , 3 , channels , batch_size , height , width );
	if( no_param_gradient ) {
		return;
	}

	//compute parameter gradients of each layer of the neural network
	param_gradient_pointwise_convolution( layer_inputs[46] , layer_output_gradients[46] , layer_parameter_gradients[46] , channels , NR_CIFAR100_CLASSES , batch_size , 1 , 1 );
	for( i = 0 ; i < 11 ; i++ ) {
		param_gradient_bias( layer_inputs[3+4*i] , layer_output_gradients[3+4*i] , layer_parameter_gradients[3+4*i] , channels , channels , batch_size , height , width );
		param_gradient_pointwise_convolution( layer_inputs[2+4*i] , layer_output_gradients[2+4*i] , layer_parameter_gradients[2+4*i] , channels , channels , batch_size , height , width );
		param_gradient_depthwise_convolution( layer_inputs[1+4*i] , layer_output_gradients[1+4*i] , layer_parameter_gradients[1+4*i] , channels , channels , batch_size , height , width , 7 , 7 );
	}
	param_gradient_pointwise_convolution( layer_inputs[0] , layer_output_gradients[0] , layer_parameter_gradients[0] , 3 , channels , batch_size , height , width );
} // }}}

void set_up_buffers( Buffers& layer_inputs , Buffers& layer_outputs , Buffers& layer_input_gradients , Buffers& layer_output_gradients , Buffers& layer_parameters , Buffers& layer_parameter_gradients , Buffers& layer_parameter_gradients2 , Buffers& extra_data , Buffers& layer_parameter_updates , Buffers& layer_1st_moment_estimates , Buffers& layer_2nd_moment_estimates , Buffer& target , Buffer& objective , uint32_t channels , uint32_t batch_size , uint32_t height , uint32_t width , bool forward_only , bool no_param_gradient , bool set_up_thread_independent_buffers ) { // {{{
	uint32_t i;
	std::vector<std::tuple<uint32_t,uint32_t,uint32_t,uint32_t>> buffer_sizes = std::vector<std::tuple<uint32_t,uint32_t,uint32_t,uint32_t>>();
	buffer_sizes.push_back(buffer_sizes_pointwise_convolution( 3 , channels , batch_size , height , width ));
	for( i = 0 ; i < 11 ; i++ ) {
		buffer_sizes.push_back(buffer_sizes_depthwise_convolution( channels , channels , batch_size , height , width , 7 , 7 ));
		buffer_sizes.push_back(buffer_sizes_pointwise_convolution(channels , channels , batch_size , height , width));
		buffer_sizes.push_back(buffer_sizes_bias(channels , channels , batch_size , height , width));
		buffer_sizes.push_back(buffer_sizes_nonlin_leaky_relu(channels , channels , batch_size , height , width));
	}
	buffer_sizes.push_back(buffer_sizes_global_max_pooling(channels , channels , batch_size , height , width));
	buffer_sizes.push_back(buffer_sizes_pointwise_convolution(channels , NR_CIFAR100_CLASSES , batch_size , 1 , 1));
	buffer_sizes.push_back(buffer_sizes_nonlin_poly_exp(NR_CIFAR100_CLASSES , NR_CIFAR100_CLASSES , batch_size , 1 , 1));
	//buffer_sizes.push_back(buffer_sizes_nonlin_exp(NR_CIFAR100_CLASSES , NR_CIFAR100_CLASSES , batch_size , 1 , 1));
	buffer_sizes.push_back(buffer_sizes_channel_normalization(NR_CIFAR100_CLASSES , NR_CIFAR100_CLASSES , batch_size , 1 , 1));
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

void multithreaded_training_iteration( std::vector<Buffers>& thread_layer_inputs , std::vector<Buffers>& thread_layer_outputs , std::vector<Buffers>& thread_layer_input_gradients , std::vector<Buffers>& thread_layer_output_gradients , Buffers& layer_parameters , std::vector<Buffers>& thread_layer_parameter_gradients , Buffers& layer_parameter_gradients2 , std::vector<Buffers>& thread_extra_data , Buffers& layer_parameter_updates , Buffers& layer_1st_moment_estimates , Buffers& layer_2nd_moment_estimates , Buffers& thread_target , Buffers& thread_objective , uint32_t& iteration , uint32_t iterations , const CIFAR100_Buffer cifar100_images , std::mt19937& rng , bool test_set , uint32_t channels , uint32_t batch_size , uint32_t effective_batch_size , uint32_t width , uint32_t height , float learning_rate ) { // {{{
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
			generate_data_and_targets( cifar100_images , rng , test_set , thread_layer_inputs.at(thread_nr).front() , thread_target.at(thread_nr) , batch_size );
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
					output_max_index = NR_CIFAR100_CLASSES;
					for( i = 0 ; i < NR_CIFAR100_CLASSES ; i++ ) {
						if( temp < thread_layer_outputs.at(thread_nr).back()->at( batch_size * i + j ) ) {
							temp = thread_layer_outputs.at(thread_nr).back()->at( batch_size * i + j );
							output_max_index = i;
						}
					}
					temp = std::numeric_limits<float>::lowest();
					target_max_index = NR_CIFAR100_CLASSES;
					for( i = 0 ; i < NR_CIFAR100_CLASSES ; i++ ) {
						if( temp < thread_target.at(thread_nr)->at( batch_size * i + j ) ) {
							temp = thread_target.at(thread_nr)->at( batch_size * i + j );
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

void training_iteration( Buffers& layer_inputs , Buffers& layer_outputs , Buffers& layer_input_gradients , Buffers& layer_output_gradients , Buffers& layer_parameters , Buffers& layer_parameter_gradients , Buffers& layer_parameter_gradients2 , Buffers& extra_data , Buffers& layer_parameter_updates , Buffers& layer_1st_moment_estimates , Buffers layer_2nd_moment_estimates , Buffer target , Buffer objective , uint32_t& iteration , uint32_t iterations , const CIFAR100_Buffer cifar100_images , std::mt19937& rng , bool test_set , uint32_t channels , uint32_t batch_size , uint32_t effective_batch_size , uint32_t width , uint32_t height , float learning_rate ) { // {{{
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
		generate_data_and_targets( cifar100_images , rng , test_set , layer_inputs.front() , target , batch_size );
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
				output_max_index = NR_CIFAR100_CLASSES;
				for( i = 0 ; i < NR_CIFAR100_CLASSES ; i++ ) {
					if( temp < layer_outputs.back()->at( batch_size * i + j ) ) {
						temp = layer_outputs.back()->at( batch_size * i + j );
						output_max_index = i;
					}
				}
				temp = std::numeric_limits<float>::lowest();
				target_max_index = NR_CIFAR100_CLASSES;
				for( i = 0 ; i < NR_CIFAR100_CLASSES ; i++ ) {
					if( temp < target->at( batch_size * i + j ) ) {
						temp = target->at( batch_size * i + j );
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
	CIFAR100_Buffer cifar100_images;
	uint32_t channels = 32;
	uint32_t batch_size = 4;
	uint32_t effective_batch_size = 240;
	uint32_t height = 32;
	uint32_t width = 32;
	uint32_t iterations = 25000;
	rng.seed( (uint_fast32_t) time( NULL ) );
	set_up_buffers( layer_inputs , layer_outputs , layer_input_gradients , layer_output_gradients , layer_parameters , layer_parameter_gradients , layer_parameter_gradients2 , extra_data , layer_parameter_updates , layer_1st_moment_estimates , layer_2nd_moment_estimates , target , objective , channels , batch_size , height , width , false , false , true );
	cifar100_images = load_cifar100_data( false );
	gaussian_init_parameters( layer_parameters , rng , 0.13 );
	for( iteration = 0 ; iteration < iterations ; iteration++ ) {
		training_iteration( layer_inputs , layer_outputs , layer_input_gradients , layer_output_gradients , layer_parameters , layer_parameter_gradients , layer_parameter_gradients2 , extra_data , layer_parameter_updates , layer_1st_moment_estimates , layer_2nd_moment_estimates , target , objective , iteration , iterations , cifar100_images , rng , false , channels , batch_size , effective_batch_size , width , height , learning_rate_schedule( iteration , iterations ) );
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
	CIFAR100_Buffer cifar100_images;
	uint32_t channels = 32;
	uint32_t batch_size = 4;
	uint32_t effective_batch_size = 240;
	uint32_t height = 32;
	uint32_t width = 32;
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
	cifar100_images = load_cifar100_data( false );
	gaussian_init_parameters( layer_parameters , rng , 0.13 );
	for( iteration = 0 ; iteration < iterations ; iteration++ ) {
		multithreaded_training_iteration( thread_layer_inputs , thread_layer_outputs , thread_layer_input_gradients , thread_layer_output_gradients , layer_parameters , thread_layer_parameter_gradients , layer_parameter_gradients2 , thread_extra_data , layer_parameter_updates , layer_1st_moment_estimates , layer_2nd_moment_estimates , thread_target , thread_objective , iteration , iterations , cifar100_images , rng , false , channels , batch_size , effective_batch_size , width , height , learning_rate_schedule( iteration , iterations ) );
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
	CIFAR100_Buffer cifar100_images;
	uint32_t channels = 32;
	uint32_t batch_size = 4;
	uint32_t iterations = 10000;
	uint32_t height = 32;
	uint32_t width = 32;
	float average_obj = 0.0;
	float accuracy = 0;
	float maximum_output = std::numeric_limits<float>::lowest();
	uint64_t sum_microseconds;
	uint32_t output_max_index,target_max_index;
	float temp;
	double objective_2nd_moment; //needs to be double precision to reduce rounding error
	rng.seed( (uint_fast32_t) time( NULL ) );
	set_up_buffers( layer_inputs , layer_outputs , layer_input_gradients , layer_output_gradients , dummy , layer_parameter_gradients , layer_parameter_gradients2 , extra_data , layer_parameter_updates , layer_1st_moment_estimates , layer_2nd_moment_estimates , target , objective , channels , batch_size , height , width , true , true , true );
	cifar100_images = load_cifar100_data( test_set );
	sum_microseconds = 0;
	objective_2nd_moment = 0;
	for( iteration = 0 ; iteration < iterations ; iteration += batch_size ) {
		zero_data(layer_inputs);
		zero_data(layer_outputs);
		generate_data_and_targets( cifar100_images , rng , test_set , layer_inputs.front() , target , batch_size , iteration );
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
			output_max_index = NR_CIFAR100_CLASSES;
			for( i = 0 ; i < NR_CIFAR100_CLASSES ; i++ ) {
				if( temp < layer_outputs.back()->at( batch_size * i + j ) ) {
					temp = layer_outputs.back()->at( batch_size * i + j );
					output_max_index = i;
				}
			}
			temp = std::numeric_limits<float>::lowest();
			target_max_index = NR_CIFAR100_CLASSES;
			for( i = 0 ; i < NR_CIFAR100_CLASSES ; i++ ) {
				if( temp < target->at( batch_size * i + j ) ) {
					temp = target->at( batch_size * i + j );
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

int main() {
	Buffers layer_parameters;
	train_multithreaded( layer_parameters , 6 );
	//train( layer_parameters );
	std::ofstream parameter_file = std::ofstream( "extra_tiny_cifar100_parameters.out" , std::ofstream::out | std::ofstream::binary );
	if( parameter_file.bad() ) {
		throw std::length_error( std::string( CURRENT_FUNCTION_NAME ) + ": error while trying to open extra_tiny_cifar100_parameters.out" );
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
		std::cerr << "error while trying to write to extra_tiny_cifar100_parameters.out because of: " << std::endl;
		throw;
	}
	std::cerr << "training set evaluation:" << std::endl;
	test( layer_parameters , false );
	std::cerr << "test set evaluation:" << std::endl;
	test( layer_parameters , true );
	return 0;
}

#ifdef USE_PNG

////apply (a.k.a. inference) neural network stored in extra_tiny_cifar100_parameters.out to a single PNG image
//int main( int argc, char** argv ) {
//	if( 2 != argc ) {
//		std::cerr << "gimme a filename please!" << std::endl;
//		return 1;
//	}
//	Buffers layer_parameters;
//	std::ifstream parameter_file = std::ifstream( "extra_tiny_cifar100_parameters.out" , std::ifstream::in | std::ifstream::binary );
//	if( parameter_file.bad() ) {
//		throw std::length_error( std::string( CURRENT_FUNCTION_NAME ) + ": error while trying to open extra_tiny_cifar100_parameters.out" );
//	}
//	uint32_t temp,row,col,class_index;
//	float temp2;
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
//		std::cerr << "error while trying to read from extra_tiny_cifar100_parameters.out because of: " << std::endl;
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
//	uint32_t channels = 32;
//	uint32_t batch_size = 1;
//	uint32_t height = 32;
//	uint32_t width = 32;
//	uint32_t output_max_index;
//	set_up_buffers( layer_inputs , layer_outputs , layer_input_gradients , layer_output_gradients , dummy , layer_parameter_gradients , layer_parameter_gradients2 , extra_data , layer_parameter_updates , layer_1st_moment_estimates , layer_2nd_moment_estimates , target , objective , channels , batch_size , height , width , true , true , true );
//	struct png_image_t input_image = read_png_file( std::string(argv[1]) );
//	if( ( static_cast<int32_t>(height) != input_image.height ) || ( static_cast<int32_t>(width) != input_image.width ) ) {
//		throw std::length_error( std::string( CURRENT_FUNCTION_NAME ) + ": can only handle PNG images of size " + std::to_string(height) + " x " + std::to_string(width) );
//	}
//	png_bytep rowp = NULL;
//	png_bytep px = NULL;
//	for( row = 0 ; row < height ; row++ ) {
//		rowp = input_image.row_pointers[row];
//		for( col = 0 ; col < width ; col++ ) {
//			px = &(rowp[col * 4]);
//			layer_inputs.front()->at( 0 * 32 * 32 + row * 32 + col ) = px[0];
//			layer_inputs.front()->at( 1 * 32 * 32 + row * 32 + col ) = px[1];
//			layer_inputs.front()->at( 2 * 32 * 32 + row * 32 + col ) = px[2];
//		}
//	}
//	//std::mt19937 rng;
//	//rng.seed( (uint_fast32_t) time( NULL ) );
//	//gaussian_init_parameters( layer_parameters , rng , 0.13 );
//	forward_and_backward( layer_inputs , layer_outputs , layer_input_gradients , layer_output_gradients , layer_parameters , layer_parameter_gradients , extra_data , target , objective , channels , batch_size , height , width , true , true );
//	temp2 = std::numeric_limits<float>::lowest();
//	output_max_index = 0;
//	for( class_index = 0 ; class_index < NR_CIFAR100_CLASSES ; class_index++ ) {
//		if( temp2 < layer_outputs.back()->at( class_index ) ) {
//			temp2 = layer_outputs.back()->at( class_index );
//			output_max_index = class_index;
//		}
//	}
//	std::string label_names[NR_CIFAR100_CLASSES] = {"apple","aquarium_fish","baby","bear","beaver","bed","bee","beetle","bicycle","bottle","bowl","boy","bridge","bus","butterfly","camel","can","castle","caterpillar","cattle","chair","chimpanzee","clock","cloud","cockroach","couch","crab","crocodile","cup","dinosaur","dolphin","elephant","flatfish","forest","fox","girl","hamster","house","kangaroo","keyboard","lamp","lawn_mower","leopard","lion","lizard","lobster","man","maple_tree","motorcycle","mountain","mouse","mushroom","oak_tree","orange","orchid","otter","palm_tree","pear","pickup_truck","pine_tree","plain","plate","poppy","porcupine","possum","rabbit","raccoon","ray","road","rocket","rose","sea","seal","shark","shrew","skunk","skyscraper","snail","snake","spider","squirrel","streetcar","sunflower","sweet_pepper","table","tank","telephone","television","tiger","tractor","train","trout","tulip","turtle","wardrobe","whale","willow_tree","wolf","woman","worm"};
//	std::cout << "most probable class is '" << (label_names[output_max_index]) << "', probability is " << temp2 << std::endl;
//	//std::cout << "P3" << std::endl << "32 32" << std::endl << "255" << std::endl;
//	//for( row = 0 ; row < height ; row++ ) {
//	//	rowp = input_image.row_pointers[row];
//	//	for( col = 0 ; col < width ; col++ ) {
//	//		px = &(rowp[col * 4]);
//	//		std::cout << ((int)(px[0])) << " " << ((int)(px[1])) << " " << ((int)(px[2]));
//	//		if( width-1 != col ) {
//	//			std::cout << " ";
//	//		}
//	//	}
//	//	std::cout << std::endl;
//	//}
//	return 0;
//}

#endif

//potential tasks to solve:
//- train larger network
//- make the number of channels and number of neural network layers easily changeable
//- make a nice graph as output during training to visualize progress (e.g. using gnuplot)
//- change inference so that it works on multiple images simultaneously using batch_size > 1
//- regularly write the current state (parameters, iteration, ADAM's moment estimates, RNG state) to disk during optimization so that you can restart from there if training is interrupted
//- implement multithreaded version of test()
//- more data augmentation methods (horizontal reflection, mixup, small translation, adding patches of noise)
//- export network description and parameters into a standard format (e.g. ONNX, ConvNetJS)
//- knowledge distillation from a large network to a small network

// vim: ts=8 : autoindent : textwidth=0 : foldmethod=marker
