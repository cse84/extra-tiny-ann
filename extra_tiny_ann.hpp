// Copyright 2024 Daniel Müller
//
// Licensed under the terms given in the file named "LICENSE"
// in the root folder of the project

#include <iostream>
#include <cstdint>
#include <cstdarg>
#include <cstring>
#include <string>
#include <stdexcept>
#include <numeric>
#include <algorithm>
#include <limits>
#include <map>
#include <type_traits>
#include <vector>
#include <cmath>
#include <memory>
#include <random>
#include <chrono>

#ifdef USE_BLAS
#include "/usr/include/x86_64-linux-gnu/cblas.h"
#endif

// why can't there be a standard macro/variable containing the complete function name including namespace and classname?
#ifdef __GNUC__
#define CURRENT_FUNCTION_NAME __PRETTY_FUNCTION__
#else
#define CURRENT_FUNCTION_NAME __func__
#endif

#define MIN(X,Y) (((X)<(Y))?(X):(Y))
#define MAX(X,Y) (((X)>(Y))?(X):(Y))

typedef std::shared_ptr<std::vector<float>> Buffer;
typedef std::vector<Buffer> Buffers;

//i wanted to do it without templates for the sake of simplicity, but this one doesn't work without templates
template<class F>
void zip_with( Buffer buffer0 , Buffer buffer1 , F&& function ) {
	uint32_t i;
	if( buffer0->empty() ) {
		return;
	}
	for( i = 0 ; i < buffer0->size() ; i++ ) {
		buffer0->at(i) = function( buffer0->at(i) , buffer1->at(i) );
	}
}

template<class F>
void zip_with( Buffers& buffers0 , Buffers& buffers1 , F&& function ) {
	uint32_t i;
	for( i = 0 ; i < buffers0.size() ; i++ ) {
		zip_with( buffers0[i] , buffers1[i] , function );
	}
}

template<class F>
void map( Buffer buffer , F&& function ) {
	for( float& x : (*buffer) ) {
		x = function( x );
	}
}

template<class F>
void map( Buffers& buffers , F&& function ) {
	for( Buffer buffer : buffers ) {
		map( buffer , function );
	}
}

template<class T,class F>
T foldl( Buffer buffer , T accumulator , F&& function ) {
	uint32_t i;
	i = 0;
	for( float& x : (*buffer) ) {
		accumulator = function( i , accumulator , x );
		i += 1;
	}
	return accumulator;
}

template<class T,class F>
T foldl( Buffers buffers , T accumulator , F&& function ) {
	uint32_t i;
	i = 0;
	for( Buffer buffer : buffers ) {
		for( float& x : (*buffer) ) {
			accumulator = function( i , accumulator , x );
			i += 1;
		}
	}
	return accumulator;
}

void add( Buffer buffer0 , Buffer buffer1 ) {
	zip_with( buffer0 , buffer1 , [](float x,float y){return (x+y);} );
}

void add( Buffers& buffers0 , Buffers& buffers1 ) {
	zip_with( buffers0 , buffers1 , [](float x,float y){return (x+y);} );
}

void scale( Buffer buffer , float scale ) {
	map( buffer , [=](float x){return (x*scale);} );
}

void scale( Buffers& buffers , float scale ) {
	map( buffers , [=](float x){return (x*scale);} );
}

void zero_data( Buffer buffer ) {
	map( buffer , [](float x){return 0.0;} );
}

void zero_data( Buffers& buffers ) {
	map( buffers , [](float x){return 0.0;} );
}

float norm( const Buffer& buffer ) {
	return (sqrt( foldl(buffer,0.0,[](uint32_t i,float a,float x){return (a+x*x);}) ));
}

float norm( const Buffers buffers ) {
	return (sqrt( foldl(buffers,0.0,[](uint32_t i,float a,float x){return (a+x*x);}) ));
}

uint64_t length( const Buffer& buffer ) {
	return (buffer->size());
}

uint64_t length( const Buffers buffers ) {
	uint64_t result;
	result = 0;
	for( Buffer buffer : buffers ) {
		result += length(buffer);
	}
	return result;
}

void log_norms( const Buffers& buffers ) {
	for( const Buffer& b : buffers ) {
		std::cerr << (9+((int)(MAX(-9.0,log(norm(b)))))) << " ";
	}
	std::cerr << std::endl;
	return;
}

//adopted from BLAS' reference implementation in Fortran
void my_sgemm( bool trans_a , bool trans_b , uint_fast32_t m , uint_fast32_t n , uint_fast32_t k , float alpha , const float* a , uint_fast32_t lda , const float* b , uint_fast32_t ldb , float beta , float* c , uint_fast32_t ldc ) { // {{{
	float temp,temp0,temp1,temp2,temp3;
	uint_fast32_t i,j,l/*,ncol_a*/,nrow_a,nrow_b;
	nrow_a = ( ! trans_a )?m:k;
	//ncol_a = ( ! trans_a )?k:m; // WTF? the reference implementation contains an unused variable.
	nrow_b = ( ! trans_b )?k:n;
	if( lda < MAX(1,nrow_a) ) {
		throw std::length_error( std::string( CURRENT_FUNCTION_NAME ) + ": lda must not be less than " + std::to_string( MAX(1,nrow_a) ) );
	} else if( ldb < MAX(1,nrow_b) ) {
		throw std::length_error( std::string( CURRENT_FUNCTION_NAME ) + ": ldb must not be less than " + std::to_string( MAX(1,nrow_b) ) );
	} else if( ldc < MAX(1,m) ) {
		throw std::length_error( std::string( CURRENT_FUNCTION_NAME ) + ": ldc must not be less than " + std::to_string( MAX(1,m) ) );
	}
	if( ( 0 == m ) || ( 0 == n ) || ( ( ( 0.0 == alpha ) || ( 0 == k ) ) && ( 1.0 == beta ) ) ) {
		return;
	}
	if( 0.0 == alpha ) {
		for( i = 0 ; i < m * n ; i++ ) {
			c[i] *= beta;
		}
		return;
	}
	if( ! trans_b ) {
		if( ! trans_a ) {
			for( j = 0 ; j < n ; j++ ) { // {{{
				for( i = 0 ; i < m ; i++ ) {
					c[ j * ldc + i ] *= beta;
				}
				if( 4 <= k ) {
					for( l = 0 ; l < k - k % 4 ; l += 4 ) {
						temp0 = alpha * b[ ldb * j + ( l + 0 ) ];
						temp1 = alpha * b[ ldb * j + ( l + 1 ) ];
						temp2 = alpha * b[ ldb * j + ( l + 2 ) ];
						temp3 = alpha * b[ ldb * j + ( l + 3 ) ];
						for( i = 0 ; i < m ; i++ ) {
							c[ j * ldc + i ] += temp0 * a[ ( l + 0 ) * lda + i ];
							c[ j * ldc + i ] += temp1 * a[ ( l + 1 ) * lda + i ];
							c[ j * ldc + i ] += temp2 * a[ ( l + 2 ) * lda + i ];
							c[ j * ldc + i ] += temp3 * a[ ( l + 3 ) * lda + i ];
						}
					}
				} else {
					l = 0;
				}
				for( ; l < k ; l++ ) {
					temp = alpha * b[ ldb * j + l ];
					for( i = 0 ; i < m ; i++ ) {
						c[ j * ldc + i ] += temp*a[ l * lda + i ];
					}
				}
			} // }}}
		} else {
			for( j = 0 ; j < n ; j++ ) { // {{{
				if( 4 <= m ) {
					for( i = 0 ; i < m - m % 4 ; i += 4 ) {
						temp0 = 0.0;
						temp1 = 0.0;
						temp2 = 0.0;
						temp3 = 0.0;
						for( l = 0 ; l < k ; l++ ) {
							temp0 += a[ ( i + 0 ) * lda + l ] * b[ j * ldb + l ];
							temp1 += a[ ( i + 1 ) * lda + l ] * b[ j * ldb + l ];
							temp2 += a[ ( i + 2 ) * lda + l ] * b[ j * ldb + l ];
							temp3 += a[ ( i + 3 ) * lda + l ] * b[ j * ldb + l ];
						}
						c[ j * ldc + ( i + 0 ) ] = alpha * temp0 + beta * c[ j * ldc + ( i + 0 ) ];
						c[ j * ldc + ( i + 1 ) ] = alpha * temp1 + beta * c[ j * ldc + ( i + 1 ) ];
						c[ j * ldc + ( i + 2 ) ] = alpha * temp2 + beta * c[ j * ldc + ( i + 2 ) ];
						c[ j * ldc + ( i + 3 ) ] = alpha * temp3 + beta * c[ j * ldc + ( i + 3 ) ];
					}
				} else {
					i = 0;
				}
				for( ; i < m ; i++ ) {
					temp = 0.0;
					for( l = 0 ; l < k ; l++ ) {
						temp += a[ i * lda + l ] * b[ j * ldb + l ];
					}
					c[ j * ldc + i ] = alpha * temp + beta * c[ j * ldc + i ];
				}
			} // }}}
		}
	} else {
		if( ! trans_a ) {
			for( j = 0 ; j < n ; j++ ) { // {{{
				for( i = 0 ; i < m ; i++ ) {
					c[ j * ldc + i ] *= beta;
				}
				if( 4 <= k ) {
					for( l = 0 ; l < k - k % 4 ; l += 4 ) {
						temp0 = alpha * b[ ( l + 0 ) * ldb + j ];
						temp1 = alpha * b[ ( l + 1 ) * ldb + j ];
						temp2 = alpha * b[ ( l + 2 ) * ldb + j ];
						temp3 = alpha * b[ ( l + 3 ) * ldb + j ];
						for( i = 0 ; i < m ; i++ ) {
							c[ j * ldc + i ] += temp0 * a[ ( l + 0 ) * lda + i ];
							c[ j * ldc + i ] += temp1 * a[ ( l + 1 ) * lda + i ];
							c[ j * ldc + i ] += temp2 * a[ ( l + 2 ) * lda + i ];
							c[ j * ldc + i ] += temp3 * a[ ( l + 3 ) * lda + i ];
						}
					}
				} else {
					l = 0;
				}
				for( ; l < k ; l++ ) {
					temp = alpha * b[ l * ldb + j ];
					for( i = 0 ; i < m ; i++ ) {
						c[ j * ldc + i ] += temp * a[ l * lda + i ];
					}
				}
			} // }}}
		} else {
			for( j = 0 ; j < n ; j++ ) { // {{{
				if( 4 <= m ) {
					for( i = 0 ; i < m - m % 4 ; i += 4 ) {
						temp0 = 0.0;
						temp1 = 0.0;
						temp2 = 0.0;
						temp3 = 0.0;
						for( l = 0 ; l < k ; l++ ) {
							temp0 += a[ ( i + 0 ) * lda + l ] * b[ l * ldb + j ];
							temp1 += a[ ( i + 1 ) * lda + l ] * b[ l * ldb + j ];
							temp2 += a[ ( i + 2 ) * lda + l ] * b[ l * ldb + j ];
							temp3 += a[ ( i + 3 ) * lda + l ] * b[ l * ldb + j ];
						}
						c[ j * ldc + ( i + 0 ) ] += alpha * temp0 + beta * c[ j * ldc + ( i + 0 ) ];
						c[ j * ldc + ( i + 1 ) ] += alpha * temp1 + beta * c[ j * ldc + ( i + 1 ) ];
						c[ j * ldc + ( i + 2 ) ] += alpha * temp2 + beta * c[ j * ldc + ( i + 2 ) ];
						c[ j * ldc + ( i + 3 ) ] += alpha * temp3 + beta * c[ j * ldc + ( i + 3 ) ];
					}
				} else {
					i = 0;
				}
				for( ; i < m ; i++ ) {
					temp = 0.0;
					for( l = 0 ; l < k ; l++ ) {
						temp += a[ i * lda + l ] * b[ l * ldb + j ];
					}
					c[ j * ldc + i ] += alpha * temp + beta * c[ j * ldc + i ];
				}
			} // }}}
		}
	}
} // }}}

#ifndef USE_BLAS
typedef enum CBLAS_LAYOUT {CblasRowMajor=101, CblasColMajor=102} CBLAS_LAYOUT;
typedef enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112} CBLAS_TRANSPOSE;
#endif

void my_cblas_sgemm( const CBLAS_LAYOUT layout , const CBLAS_TRANSPOSE TransA , const CBLAS_TRANSPOSE TransB , const int M , const int N , const int K , const float alpha , const float* A , const int lda , const float* B , const int ldb , const float beta , float* C , const int ldc ) {
	if( layout == CblasColMajor ) {
		my_sgemm( TransA == CblasTrans , TransB == CblasTrans , (uint_fast32_t) M , (uint_fast32_t) N , (uint_fast32_t) K , alpha , A , (uint_fast32_t) lda , B , (uint_fast32_t) ldb , beta , C , (uint_fast32_t) ldc );
	} else if ( layout == CblasRowMajor ) {
		my_sgemm( TransB == CblasTrans , TransA == CblasTrans , (uint_fast32_t) N , (uint_fast32_t) M , (uint_fast32_t) K , alpha , B , (uint_fast32_t) ldb , A , (uint_fast32_t) lda , beta , C , (uint_fast32_t) ldc );
	} else {
		throw std::length_error( std::string( CURRENT_FUNCTION_NAME ) + ": Illegal layout setting" );
	}
}

void gemm( CBLAS_TRANSPOSE trans_a , CBLAS_TRANSPOSE trans_b , uint_fast32_t rows_a , uint_fast32_t cols_b , uint_fast32_t cols_a , const float* a , const float* b , float beta , float* c , uint_fast32_t stride_a = 0 , uint_fast32_t stride_b = 0 , uint_fast32_t stride_c = 0 ) {
	stride_a = (0==stride_a)?((CblasTrans==trans_a)?rows_a:cols_a):stride_a;
	stride_b = (0==stride_b)?((CblasTrans==trans_b)?cols_a:cols_b):stride_b;
	stride_c = (0==stride_c)?cols_b:stride_c;
#ifdef USE_BLAS
	cblas_sgemm( CblasRowMajor , trans_a , trans_b , rows_a , cols_b , cols_a , 1.0 , a , stride_a , b , stride_b , beta , c , stride_c );
#else
	my_cblas_sgemm( CblasRowMajor , trans_a , trans_b , rows_a , cols_b , cols_a , 1.0 , a , stride_a , b , stride_b , beta , c , stride_c );
#endif
}

void forward_nonlin_leaky_relu( Buffer input , Buffer output , uint32_t input_channels , uint32_t output_channels , uint32_t batch_size , uint32_t height , uint32_t width ) {
	uint32_t i,n;
	n = input_channels*batch_size*height*width;
	for( i = 0 ; i < n ; i++ ) {
		(*output)[i]=(0<(*input)[i])?((*input)[i]):(0.1*(*input)[i]);
	}
}

void backward_nonlin_leaky_relu( const Buffer output , Buffer output_gradient , Buffer input_gradient , const Buffer parameters , uint32_t input_channels , uint32_t output_channels , uint32_t batch_size , uint32_t height , uint32_t width ) {
	uint32_t i,n;
	n = input_channels*batch_size*height*width;
	for( i = 0 ; i < n ; i++ ) {
		(*input_gradient)[i]=(0<(*output)[i])?((*output_gradient)[i]):(0.1*(*output_gradient)[i]);
	}
}

std::tuple<uint32_t,uint32_t,uint32_t,uint32_t> buffer_sizes_nonlin_leaky_relu( uint32_t input_channels , uint32_t output_channels , uint32_t batch_size , uint32_t height , uint32_t width ) {
	return (std::tuple<uint32_t,uint32_t,uint32_t,uint32_t>(input_channels*batch_size*height*width,input_channels*batch_size*height*width,0,0));
}

void forward_nonlin_exp( Buffer input , Buffer output , uint32_t input_channels , uint32_t output_channels , uint32_t batch_size , uint32_t height , uint32_t width ) {
	uint32_t i,n;
	n = input_channels*batch_size*height*width;
	for( i = 0 ; i < n ; i++ ) {
		(*output)[i]=exp((*input)[i]);
	}
}

void backward_nonlin_exp( const Buffer output , Buffer output_gradient , Buffer input_gradient , Buffer parameters , uint32_t input_channels , uint32_t output_channels , uint32_t batch_size , uint32_t height , uint32_t width ) {
	uint32_t i,n;
	n = input_channels*batch_size*height*width;
	for( i = 0 ; i < n ; i++ ) {
		(*input_gradient)[i]=(*output)[i]*(*output_gradient)[i];
	}
}

std::tuple<uint32_t,uint32_t,uint32_t,uint32_t> buffer_sizes_nonlin_exp( uint32_t input_channels , uint32_t output_channels , uint32_t batch_size , uint32_t height , uint32_t width ) {
	return (buffer_sizes_nonlin_leaky_relu( input_channels , output_channels , batch_size , height , width ));
}

void forward_nonlin_poly_exp( Buffer input , Buffer output , uint32_t input_channels , uint32_t output_channels , uint32_t batch_size , uint32_t height , uint32_t width ) {
	uint32_t i,n;
	n = input_channels*batch_size*height*width;
	for( i = 0 ; i < n ; i++ ) {
		(*output)[i]=pow((*input)[i]+sqrt(1+(*input)[i]*(*input)[i]),2);
	}
}

void backward_nonlin_poly_exp( Buffer output , Buffer output_gradient , Buffer input_gradient , const Buffer parameters , uint32_t input_channels , uint32_t output_channels , uint32_t batch_size , uint32_t height , uint32_t width ) {
	uint32_t i,n;
	n = input_channels*batch_size*height*width;
	for( i = 0 ; i < n ; i++ ) {
		(*input_gradient)[i]=4*pow((*output)[i],1.5)/(1+(*output)[i])*(*output_gradient)[i];
	}
}

std::tuple<uint32_t,uint32_t,uint32_t,uint32_t> buffer_sizes_nonlin_poly_exp( uint32_t input_channels , uint32_t output_channels , uint32_t batch_size , uint32_t height , uint32_t width ) {
	return (buffer_sizes_nonlin_leaky_relu( input_channels , output_channels , batch_size , height , width ));
}

void forward_bias( Buffer input , Buffer output , const Buffer parameters , uint32_t input_channels , uint32_t output_channels , uint32_t batch_size , uint32_t height , uint32_t width ) {
	uint_fast32_t i,j,k;
	uint_fast32_t l = height*width;
	uint_fast32_t o = batch_size*height*width;
	for( i = 0 ; i < input_channels ; i++ ) {
		for( j = 0 ; j < batch_size ; j++ ) {
			for( k = 0 ; k < l ; k++ ) {
				(*output)[ o * i + l * j + k ] = (*input)[ o * i + l * j + k ] + (*parameters)[ i ];
			}
		}
	}
}

void backward_bias( const Buffer output , Buffer output_gradient , Buffer input_gradient , const Buffer parameters , uint32_t input_channels , uint32_t output_channels , uint32_t batch_size , uint32_t height , uint32_t width ) {
	memcpy( input_gradient->data() , output_gradient->data() , input_channels * batch_size * height * width * sizeof( float ) );
}

void param_gradient_bias( Buffer input , Buffer output_gradient , Buffer parameter_gradient , uint32_t input_channels , uint32_t output_channels , uint32_t batch_size , uint32_t height , uint32_t width ) {
	uint_fast32_t i,j,k;
	uint_fast32_t l = height*width;
	uint_fast32_t o = batch_size*height*width;
	for( i = 0 ; i < input_channels ; i++ ) {
		(*parameter_gradient)[ i ] = 0;
	}
	for( i = 0 ; i < input_channels ; i++ ) {
		for( j = 0 ; j < batch_size ; j++ ) {
			for( k = 0 ; k < l ; k++ ) {
				(*parameter_gradient)[ i ] += (*output_gradient)[ o * i + l * j + k ];
			}
		}
	}
}

std::tuple<uint32_t,uint32_t,uint32_t,uint32_t> buffer_sizes_bias( uint32_t input_channels , uint32_t output_channels , uint32_t batch_size , uint32_t height , uint32_t width ) {
	return (std::tuple<uint32_t,uint32_t,uint32_t,uint32_t>(input_channels*batch_size*height*width,input_channels*batch_size*height*width,input_channels,0));
}

void forward_pointwise_convolution( Buffer input , Buffer output , const Buffer parameters , uint32_t input_channels , uint32_t output_channels , uint32_t batch_size , uint32_t height , uint32_t width ) {
	if( ( input.get() ) == ( output.get() ) ) {
		throw std::domain_error( std::string( CURRENT_FUNCTION_NAME ) + ": internal error" );
	}
	gemm( CblasNoTrans , CblasNoTrans , output_channels , height * width * batch_size , input_channels , parameters->data() , input->data() , (float) 0.0 , output->data() );
}

void backward_pointwise_convolution( const Buffer output , Buffer output_gradient , Buffer input_gradient , const Buffer parameters , uint32_t input_channels , uint32_t output_channels , uint32_t batch_size , uint32_t height , uint32_t width ) {
	if( ( output_gradient.get() ) == ( input_gradient.get() ) ) {
		throw std::domain_error( std::string( CURRENT_FUNCTION_NAME ) + ": internal error" );
	}
	gemm( CblasTrans , CblasNoTrans , input_channels , height * width * batch_size , output_channels , parameters->data() , output_gradient->data() , (float) 0.0 , input_gradient->data() );
}

void param_gradient_pointwise_convolution( Buffer input , Buffer output_gradient , Buffer parameter_gradient , uint32_t input_channels , uint32_t output_channels , uint32_t batch_size , uint32_t height , uint32_t width ) {
	if( ( input.get() ) == ( output_gradient.get() ) ) {
		throw std::domain_error( std::string( CURRENT_FUNCTION_NAME ) + ": internal error" );
	}
	gemm( CblasNoTrans , CblasTrans , output_channels , input_channels , height * width * batch_size , output_gradient->data() , input->data() , (float) 0.0 , parameter_gradient->data() );
}

std::tuple<uint32_t,uint32_t,uint32_t,uint32_t> buffer_sizes_pointwise_convolution( uint32_t input_channels , uint32_t output_channels , uint32_t batch_size , uint32_t height , uint32_t width ) {
	return (std::tuple<uint32_t,uint32_t,uint32_t,uint32_t>(output_channels*batch_size*height*width,input_channels*batch_size*height*width,input_channels*output_channels,0));
}

void depthwise_helper( Buffer input , Buffer output , Buffer parameters , uint32_t input_channels , uint32_t output_channels , uint32_t batch_size , uint32_t height , uint32_t width , uint32_t filter_height , uint32_t filter_width , int_fast32_t offset_sign ) { // {{{
	int_fast32_t o,p,up/*,down*/,left,right,h,i,m,n,filter_size,a,b,c,d,e,filter_offset_correction;
	zero_data(output);
	up = ( filter_height + 1 ) / 2 - 1;	/* how many filter taps are above the center tap */
	//down = filter_height / 2;			/* how many filter taps are below the center tap */
	left = ( filter_width + 1 ) / 2 - 1;	/* how many filter taps are to the left of the center tap */
	right = filter_width / 2;			/* how many filter taps are to the right of the center tap */
	filter_size = filter_height * filter_width;
	a = batch_size * height * width;
	b = a / batch_size;
	offset_sign = (offset_sign>0)?1:(-1); // i hope this helps the compiler optimising
	filter_offset_correction = ( filter_size - 1 ) * ( 1 - offset_sign ) / 2; // when going through the filter backwards, we have to start at the end, not at 0
	for( i = 0 ; i < output_channels ; i++ ) {
		for( h = 0 ; h < batch_size ; h++ ) {
			for( m = 0 ; m < height ; m++ ) {
				c = i * a + h * b + m * width;
				d = i * a + h * b + ( m + up ) * width + left;
				e = i * filter_size + filter_offset_correction;
				for( o = 0 ; o < filter_height ; o++ ) {
					if( ( o - up <= (int_fast32_t) m ) && ( (int_fast32_t) m <= height - 1 + o - up ) ) {
						for( n = 0 ; n < right ; n++ ) {
							for( p = 0 ; p <= left + n ; p++ ) {
								(*output)[ c + n ] += (*parameters)[ e + offset_sign * p ] * (*input)[ d + n - p ];
							}
						}
						for( n = right ; n < width-left ; n++ ) {
							for( p = 0 ; p < filter_width ; p++ ) {
								(*output)[ c + n ] += (*parameters)[ e + offset_sign * p ] * (*input)[ d + n - p ];
							}
						}
						for( n = width-left ; n < width ; n++ ) {
							for( p = n-width+left+1 ; p < filter_width ; p++ ) {
								(*output)[ c + n ] += (*parameters)[ e + offset_sign * p ] * (*input)[ d + n - p ];
							}
						}
					}
					d -= width;
					e += offset_sign * filter_width;
				}
			}
		}
	}
} // }}}

void forward_depthwise_convolution( const Buffer input , Buffer output , const Buffer parameters , uint32_t input_channels , uint32_t output_channels , uint32_t batch_size , uint32_t height , uint32_t width , uint32_t filter_height , uint32_t filter_width ) {
	depthwise_helper( input , output , parameters , input_channels , output_channels , batch_size , height , width , filter_height , filter_width , 1.0 );
}

void backward_depthwise_convolution( const Buffer output , const Buffer output_gradient , Buffer input_gradient , const Buffer parameters , uint32_t input_channels , uint32_t output_channels , uint32_t batch_size , uint32_t height , uint32_t width , uint32_t filter_height , uint32_t filter_width ) {
	depthwise_helper( output_gradient , input_gradient , parameters , input_channels , output_channels , batch_size , height , width , filter_height , filter_width , -1.0 );
}

void param_gradient_depthwise_convolution( Buffer input , Buffer output_gradient , Buffer parameter_gradient , uint32_t input_channels , uint32_t output_channels , uint32_t batch_size , uint32_t height , uint32_t width , uint32_t filter_height , uint32_t filter_width ) { // {{{
	int_fast32_t o,p,up/*,down*/,left,right,h,i,m,n,filter_size,a,b,c,d,e;
	up = ( filter_height + 1 ) / 2 - 1;	/* how many filter taps are above the center tap */
	//down = 0;//this->filter_height / 2;			/* how many filter taps are below the center tap */
	left = ( filter_width + 1 ) / 2 - 1;	/* how many filter taps are to the left of the center tap */
	right = filter_width / 2;			/* how many filter taps are to the right of the center tap */
	filter_size = filter_height * filter_width;
	for( i = 0 ; i < filter_size * output_channels ; i++ ) {
		(*parameter_gradient)[ i ] = 0.0;
	}
	a = height * width * batch_size;
	b = a / batch_size;
	for( i = 0 ; i < output_channels ; i++ ) {
		for( h = 0 ; h < batch_size ; h++ ) {
			for( m = 0 ; m < height ; m++ ) {
				c = i * a + h * b + m * width;
				d = i * a + h * b + ( m + up ) * width + left;
				e = i * filter_size;
				for( o = 0 ; o < filter_height ; o++ ) {
					if( ( o - up <= (int_fast32_t) m ) && ( (int_fast32_t) m <= height - 1 + o - up ) ) {
						for( n = 0 ; n < right ; n++ ) {
							for( p = 0 ; p <= left + n ; p++ ) {
								(*parameter_gradient)[ e + p ] += (*output_gradient)[ c + n ] * (*input)[ d + n - p ];
							}
						}
						for( n = right ; n < width-left ; n++ ) {
							for( p = 0 ; p < filter_width ; p++ ) {
								(*parameter_gradient)[ e + p ] += (*output_gradient)[ c + n ] * (*input)[ d + n - p ];
							}
						}
						for( n = width-left ; n < width ; n++ ) {
							for( p = n-width+left+1 ; p < filter_width ; p++ ) {
								(*parameter_gradient)[ e + p ] += (*output_gradient)[ c + n ] * (*input)[ d + n - p ];
							}
						}
					}
					d -= width;
					e += filter_width;
				}
			}
		}
	}
} // }}}

std::tuple<uint32_t,uint32_t,uint32_t,uint32_t> buffer_sizes_depthwise_convolution( uint32_t input_channels , uint32_t output_channels , uint32_t batch_size , uint32_t height , uint32_t width , uint32_t filter_height , uint32_t filter_width ) {
	return (std::tuple<uint32_t,uint32_t,uint32_t,uint32_t>(input_channels*batch_size*height*width,input_channels*batch_size*height*width,input_channels*filter_height*filter_width,0));
}

void forward_channel_normalization( const Buffer input , Buffer output , uint32_t input_channels , uint32_t output_channels , uint32_t batch_size , uint32_t height , uint32_t width , Buffer inverted_sums_copy ) { // {{{
	uint_fast32_t i,j,k,size_2,size_4;
	float sum;
	size_2 = width * height;
	size_4 = width * height * batch_size;
	for( j = 0 ; j < batch_size ; j++ ) {
		sum = 0;
		for( i = 0 ; i < input_channels ; i++ ) {
			for( k = 0 ; k < size_2 ; k++ ) {
				sum += (*input)[ j * size_2 + i * size_4 + k ];
			}
		}
		sum = ( 0 == sum )?sum:( 1.0 / sum );
		(*inverted_sums_copy)[j] = sum;
		for( i = 0 ; i < input_channels ; i++ ) {
			for( k = 0 ; k < size_2 ; k++ ) {
				(*output)[ j * size_2 + i * size_4 + k ] = sum * (*input)[ j * size_2 + i * size_4 + k ];
			}
		}
	}
} // }}}

void backward_channel_normalization( const Buffer output , const Buffer output_gradient , Buffer input_gradient , const Buffer parameters , uint32_t input_channels , uint32_t output_channels , uint32_t batch_size , uint32_t height , uint32_t width , Buffer inverted_sums_copy ) { // {{{
	uint_fast32_t i,j,k,size_2,size_4;
	float inner_product,inverted_sum;
	size_2 = width * height;
	size_4 = width * height * batch_size;
	for( j = 0 ; j < batch_size ; j++ ) {
		inner_product = 0;
		for( i = 0 ; i < input_channels ; i++ ) {
			for( k = 0 ; k < size_2 ; k++ ) {
				inner_product += (*output_gradient)[ j * size_2 + i * size_4 + k ] * (*output)[ j * size_2 + i * size_4 + k ];
			}
		}
		inverted_sum = (*inverted_sums_copy)[j];
		for( i = 0 ; i < input_channels ; i++ ) {
			for( k = 0 ; k < size_2 ; k++ ) {
				(*input_gradient)[ j * size_2 + i * size_4 + k ] = ( (*output_gradient)[ j * size_2 + i * size_4 + k ] - inner_product ) * inverted_sum;
			}
		}
	}
} // }}}

std::tuple<uint32_t,uint32_t,uint32_t,uint32_t> buffer_sizes_channel_normalization( uint32_t input_channels , uint32_t output_channels , uint32_t batch_size , uint32_t height , uint32_t width ) {
	return (std::tuple<uint32_t,uint32_t,uint32_t,uint32_t>(input_channels*batch_size*height*width,input_channels*batch_size*height*width,0,batch_size));
}

void forward_global_max_pooling( Buffer input , Buffer output , uint32_t input_channels , uint32_t output_channels , uint32_t batch_size , uint32_t height , uint32_t width , Buffer maxima_indices ) { // {{{
	uint_fast32_t i,j,k;
	int_fast32_t index;
	float tmp,accu;
	for( j = 0 ; j < input_channels ; j++ ) {
		for( i = 0 ; i < batch_size ; i++ ) {
			accu = std::numeric_limits<float>::lowest();
			index = -1;
			for( k = 0 ; k < height * width ; k++ ) {
				tmp = (*input)[ j * batch_size * height * width + i * height * width + k ];
				if( accu < tmp ) {
					accu = tmp;
					index = k;
				}
			}
			(*maxima_indices)[ i * input_channels + j ] = index;
			(*output)[ j * batch_size + i ] = accu;
		}
	}
} // }}}

void backward_global_max_pooling( const Buffer output , Buffer output_gradient , Buffer input_gradient , const Buffer parameters , uint32_t input_channels , uint32_t output_channels , uint32_t batch_size , uint32_t height , uint32_t width , Buffer maxima_indices ) { // {{{
	uint_fast32_t i,j;
	for( i = 0 ; i < input_channels * batch_size * height * width ; i++ ) {
		(*input_gradient)[i] = 0;
	}
	for( j = 0 ; j < input_channels ; j++ ) {
		for( i = 0 ; i < batch_size ; i++ ) {
			(*input_gradient)[ j * batch_size * height * width + i * height * width + (*maxima_indices)[ i * input_channels + j ] ] =
				(*output_gradient)[ j * batch_size + i ];
		}
	}
} // }}}

std::tuple<uint32_t,uint32_t,uint32_t,uint32_t> buffer_sizes_global_max_pooling( uint32_t input_channels , uint32_t output_channels , uint32_t batch_size , uint32_t height , uint32_t width ) {
	return (std::tuple<uint32_t,uint32_t,uint32_t,uint32_t>(input_channels*batch_size,input_channels*batch_size*height*width,0,batch_size*input_channels));
}

void cross_entropy_forward( Buffer output , Buffer target , Buffer objective , uint32_t channels , uint32_t batch_size , uint32_t height , uint32_t width ) { // {{{
	uint_fast32_t i,j,k,m,n;
	float eps;
	eps = 4 * std::numeric_limits<float>::epsilon();
	m = batch_size * height * width;
	n = height * width;
	for( i = 0 ; i < batch_size ; i++ ) {
		(*objective)[i] = 0;
	}
	for( j = 0 ; j < channels ; j++ ) {
		for( i = 0 ; i < batch_size ; i++ ) {
			for( k = 0 ; k < n ; k++ ) {
				(*objective)[i] -= log( MIN(1.0-eps,MAX(eps,(*output)[i*n+j*m+k])) ) * (*target)[i*n+j*m+k];
			}
		}
	}
} // }}}

void cross_entropy_backward( Buffer output , Buffer target , Buffer gradient , uint32_t channels , uint32_t batch_size , uint32_t height , uint32_t width ) {
	uint_fast32_t j,m;
	float eps = 4 * std::numeric_limits<float>::epsilon();
	m = channels * batch_size * height * width;
	for( j = 0 ; j < m ; j++ ) {
		(*gradient)[j] = (*target)[j] * ( eps + 1.0 / ( (*output)[j] + eps ) );
	}
}

void l2_forward( Buffer output , Buffer target , Buffer objective , uint32_t channels , uint32_t batch_size , uint32_t height , uint32_t width ) { // {{{
	uint_fast32_t i,j,k,m,n;
	m = batch_size * height * width;
	n = height * width;
	zero_data(objective);
	for( j = 0 ; j < channels ; j++ ) {
		for( i = 0 ; i < batch_size ; i++ ) {
			for( k = 0 ; k < n ; k++ ) {
				(*objective)[i] += 0.5 * pow( ((*output)[i*n+j*m+k]) - ((*target)[i*n+j*m+k]) , 2.0 );
			}
		}
	}
} // }}}

void l2_backward( Buffer output , Buffer target , Buffer gradient , uint32_t channels , uint32_t batch_size , uint32_t height , uint32_t width ) {
	uint_fast32_t j,m;
	m = channels * batch_size * height * width;
	for( j = 0 ; j < m ; j++ ) {
		(*gradient)[j] = (*target)[j] - (*output)[j];
	}
}

// Adam, aka adaptive moment estimation from "Adam: A Method for Stochastic Optimization" by Kingma & Ba
void adam( Buffers& layer_parameter_gradients , Buffers& layer_parameter_updates , Buffers& layer_1st_moment_estimates , Buffers& layer_2nd_moment_estimates , uint32_t iteration , float learning_rate = 0.001 , float linear_momentum_coefficient = 0.9 , float square_momentum_coefficient = 0.999 , float epsilon = 1e-8 ) { // {{{
	uint32_t i,j,m,n;
	float temp;
	float bias_correction_1 = 1.0 - pow( linear_momentum_coefficient , 1.0+iteration ); // 1.0+iteration to ensure we don't divide by zero
	float bias_correction_2 = 1.0 - pow( square_momentum_coefficient , 1.0+iteration );
	n = layer_1st_moment_estimates.size();
	for( i = 0 ; i < n ; i++ ) {
		m = layer_parameter_gradients.at(i)->size();
		if( 0 == m ) {
			continue;
		}
		for( j = 0 ; j < m ; j++ ) {
			(*(layer_1st_moment_estimates[i]))[j] = linear_momentum_coefficient * (*(layer_1st_moment_estimates[i]))[j] + ( 1.0 - linear_momentum_coefficient ) * (*(layer_parameter_gradients[i]))[j];
			(*(layer_2nd_moment_estimates[i]))[j] = square_momentum_coefficient * (*(layer_2nd_moment_estimates[i]))[j] + ( 1.0 - square_momentum_coefficient ) * (*(layer_parameter_gradients[i]))[j] * (*(layer_parameter_gradients[i]))[j];
			temp = ( sqrt( (*(layer_2nd_moment_estimates[i]))[j] / bias_correction_2 ) + epsilon ) * bias_correction_1 / learning_rate;
			(*(layer_parameter_updates[i]))[j] = (*(layer_1st_moment_estimates[i]))[j] / temp;
		}
	}
} // }}}

// SGD a.k.a. stochastic gradient descent (without momentum in this case, for the sake of simplicity)
void sgd( Buffers& layer_parameter_gradients , Buffers& layer_parameter_updates , Buffers& layer_1st_moment_estimates , Buffers& layer_2nd_moment_estimates , uint32_t iteration , float learning_rate = 0.001 , float linear_momentum_coefficient = 0.9 , float square_momentum_coefficient = 0.999 , float epsilon = 1e-8 ) {
	zip_with( layer_parameter_updates , layer_parameter_gradients , [=](float x,float y){return (learning_rate*y);} );
}

void gaussian_init_parameters( Buffers& layer_parameters , std::mt19937& rng , float standard_deviation = 0.2 ) {
	std::normal_distribution<> nd{0,standard_deviation};
	for( Buffer b : layer_parameters ) {
		for( float& x : (*b) ) {
			x = nd(rng);
		}
	}
}

void parameter_norm_clipping( Buffers& layer_parameters , float maximum_norm_per_parameter = 0.13 ) {
	uint32_t n;
	float parameter_norm = 0.0;
	n = length(layer_parameters);
	parameter_norm = norm(layer_parameters);
	if( parameter_norm > sqrt(n) * maximum_norm_per_parameter ) {
		scale( layer_parameters , sqrt(n) * maximum_norm_per_parameter / parameter_norm );
	}
}

// vim: ts=8 : autoindent : textwidth=0 : foldmethod=marker
