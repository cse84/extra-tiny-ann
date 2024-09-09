/*
 * A simple libpng example program
 * http://zarb.org/~gc/html/libpng.html
 *
 * Modified by Yoshimasa Niwa to make it much simpler
 * and support all defined color_type.
 *
 * Modified further by Daniel Müller to C++ify it as much as
 * possible and throw out everything I don't need.
 *
 * Copyright 2002-2010 Guillaume Cottenceau.
 *
 * This software may be freely redistributed under the terms
 * of the X11 license.
 *
 */

#pragma pack(push, 1)
struct png_image_t {
	int32_t width, height;
	png_byte color_type;
	png_byte bit_depth;
	png_bytep *row_pointers;
};
#pragma pack(pop)

//copypasted from https://gist.github.com/niw/5963798 (How to read and write PNG file using libpng. Covers trivial method calls like png_set_filler)
//and slightly modified
struct png_image_t read_png_file( std::string filename ) {
	struct png_image_t result;
	FILE* fp = fopen(filename.c_str(), "rb"); //libpng is C code, not C++, so we have to use C file I/O
	if( NULL == fp ) {
		throw std::length_error( std::string( CURRENT_FUNCTION_NAME ) + ": error while trying to open PNG image '" + filename + "'" );
	}
	png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
	if(!png) {
		throw std::length_error( std::string( CURRENT_FUNCTION_NAME ) + ": error while trying to read PNG image '" + filename + "'" );
	}
	png_infop info = png_create_info_struct(png);
	if(!info) {
		throw std::length_error( std::string( CURRENT_FUNCTION_NAME ) + ": error while trying to read PNG image '" + filename + "'" );
	}
	if(setjmp(png_jmpbuf(png))) {
		throw std::length_error( std::string( CURRENT_FUNCTION_NAME ) + ": error while trying to read PNG image '" + filename + "'" );
	}
	png_init_io(png, fp);
	png_read_info(png, info);
	result.width      = png_get_image_width(png, info);
	result.height     = png_get_image_height(png, info);
	result.color_type = png_get_color_type(png, info);
	result.bit_depth  = png_get_bit_depth(png, info);
	// Read any color_type into 8bit depth, RGBA format.
	// See http://www.libpng.org/pub/png/libpng-manual.txt
	if(result.bit_depth == 16) {
		png_set_strip_16(png);
	}
	if(result.color_type == PNG_COLOR_TYPE_PALETTE) {
		png_set_palette_to_rgb(png);
	}
	// PNG_COLOR_TYPE_GRAY_ALPHA is always 8 or 16bit depth.
	if(result.color_type == PNG_COLOR_TYPE_GRAY && result.bit_depth < 8) {
		png_set_expand_gray_1_2_4_to_8(png);
	}
	if(png_get_valid(png, info, PNG_INFO_tRNS)) {
		png_set_tRNS_to_alpha(png);
	}
	// These color_types don't have an alpha channel, so fill it with 0xff.
	if(result.color_type == PNG_COLOR_TYPE_RGB || result.color_type == PNG_COLOR_TYPE_GRAY || result.color_type == PNG_COLOR_TYPE_PALETTE) {
		png_set_filler(png, 0xFF, PNG_FILLER_AFTER);
	}
	if(result.color_type == PNG_COLOR_TYPE_GRAY || result.color_type == PNG_COLOR_TYPE_GRAY_ALPHA) {
		png_set_gray_to_rgb(png);
	}
	png_read_update_info(png, info);
	result.row_pointers = (png_bytep*)malloc(sizeof(png_bytep) * result.height);
	for(int y = 0; y < result.height; y++) {
		result.row_pointers[y] = (png_byte*)malloc(png_get_rowbytes(png,info));
	}
	png_read_image(png, result.row_pointers);
	fclose(fp);
	png_destroy_read_struct(&png, &info, NULL);
	return result;
}
