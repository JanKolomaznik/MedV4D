/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file ImageFactory.cpp 
 * @{ 
 **/

#include "Imaging/ImageFactory.h"
#include "Debug.h"

/**
 *  @addtogroup imaging Imaging Library
 *  @{
 *
 *  @author Jan Kolomaznik
 */

namespace M4D
{
namespace Imaging
{


void
ImageFactory::PrepareElementArrayFromTypeID( 
		int 		typeId, 
		uint32 		imageSize, 
		uint8		*& dataArray 
		)
{
	//We will generate switch over common numerical types. For more see Common.h
	TYPE_TEMPLATE_SWITCH_MACRO( 
		typeId, dataArray = (uint8 *) PrepareElementArray< TTYPE >( imageSize ) );
}


AbstractImageData*
ImageFactory::CreateImageFromDataAndTypeID(
		int 			typeId, 
		uint32 			imageSize, 
		uint8			* dataArray, 
		DimensionInfo		* info
		)
{
	AbstractImageData*	image;

	//We will generate switch over common numerical types. For more see Common.h
	TYPE_TEMPLATE_SWITCH_MACRO( 
		typeId, image = new ImageDataTemplate< TTYPE >( (TTYPE*)dataArray, info, 3, imageSize ) );

	return image;
}

void
ImageFactory::DumpImage( std::string filename, const AbstractImage & image )
{
	IMAGE_TYPE_TEMPLATE_SWITCH_MACRO( image, ImageFactory::DumpImage< TTYPE, DIM >( filename, static_cast< const Image<TTYPE,DIM> &>(image) ) );	
}

template< typename ElementType, uint32 Dimension >
void
LoadDumpedImageData( std::istream &stream, Image< ElementType, Dimension >& image )
{
	typename Image< ElementType, Dimension >::Iterator iterator = image.GetIterator();
	while( !iterator.IsEnd() && !stream.eof() ) {
		BINSTREAM_READ_MACRO( stream, *iterator );
		++iterator;
	}
}


AbstractImage::Ptr
ImageFactory::LoadDumpedImage( std::istream &stream )
{
	uint32 startMAGIC = 0;
	uint32 headerEndMagic = 0;
	uint32 formatVersion = 0;

	//Read stream header
	BINSTREAM_READ_MACRO( stream, startMAGIC );
	if( startMAGIC != DUMP_START_MAGIC_NUMBER ) {
		_THROW_ EWrongStreamBeginning();
	}
	
	BINSTREAM_READ_MACRO( stream, formatVersion );
	if( formatVersion != ACTUAL_FORMAT_VERSION ) {
		_THROW_ EWrongFormatVersion();
	}


	uint32 dimension;
	BINSTREAM_READ_MACRO( stream, dimension );
	
	uint32 elementTypeID;
	BINSTREAM_READ_MACRO( stream, elementTypeID );

	int32 *minimums = new int32[ dimension ];
	int32 *maximums = new int32[ dimension ];
	float32 *elementExtents = new float32[ dimension ];

	for ( unsigned i = 0; i < dimension; ++i ) {
		BINSTREAM_READ_MACRO( stream, minimums[i] );
		BINSTREAM_READ_MACRO( stream, maximums[i] );
		BINSTREAM_READ_MACRO( stream, elementExtents[i] );
	}

	BINSTREAM_READ_MACRO( stream, headerEndMagic );
	if( headerEndMagic != DUMP_HEADER_END_MAGIC_NUMBER ) {
		_THROW_ EWrongHeader();
	}
	//header read


	AbstractImage::Ptr image;
	TYPE_TEMPLATE_SWITCH_MACRO(
		elementTypeID,
		image = CreateEmptyImageFromExtents< TTYPE >( 
				dimension,
				minimums,
				maximums,
				elementExtents
			);
		DIMENSION_TEMPLATE_SWITCH_MACRO(
			dimension,
			LoadDumpedImageData< TTYPE, DIM >( stream, static_cast< Image< TTYPE, DIM > &>( *(image.get()) ) );
			)
	);

	delete [] minimums;
	delete [] maximums;
	delete [] elementExtents;

	return image;
}

AbstractImage::Ptr
ImageFactory::LoadDumpedImage( std::string filename )
{
	std::fstream input( filename.data(), std::ios::in | std::ios::binary );

	return LoadDumpedImage( input );
}



}/*namespace Imaging*/
}/*namespace M4D*/

/** @} */
/** @} */

