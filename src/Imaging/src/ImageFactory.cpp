/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file ImageFactory.cpp 
 * @{ 
 **/

#include "Imaging/ImageFactory.h"
#include "common/Debug.h"

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

AbstractImage::Ptr
ImageFactory::DeserializeImage(M4D::IO::InStream &stream)
{
	uint32 startMAGIC = 0;
	uint32 datasetType = 0;
	uint32 formatVersion = 0;

	//Read stream header
	stream.Get<uint32>( startMAGIC );
	if( startMAGIC != DUMP_START_MAGIC_NUMBER ) {
		_THROW_ EWrongStreamBeginning();
	}
	
	stream.Get<uint32>( formatVersion );
	if( formatVersion != ACTUAL_FORMAT_VERSION ) {
		_THROW_ EWrongFormatVersion();
	}

	stream.Get<uint32>( datasetType );
	if( datasetType != DATASET_IMAGE ) {
		_THROW_ EWrongDatasetTypeIdentification();
	}

	return ImageFactory::DeserializeImageFromStream(stream);

	
}

void
ImageFactory::DeserializeImage(M4D::IO::InStream &stream, AbstractImage &existingImage )
{
	IMAGE_TYPE_TEMPLATE_SWITCH_MACRO( existingImage, ImageFactory::DeserializeImage< TTYPE, DIM >( stream, static_cast< Image<TTYPE,DIM> &>(existingImage) ) );
}

template< typename ElementType, unsigned Dimension >
void
ImageFactory::DeserializeImage(M4D::IO::InStream &stream, Image< ElementType, Dimension > &existingImage )
{
	uint32 startMAGIC = 0;
	uint32 datasetType = 0;
	uint32 formatVersion = 0;

	//Read stream header
	stream.Get<uint32>( startMAGIC );
	if( startMAGIC != DUMP_START_MAGIC_NUMBER ) {
		_THROW_ EWrongStreamBeginning();
	}
	
	stream.Get<uint32>( formatVersion );
	if( formatVersion != ACTUAL_FORMAT_VERSION ) {
		_THROW_ EWrongFormatVersion();
	}

	stream.Get<uint32>( datasetType );
	if( datasetType != DATASET_IMAGE ) {
		_THROW_ EWrongDatasetTypeIdentification();
	}

	uint32 dimension;
	stream.Get<uint32>( dimension );
	
	uint32 elementTypeID;
	stream.Get<uint32>( elementTypeID );

	if( dimension != Dimension || 
		(int32)elementTypeID != GetNumericTypeID< ElementType >() ) {
		_THROW_ EWrongDatasetType();
	}

	Vector< int32, Dimension > minimum;
	Vector< int32, Dimension > maximum;
	Vector< float32, Dimension > elementExtents;

	for ( unsigned i = 0; i < dimension; ++i ) {
		stream.Get<int32>( minimum[i] );
		stream.Get<int32>( maximum[i] );
		stream.Get<float32>( elementExtents[i] );
	}

	uint32 headerEndMagic = 0;
	stream.Get<uint32>( headerEndMagic );
	if( headerEndMagic != DUMP_HEADER_END_MAGIC_NUMBER ) {
		_THROW_ EWrongHeader();
	}

	ChangeImageSize( existingImage, minimum, maximum, elementExtents );

	LoadSerializedImageData( stream, existingImage );

	uint32 eoDataset = 0;
	stream.Get<uint32>( eoDataset );
	if( eoDataset != DUMP_END_MAGIC_NUMBER ) {
		_THROW_ EWrongStreamEnd();
	}
}

template< typename ElementType, uint32 Dimension >
void
LoadSerializedImageData( M4D::IO::InStream &stream, Image< ElementType, Dimension >& image )
{
	typename Image< ElementType, Dimension >::Iterator iterator = image.GetIterator();
	while( !iterator.IsEnd() && !stream.eof() ) {
		stream.Get< ElementType >( *iterator );
		++iterator;
	}
}

AbstractImage::Ptr
ImageFactory::DeserializeImageFromStream(M4D::IO::InStream &stream)
{
	uint32 dimension;
	stream.Get<uint32>( dimension );
	
	uint32 elementTypeID;
	stream.Get<uint32>( elementTypeID );

	int32 *minimums = new int32[ dimension ];
	int32 *maximums = new int32[ dimension ];
	float32 *elementExtents = new float32[ dimension ];

	for ( unsigned i = 0; i < dimension; ++i ) {
		stream.Get<int32>( minimums[i] );
		stream.Get<int32>( maximums[i] );
		stream.Get<float32>( elementExtents[i] );
	}

	uint32 headerEndMagic = 0;
	stream.Get<uint32>( headerEndMagic );
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
			LoadSerializedImageData< TTYPE, DIM >( stream, static_cast< Image< TTYPE, DIM > &>( *(image.get()) ) );
			)
	);

	uint32 eoDataset = 0;
	stream.Get<uint32>( eoDataset );
	if( eoDataset != DUMP_END_MAGIC_NUMBER ) {
		_THROW_ EWrongStreamEnd();
	}

	delete [] minimums;
	delete [] maximums;
	delete [] elementExtents;

	return image;

}

void 
ImageFactory::SerializeImage( M4D::IO::OutStream &stream, const AbstractImage &image )
{
	IMAGE_TYPE_TEMPLATE_SWITCH_MACRO( image, ImageFactory::SerializeImage< TTYPE, DIM >( stream, static_cast< const Image<TTYPE,DIM> &>(image) ) );
}

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
	AbstractImageData*	image = NULL;

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
/*
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
}*/

AbstractImage::Ptr
ImageFactory::LoadDumpedImage( std::string filename )
{
	//std::fstream input( filename.data(), std::ios::in | std::ios::binary );

	M4D::IO::FInStream input( filename );

	return DeserializeImage( input );
}



}/*namespace Imaging*/
}/*namespace M4D*/

/** @} */
/** @} */

