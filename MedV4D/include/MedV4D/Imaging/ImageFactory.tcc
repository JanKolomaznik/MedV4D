/**
 * @ingroup imaging
 * @author Jan Kolomaznik
 * @file ImageFactory.tcc
 * @{
 **/

#ifndef _IMAGE_FACTORY_H
#error File ImageDataTemplate.tcc cannot be included directly!
#else

/**
 *  @addtogroup imaging Imaging Library
 *  @{
 */

#include "MedV4D/Imaging/DatasetSerializationTools.h"

namespace M4D
{
namespace Imaging {

/*template< typename ElementType, unsigned Dim >
void
ImageFactory::AllocateDataAccordingProperties(Image<ElementType, Dim> &image)
{
	Vector< int32, Dim > 	minimum;
	Vector< int32, Dim > 	maximum;
	Vector< float32, Dim >	elementExtents;

	for( unsigned i = 0; i < Dim; ++i ) {
		minimum[i] = image.GetDimensionExtents( i ).minimum;
		maximum[i] = image.GetDimensionExtents( i ).maximum;
		elementExtents[i] = image.GetDimensionExtents( i ).elementExtent;
	}

	ChangeImageSize(image, minimum, maximum, elementExtents);
}*/

template< typename ElementType, size_t Dimension >
void
ImageFactory::SerializeImage ( M4D::IO::OutStream &stream, const Image< ElementType, Dimension > &image )
{
        SerializeHeader ( stream, DATASET_IMAGE );

        /*stream.Put<uint32>( DUMP_START_MAGIC_NUMBER );
        stream.Put<uint32>( ACTUAL_FORMAT_VERSION );
        stream.Put<uint32>( DATASET_IMAGE );*/

        uint32 Dim = Dimension;
        stream.Put<uint32> ( Dim );
        uint32 numTypeID = GetNumericTypeID< ElementType >();
        stream.Put<uint32> ( numTypeID );

        for ( size_t i = 0; i < Dimension; ++i ) {
                const DimensionExtents & dimExtents = image.GetDimensionExtents ( i );
                stream.Put<int32> ( dimExtents.minimum );
                stream.Put<int32> ( dimExtents.maximum );
                stream.Put<float32> ( dimExtents.elementExtent );
        }

        stream.Put<uint32> ( DUMP_HEADER_END_MAGIC_NUMBER );

        if ( image.IsDataContinuous() ) {
                D_PRINT ( "Buffered saving of image" );
                typename Image< ElementType, Dimension >::SizeType size;
                typename Image< ElementType, Dimension >::PointType strides;
                ElementType * pointer = image.GetPointer ( size, strides );
                //TODO check invariants needed for buffered load
                stream.Put< ElementType >( pointer, VectorCoordinateProduct ( size ) );
        } else {
                D_PRINT ( "Slow saving of image" );
                typename Image< ElementType, Dimension >::Iterator iterator = image.GetIterator();
		D_COMMAND( size_t i = 0; );
                while ( !iterator.IsEnd() ) {
                        stream.Put<ElementType> ( *iterator );
                        ++iterator;
			D_COMMAND( ++i; );
                }
                D_PRINT( "Element count = " << VectorCoordinateProduct( image.GetSize() ) << " written = " << i );
                ASSERT( i == VectorCoordinateProduct( image.GetSize() ) );
        }


        stream.Put<uint32> ( DUMP_END_MAGIC_NUMBER );
}



/**
 * Function creating templated array of desired size.
 * @param ElementType
 * @param size Size of required array.
 * @exception EFailedArrayAllocation If array couldn't be allocated.
 **/
template< typename ElementType >
AlignedArrayPointer< ElementType >
PrepareElementArray ( uint32 size )
{
        try {
                //TODO
#ifdef CELL_DEFINED
                return AlignedNew< ElementType, 7 > ( size );
#else
                ElementType *arrayP = NULL;
                arrayP = new ElementType[size];

                D_PRINT ( "******** Allocating array - size:= " << size << "; pointer:= "
                          << ( int* ) arrayP << "; end:= " << ( int* ) ( arrayP + size ) );

                return AlignedArrayPointer< ElementType > ( arrayP, arrayP );
#endif
        } catch ( ... ) {
                _THROW_ EFailedArrayAllocation();
        }
}

template< typename ElementType >
ElementType*
PrepareElementArraySimple ( uint32 size )
{
        try {
                ElementType *arrayP = NULL;
                arrayP = new ElementType[size];

                D_PRINT ( "******** Allocating array ********" << std::endl
                          << "element type - " << TypeTraits<ElementType>::Typename() << std::endl
                          << "size := " << size  << std::endl
                          << "size (bytes) := " << size*sizeof ( ElementType )  << std::endl
                          << "pointer := " << ( int* ) arrayP  << std::endl
                          << "end := " << ( int* ) ( arrayP + size )
                        );

                return arrayP;
        } catch ( ... ) {
                _THROW_ EFailedArrayAllocation();
        }
}

template< typename ElementType >
AImage::Ptr
ImageFactory::CreateEmptyImageFromExtents (
        size_t			dim,
        const int32		minimums[],
        const int32		maximums[],
        const float32		elementExtents[]
)
{
        AImage::Ptr image;
        DIMENSION_TEMPLATE_SWITCH_MACRO ( dim,
                                          image = ImageFactory::CreateEmptyImageFromExtents< ElementType, DIM > (
                                                  Vector< int32, DIM > ( minimums ),
                                                  Vector< int32, DIM > ( maximums ),
                                                  Vector< float32, DIM > ( elementExtents ) );
                                        );
        return image;
}

template< typename ElementType, size_t Dim >
typename Image< ElementType, Dim >::Ptr
ImageFactory::CreateEmptyImageFromExtents (
        Vector< int32, Dim >	minimum,
        Vector< int32, Dim >	maximum,
        Vector< float32, Dim >	elementExtents
)
{
        D_BLOCK_COMMENT ( "++++++++ Create Empty Image from Extents ++++++++", "++++++++ Create Empty Image finished ++++++++" )
        D_PRINT ( "++++++++ Element type = " << TypeTraits<ElementType>::Typename() << std::endl
                  << "++++++++ Dimension    = " << Dim  << std::endl
                  << "++++++++ minimum      = " << minimum  << std::endl
                  << "++++++++ maximum      = " << maximum  << std::endl
                  << "++++++++ ElementExt   = " << elementExtents  << std::endl
                );
        typename ImageDataTemplate< ElementType >::Ptr data =
                ImageFactory::CreateEmptyImageDataTyped< ElementType, Dim> ( maximum - minimum, elementExtents );

        Image< ElementType, Dim > *img = new Image< ElementType, Dim > ( data, minimum, maximum );

        return typename Image< ElementType, Dim >::Ptr ( img );
}

template< typename ElementType, size_t tDim >
typename Image< ElementType, tDim >::Ptr
ImageFactory::CreateEmptyImageFromExtents (
                ImageExtentsRecord< tDim > aImageExtents
        )
{
	return CreateEmptyImageFromExtents< ElementType, tDim >( 
				aImageExtents.minimum,
				aImageExtents.maximum,
				aImageExtents.elementExtents
				);
}

template< typename ElementType >
AImage::Ptr
ImageFactory::CreateEmptyImage2D (
        uint32		width,
        uint32		height,
        float32		elementWidth,
        float32		elementHeight
)
{
        //TODO exceptions
        typename Image< ElementType, 2 >::Ptr ptr =
                ImageFactory::CreateEmptyImage2DTyped< ElementType > ( width, height, elementWidth, elementHeight );

        AImage::Ptr aptr =
                std::static_pointer_cast
                < AImage, Image<ElementType, 2 > > ( ptr );
        return aptr;
}

template< typename ElementType >
typename Image< ElementType, 2 >::Ptr
ImageFactory::CreateEmptyImage2DTyped (
        uint32		width,
        uint32		height,
        float32		elementWidth,
        float32		elementHeight
)
{
        D_BLOCK_COMMENT ( "++++++++ Creating 2D Image ++++++++", "++++++++ Image creation finished ++++++++" )
        D_PRINT ( std::endl
                  << "++++++++ Width      = " << width << std::endl
                  << "++++++++ Height     = " << height << std::endl
                  << "++++++++ ElementExt = " << elementWidth << "x" << elementHeight
                );

        //TODO exceptions
        typename ImageDataTemplate< ElementType >::Ptr ptr =
                ImageFactory::CreateEmptyImageData2DTyped< ElementType > ( width, height, elementWidth, elementHeight );

        Image< ElementType, 2 > *img = new Image< ElementType, 2 > ( ptr );

        return typename Image< ElementType, 2 >::Ptr ( img );
}

template< typename ElementType >
void
ImageFactory::ReallocateImage2DData (
        Image< ElementType, 2 >	&image,
        uint32			width,
        uint32			height,
        float32			elementWidth,
        float32			elementHeight
)
{
        D_BLOCK_COMMENT ( "++++++++ Reallocating 2D Image ++++++++", "++++++++ Image reallocation finished ++++++++" )
        D_PRINT ( std::endl
                  << "++++++++ Width      = " << width << std::endl
                  << "++++++++ Height     = " << height << std::endl
                  << "++++++++ ElementExt = " << elementWidth << "x" << elementHeight
                );

        //TODO exceptions
        typename ImageDataTemplate< ElementType >::Ptr ptr =
                ImageFactory::CreateEmptyImageData2DTyped< ElementType > ( width, height, elementWidth, elementHeight );

        image.ReallocateData ( ptr );
}

template< typename ElementType >
AImage::Ptr
ImageFactory::CreateEmptyImage3D (
        uint32		width,
        uint32		height,
        uint32		depth,
        float32		elementWidth,
        float32		elementHeight,
        float32		elementDepth
)
{
        //TODO exceptions
        typename Image< ElementType, 3 >::Ptr ptr =
                ImageFactory::CreateEmptyImage3DTyped< ElementType > ( width, height, depth, elementWidth, elementHeight, elementDepth );

        AImage::Ptr aptr =
                std::static_pointer_cast
                < AImage, Image<ElementType, 3 > > ( ptr );
        return aptr;
}

template< typename ElementType >
typename Image< ElementType, 3 >::Ptr
ImageFactory::CreateEmptyImage3DTyped (
        uint32		width,
        uint32		height,
        uint32		depth,
        float32		elementWidth,
        float32		elementHeight,
        float32		elementDepth
)
{
        D_BLOCK_COMMENT ( "++++++++ Creating 3D Image ++++++++", "++++++++ Image creation finished ++++++++" )
        D_PRINT ( std::endl
                  << "++++++++ Width      = " << width << std::endl
                  << "++++++++ Height     = " << height << std::endl
                  << "++++++++ Depth      = " << depth << std::endl
                  << "++++++++ ElementExt = " << elementWidth << "x" << elementHeight << "x" << elementDepth
                );

        //TODO exceptions
        typename ImageDataTemplate< ElementType >::Ptr ptr =
                ImageFactory::CreateEmptyImageData3DTyped< ElementType > ( width, height, depth, elementWidth, elementHeight, elementDepth );

        Image< ElementType, 3 > *img = new Image< ElementType, 3 > ( ptr );

        return typename Image< ElementType, 3 >::Ptr ( img );
}

template< typename ElementType >
void
ImageFactory::ReallocateImage3DData (
        Image< ElementType, 3 >	&image,
        uint32			width,
        uint32			height,
        uint32			depth,
        float32			elementWidth,
        float32			elementHeight,
        float32			elementDepth
)
{
        D_BLOCK_COMMENT ( "++++++++ Reallocating 3D Image ++++++++", "++++++++ Image reallocation finished ++++++++" )
        D_PRINT ( std::endl
                  << "++++++++ Width      = " << width << std::endl
                  << "++++++++ Height     = " << height << std::endl
                  << "++++++++ Depth      = " << depth << std::endl
                  << "++++++++ ElementExt = " << elementWidth << "x" << elementHeight << "x" << elementDepth
                );

        //TODO exceptions
        typename ImageDataTemplate< ElementType >::Ptr ptr =
                ImageFactory::CreateEmptyImageData3DTyped< ElementType > ( width, height, depth, elementWidth, elementHeight, elementDepth );

        image.ReallocateData ( ptr );
}

//**********************************************************************

template< typename ElementType >
AImageData::APtr
ImageFactory::CreateEmptyImageData2D (
        uint32		width,
        uint32		height,
        float32		elementWidth,
        float32		elementHeight
)
{
        typename ImageDataTemplate< ElementType >::Ptr ptr =
                ImageFactory::CreateEmptyImageData2DTyped< ElementType > ( width, height, elementWidth, elementHeight );

        AImageData::APtr aptr =
                std::static_pointer_cast
                < AImageData, ImageDataTemplate<ElementType> > ( ptr );

        return aptr;
}

template< typename ElementType >
typename ImageDataTemplate< ElementType >::Ptr
ImageFactory::CreateEmptyImageData2DTyped (
        uint32		width,
        uint32		height,
        float32		elementWidth,
        float32		elementHeight
)
{
        ImageDataTemplate< ElementType > *newImage;
        try {
                uint32 size = width * height;

                //Preparing informations about dimensionality.
                DimensionInfo *info = new DimensionInfo[ 2 ];
                info[0].Set ( width, 1, elementWidth );
                info[1].Set ( height, width, elementHeight );

                //Creating place for data storage.
                //ElementType *array = PrepareElementArray< ElementType >( size );
                AlignedArrayPointer< ElementType > array = PrepareElementArray< ElementType > ( size );

                //Creating new image, which is using allocated data storage.
                newImage = new ImageDataTemplate< ElementType > ( array, info, 2, size );
        } catch ( ... ) {
                //TODO exception handling
                throw;
        }

        //Returning smart pointer to abstract image class.
        return typename ImageDataTemplate< ElementType >::Ptr ( newImage );
}

template< typename ElementType >
AImageData::APtr
ImageFactory::CreateEmptyImageData3D (
        uint32		width,
        uint32		height,
        uint32		depth,
        float32		elementWidth,
        float32		elementHeight,
        float32		elementDepth
)
{
        typename ImageDataTemplate< ElementType >::Ptr ptr =
                ImageFactory::CreateEmptyImageData3DTyped< ElementType > ( width, height, depth, elementWidth, elementHeight, elementDepth );

        AImageData::APtr aptr =
                std::static_pointer_cast
                < AImageData, ImageDataTemplate<ElementType> > ( ptr );

        return aptr;
}


template< typename ElementType >
typename ImageDataTemplate< ElementType >::Ptr
ImageFactory::CreateEmptyImageData3DTyped (
        uint32		width,
        uint32		height,
        uint32		depth,
        float32		elementWidth,
        float32		elementHeight,
        float32		elementDepth
)
{
        //TODO exception handling
        uint32 size = width * height * depth;

        //Preparing informations about dimensionality.
        DimensionInfo *info = new DimensionInfo[ 3 ];
        info[0].Set ( width, 1, elementWidth );
        info[1].Set ( height, width, elementHeight );
        info[2].Set ( depth, ( width * height ), elementDepth );

        //Creating place for data storage.
        //ElementType *array = PrepareElementArray< ElementType >( size );
        AlignedArrayPointer< ElementType > array = PrepareElementArray< ElementType > ( size );

        //Creating new image, which is using allocated data storage.
        ImageDataTemplate< ElementType > *newImage =
                new ImageDataTemplate< ElementType > ( array, info, 3, size );

        //Returning smart pointer to abstract image class.
        return typename ImageDataTemplate< ElementType >::Ptr ( newImage );
}

template< typename ElementType, size_t Dim >
typename ImageDataTemplate< ElementType >::Ptr
ImageFactory::CreateEmptyImageDataTyped (
        Vector< int32, Dim > 	size,
        Vector< float32, Dim >	elementExtents
)
{
        //TODO exception handling

        //Preparing informations about dimensionality.
        DimensionInfo *info = new DimensionInfo[ Dim ];

        uint32 elementCount = 1;//width * height * depth;
        for ( unsigned i = 0; i < Dim; ++i ) {
                info[i].Set ( size[i], elementCount, elementExtents[i] );
                elementCount *= size[i];
        }

        //Creating place for data storage.
        //ElementType *array = PrepareElementArray< ElementType >( elementCount );
        AlignedArrayPointer< ElementType > array = PrepareElementArray< ElementType > ( elementCount );

        //Creating new image, which is using allocated data storage.
        ImageDataTemplate< ElementType > *newImage =
                new ImageDataTemplate< ElementType > ( array, info, Dim, elementCount );

        //Returning smart pointer to abstract image class.
        return typename ImageDataTemplate< ElementType >::Ptr ( newImage );
}

template< size_t Dim >
void
ImageFactory::ChangeImageSize (
        AImage			&image,
        Vector< int32, Dim > 	minimum,
        Vector< int32, Dim > 	maximum,
        Vector< float32, Dim >	elementExtents
)
{
        if ( image.GetDimension() != Dim ) {
                _THROW_ ErrorHandling::EBadDimension();
        }
        TYPE_TEMPLATE_SWITCH_MACRO ( image.GetElementTypeID(),
                                     ImageFactory::ChangeImageSize (
                                             static_cast< Image< TTYPE, Dim > &> ( image ),
                                             minimum,
                                             maximum,
                                             elementExtents
                                     );
                                   );
}

template< typename ElementType, size_t Dim >
void
ImageFactory::ChangeImageSize (
        Image< ElementType, Dim >	&image,
        Vector< int32, Dim > 	minimum,
        Vector< int32, Dim > 	maximum,
        Vector< float32, Dim >	elementExtents
)
{
        typename ImageDataTemplate< ElementType >::Ptr ptr =
                ImageFactory::CreateEmptyImageDataTyped< ElementType, Dim> ( maximum - minimum, elementExtents );

        image.ReallocateData ( ptr, minimum, maximum );
}

/*template< typename ElementType, uint32 Dimension >
void
ImageFactory::DumpImage( std::ostream &stream, const Image< ElementType, Dimension > & image )
{
	BINSTREAM_WRITE_MACRO( stream, DUMP_START_MAGIC_NUMBER );
	BINSTREAM_WRITE_MACRO( stream, ACTUAL_FORMAT_VERSION );

	uint32 Dim = Dimension;
	BINSTREAM_WRITE_MACRO( stream, Dim );
	uint32 numTypeID = GetNumericTypeID< ElementType >();
	BINSTREAM_WRITE_MACRO( stream, numTypeID );

	for( unsigned i = 0; i < Dimension; ++i ) {
		const DimensionExtents & dimExtents = image.GetDimensionExtents( i );
		BINSTREAM_WRITE_MACRO( stream, dimExtents.minimum );
		BINSTREAM_WRITE_MACRO( stream, dimExtents.maximum );
		BINSTREAM_WRITE_MACRO( stream, dimExtents.elementExtent );
	}

	BINSTREAM_WRITE_MACRO( stream, DUMP_HEADER_END_MAGIC_NUMBER );

	typename Image< ElementType, Dimension >::Iterator iterator = image.GetIterator();
	while( !iterator.IsEnd() ) {
		BINSTREAM_WRITE_MACRO( stream, *iterator );
		++iterator;
	}
}*/

/*template< typename ElementType, uint32 Dimension >
void
ImageFactory::DumpImage( std::string filename, const Image< ElementType, Dimension > & image )
{
	std::ofstream output( filename.c_str(), std::ios::out | std::ios::binary );

	DumpImage( output, image );
}*/

template< typename ElementType, size_t Dimension >
void
ImageFactory::DumpImage ( std::string filename, const Image< ElementType, Dimension > & image )
{
        //std::ofstream output( filename.c_str(), std::ios::out | std::ios::binary );
        M4D::IO::FOutStream output ( filename );

        SerializeImage ( output, image );
}

template< typename ElementType, size_t Dimension >
void
ImageFactory::RawDumpImage ( std::string filename, const Image< ElementType, Dimension > & image, std::ostream &aHeaderOutput )
{

        aHeaderOutput << "Data dimension      : " << Dimension << std::endl;
        aHeaderOutput << "Element type        : " << TypeTraits<ElementType>::Typename() << " (" << sizeof ( ElementType ) * 8 << " bits)" << std::endl;
        aHeaderOutput << "Data size           : " << image.GetSize() << std::endl;
        aHeaderOutput << "Element extents     : " << image.GetElementExtents() << std::endl;

        std::ofstream f ( filename.data(), std::ios_base::binary | std::ios_base::out );

        if ( image.IsDataContinuous() ) {
                D_PRINT ( "Buffered saving of image" );
                typename Image< ElementType, Dimension >::SizeType size;
                typename Image< ElementType, Dimension >::PointType strides;
                ElementType * pointer = image.GetPointer ( size,	strides );
                //TODO check invariants needed for buffered load
                f.write ( reinterpret_cast<char*> ( pointer ), VectorCoordinateProduct ( size ) * sizeof ( ElementType ) );
        } else {
                D_PRINT ( "Slow saving of image" );
                typename Image< ElementType, Dimension >::Iterator iterator = image.GetIterator();
                while ( !iterator.IsEnd() ) {
                        f.write ( reinterpret_cast<char*> ( & ( *iterator ) ), sizeof ( ElementType ) );
                        ++iterator;
                }
        }
        f.close();
}

template< typename ElementType, size_t Dimension >
void
ImageFactory::LoadRawDump ( std::string filename, Image< ElementType, Dimension > & image )
{
        //TODO test for failures
        std::ifstream f ( filename.data(), std::ios_base::binary | std::ios_base::in );
        ImageFactory::LoadRawDump< ElementType, Dimension > ( f, image );
}

template< typename ElementType, size_t Dimension >
void
ImageFactory::LoadRawDump ( std::istream &aInStream, Image< ElementType, Dimension > & image )
{
        if ( image.IsDataContinuous() ) {
                D_PRINT ( "Buffered loading of image" );
                typename Image< ElementType, Dimension >::SizeType size;
                typename Image< ElementType, Dimension >::PointType strides;
                ElementType * pointer = image.GetPointer ( size,	strides );
                //TODO check invariants needed for buffered load
                aInStream.read ( reinterpret_cast<char*> ( pointer ), VectorCoordinateProduct ( size ) * sizeof ( ElementType ) );
        } else {
                D_PRINT ( "Slow loading of image" );
                typename Image< ElementType, Dimension >::Iterator iterator = image.GetIterator();
                while ( !iterator.IsEnd() && !aInStream.eof() ) {
                        aInStream.read ( reinterpret_cast<char*> ( & ( *iterator ) ), sizeof ( ElementType ) );
                        ++iterator;
                }
        }
}

template< typename ElementType, size_t Dim  >
void
ImageFactory::AssignNewDataToImage (
        ElementType *pointer,
        Image<ElementType, Dim> &image,
        Vector< int32, Dim > 	&size,
        Vector< float32, Dim >	&elementSize )
{
        // NOTE: right now just for 3D case
        uint32 totalSize = size[0] * size[1] * size[2];

        DimensionInfo *info = new DimensionInfo[ 3 ];
        info[0].Set ( size[0], 1, elementSize[0] );
        info[1].Set ( size[1], size[0], elementSize[1] );
        info[2].Set ( size[2], ( size[0] * size[1] ), elementSize[2] );

        //Creating new image, which is using allocated data storage.
        ImageDataTemplate< ElementType > *newImage =
                new ImageDataTemplate< ElementType > ( pointer, info, 3, totalSize );

        typename ImageDataTemplate< ElementType >::Ptr container =
                typename ImageDataTemplate< ElementType >::Ptr ( newImage );

        image.ReallocateData ( container );
}

} /*namespace Imaging*/
} /*namespace M4D*/

/** @} */

#endif /*_IMAGE_FACTORY_H*/


/** @} */

