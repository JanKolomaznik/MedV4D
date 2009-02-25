/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file Image.tcc 
 * @{ 
 **/

#ifndef _IMAGE__H
#error File Image.tcc cannot be included directly!
#else

#ifdef _MSC_VER 
	#pragma warning (disable : 4355)
	#define WARNING_4355_DISABLED
#endif

namespace M4D
{
namespace Imaging
{



template< typename ElementType, unsigned Dim >
Image< ElementType, Dim >::Image()
: AbstractImageDim< Dim >( this->_dimExtents )
{
	for( unsigned i = 0; i < Dimension; ++i ) {
		_dimExtents[i].minimum = 0;
		_dimExtents[i].maximum = 0;
		_dimExtents[i].elementExtent = 1.0f;
		_dimOrder[i] = 0;
		_strides[i] = 0;
	}
	_imageData = typename ImageDataTemplate< ElementType >::Ptr();
	_pointer = NULL;
	_sourceDimension = 0;
	_pointerCoordinatesInSource = NULL;
}

template< typename ElementType, unsigned Dim >
Image< ElementType, Dim >::Image( AbstractImageData::APtr imageData )
: AbstractImageDim< Dim >( this->_dimExtents )
{
	try 
	{
		_imageData = ImageDataTemplate< ElementType >::CastAbstractPointer( imageData );

		FillDimensionInfo();	

	} 
	catch ( ... )
	{
		//TODO
		throw;
	}
}

template< typename ElementType, unsigned Dim >
Image< ElementType, Dim >::Image( typename ImageDataTemplate< ElementType >::Ptr imageData )
: AbstractImageDim< Dim >( this->_dimExtents )
{
	_imageData = imageData;
	
	FillDimensionInfo();
	//TODO handle exceptions
}
	
template< typename ElementType, unsigned Dim >
Image< ElementType, Dim >::Image( typename ImageDataTemplate< ElementType >::Ptr imageData, typename Image< ElementType, Dim >::SubRegion region )
: AbstractImageDim< Dim >( this->_dimExtents )
{
	_imageData = imageData;
	if( _imageData->GetDimension() < Dimension ) {
			//TODO _THROW_ exception
	}
		
	_sourceDimension = region.GetSourceDimension();
	_pointerCoordinatesInSource = new int32[_sourceDimension];
	for( unsigned i = 0; i < _sourceDimension; ++i ) {
		_pointerCoordinatesInSource[i] = region.GetPointerSourceCoordinates( i );
	}

	for( unsigned i = 0; i < Dimension; ++i ) {
		_dimOrder[i] = region.GetDimensionOrder( i );
		//TODO improve for mirrored dimension
		_dimExtents[i].minimum = _pointerCoordinatesInSource[ _dimOrder[i] ];
		_dimExtents[i].maximum = _dimExtents[i].minimum + (int32)region.GetSize( i );
		_dimExtents[i].elementExtent = _imageData->GetDimensionInfo( _dimOrder[i] ).elementExtent;
		_strides[i] = _imageData->GetDimensionInfo( _dimOrder[i] ).stride;
	}
	_pointer = region.GetPointer();
}
	
template< typename ElementType, unsigned Dim >
void
Image< ElementType, Dim >::FillDimensionInfo()
{
	if( _imageData->GetDimension() != Dimension ) {
			//TODO _THROW_ exception
	}
		
	_sourceDimension = Dimension;
	_pointerCoordinatesInSource = new int32[Dimension];
	for( unsigned i = 0; i < Dimension; ++i ) {
		_dimExtents[i].minimum = 0;
		_dimExtents[i].maximum = (int32)_imageData->GetDimensionInfo( i ).size;
		_dimExtents[i].elementExtent = _imageData->GetDimensionInfo( i ).elementExtent;
		_dimOrder[i] = i;
		_pointerCoordinatesInSource[i] = 0;
		_strides[i] = _imageData->GetDimensionInfo( _dimOrder[i] ).stride;
	}
	_pointer = &_imageData->Get( 0 );
}

template< typename ElementType, unsigned Dim >
void
Image< ElementType, Dim >::ReallocateData( typename ImageDataTemplate< ElementType >::Ptr imageData )
{
	if( imageData->GetDimension() != Dimension ) {
			//TODO _THROW_ exception
	}

	//We have to inform about changes in structure.
	this->IncStructureTimestamp();

	_imageData = imageData;

	FillDimensionInfo();
}

template< typename ElementType, unsigned Dim >
Image< ElementType, Dim >::~Image()
{
	if( _pointerCoordinatesInSource ) {
		delete [] _pointerCoordinatesInSource;
	}

}

template< typename ElementType, unsigned Dim >
inline ElementType &
Image< ElementType, Dim >::GetElement( const typename Image< ElementType, Dim >::PointType &pos )
{
	for( unsigned i = 0; i < Dimension; ++i ) {
		if( pos[i] < this->_minimum[i] || pos[i] >= this->_maximum[i] )	{
			_THROW_ ErrorHandling::EBadIndex();
		}
	}
	return *(_pointer + ( (pos - this->_minimum) * this->_strides ));
}

template< typename ElementType, unsigned Dim >
inline const ElementType &
Image< ElementType, Dim >::GetElement( const typename Image< ElementType, Dim >::PointType &pos )const
{
	for( unsigned i = 0; i < Dimension; ++i ) {
		if( pos[i] < this->_minimum[i] || pos[i] >= this->_maximum[i] )	{
			_THROW_ ErrorHandling::EBadIndex();
		}
	}
	return *(_pointer + ( (pos - this->_minimum) * this->_strides ));
}


/*
template< typename ElementType, unsigned Dim >
Image< ElementType, Dim > &
Image< ElementType, Dim >::CastAbstractImage( AbstractImage & image )
{
	//TODO - handle exception well
	return dynamic_cast< Image< ElementType, Dimension > & >( image );
}

template< typename ElementType, unsigned Dim >
const Image< ElementType, Dim > &
Image< ElementType, Dim >::CastAbstractImage( const AbstractImage & image )
{
	//TODO - handle exception well
	return dynamic_cast< const Image< ElementType, Dimension > & >( image );
}


template< typename ElementType, unsigned Dim >
typename Image< ElementType, Dim >::Ptr 
Image< ElementType, Dim >::CastAbstractImage( AbstractImage::Ptr & image )
{
	if( dynamic_cast< Image< ElementType, Dimension > * >( image.get() ) == NULL ) {
		//TODO _THROW_ exception
	}

	return boost::static_pointer_cast< Image< ElementType, Dimension > >( image );
}
*/

/*template< typename ElementType, unsigned Dim >
inline ElementType &
Image< ElementType, Dim >::GetElement( int32 x, int32 y )
{
	if( 	x < GetDimensionExtents( 0 ).minimum || 
		x >= GetDimensionExtents( 0 ).maximum 	) 
	{
		//TODO _THROW_ exception
	}
	if( 	y < GetDimensionExtents( 1 ).minimum || 
		y >= GetDimensionExtents( 1 ).maximum 	) 
	{
		//TODO _THROW_ exception
	}
	
	return _imageData->Get( x, y );
}

template< typename ElementType, unsigned Dim >
inline const ElementType &
Image< ElementType, Dim >::GetElement( int32 x, int32 y )const
{
	if( 	x < GetDimensionExtents( 0 ).minimum || 
		x >= GetDimensionExtents( 0 ).maximum 	) 
	{
		//TODO _THROW_ exception
	}
	if( 	y < GetDimensionExtents( 1 ).minimum || 
		y >= GetDimensionExtents( 1 ).maximum 	) 
	{
		//TODO _THROW_ exception
	}
	
	return _imageData->Get( x, y );
}*/

/*template< typename ElementType, unsigned Dim >
inline ElementType *
Image< ElementType, Dim >::GetPointer( 
			uint32 &width,
			uint32 &height,
			int32 &xStride,
			int32 &yStride
		  )const
{
	width = _dimExtents[0].maximum - _dimExtents[0].minimum;
	height = _dimExtents[1].maximum - _dimExtents[1].minimum;

	if( _imageData ) {
		xStride = _imageData->GetDimensionInfo( 0 ).stride;
		yStride = _imageData->GetDimensionInfo( 1 ).stride;
	
		//return &(_imageData->Get( _dimExtents[0].minimum, _dimExtents[1].minimum ));
		return _pointer;
	} else {
		xStride = 0;
		yStride = 0;

		return NULL;
	}
}*/

template< typename ElementType, unsigned Dim >
ElementType *
Image< ElementType, Dim >::GetPointer( 
			typename Image< ElementType, Dim >::SizeType &size,
			typename Image< ElementType, Dim >::PointType &strides
		  )const
{
	if( _imageData ) {
		for( unsigned i = 0; i < Dimension; ++i ) {
			size[i] = _dimExtents[i].maximum - _dimExtents[i].minimum;
			strides[i] = _imageData->GetDimensionInfo( i ).stride;
		}

		return _pointer;
	} else {
		size = SizeType( (uint32)0 );
		strides = PointType( (int32)0 );

		return NULL;
	}
}

template< typename ElementType, unsigned Dim >
template< unsigned NewDim >
typename Image< ElementType, NewDim >::Ptr
Image< ElementType, Dim >::GetRestrictedImage( 
		ImageRegion< ElementType, NewDim > region
		)
{
	Image< ElementType, NewDim > *image = new Image< ElementType, NewDim >( this->_imageData, region );
	return typename Image< ElementType, NewDim >::Ptr( image );
}
/*
template< typename ElementType, unsigned Dim >
typename Image< ElementType, 2 >::Ptr
Image< ElementType, 2 >::GetRestricted2DImage( 
		ImageRegion< ElementType, 2 > region
		)
{

}
*/
template< typename ElementType, unsigned Dim >
WriterBBoxInterface &
Image< ElementType, Dim >::SetDirtyBBox( 
			typename Image< ElementType, Dim >::PointType min,
			typename Image< ElementType, Dim >::PointType max
		)
{
	ModificationManager & modManager = _imageData->GetModificationManager();

	return modManager.AddMod( min, max );
}

template< typename ElementType, unsigned Dim >
ReaderBBoxInterface::Ptr
Image< ElementType, Dim >::GetDirtyBBox( 
			typename Image< ElementType, Dim >::PointType min,
			typename Image< ElementType, Dim >::PointType max
		)const
{
	ModificationManager & modManager = _imageData->GetModificationManager();

	return modManager.GetMod( min, max );
}

template< typename ElementType, unsigned Dim >
WriterBBoxInterface &
Image< ElementType, Dim >::SetWholeDirtyBBox()
{

	return SetDirtyBBox( 
			this->GetMinimum(),
			this->GetMaximum()
			/*GetDimensionExtents(0).minimum,
			GetDimensionExtents(1).minimum,
			GetDimensionExtents(0).maximum,
			GetDimensionExtents(1).maximum*/
			);
}

template< typename ElementType, unsigned Dim >
ReaderBBoxInterface::Ptr
Image< ElementType, Dim >::GetWholeDirtyBBox()const
{

	return GetDirtyBBox( 
			this->GetMinimum(),
			this->GetMaximum()
			);
}

template< typename ElementType, unsigned Dim >
const ModificationManager &
Image< ElementType, Dim >::GetModificationManager()const
{
	return _imageData->GetModificationManager();
}

template< typename ElementType, unsigned Dim >
typename Image< ElementType, Dim >::Iterator
Image< ElementType, Dim >::GetIterator()const
{
	SizeType size;
	PointType strides;

	ElementType * pointer = GetPointer( size, strides );
	/*uint32 width;
	uint32 height;
	int32 xStride;
	int32 yStride;

	ElementType * pointer = GetPointer( width, height, xStride, yStride );*/

	return CreateImageIteratorGen< ElementType, Dim >( pointer, size, strides, Vector< int32, Dim >() );
}

template< typename ElementType, unsigned Dim >
typename Image< ElementType, Dim >::SubRegion
Image< ElementType, Dim >::GetRegion()const
{
	return GetSubRegion( 
			CreateVector( 
				this->GetDimensionExtents(0).minimum,
				this->GetDimensionExtents(1).minimum 
				), 
			CreateVector(
				this->GetDimensionExtents(0).maximum,
				this->GetDimensionExtents(1).maximum
				)
			);
}
	
template< typename ElementType, unsigned Dim >
typename Image< ElementType, Dim >::SubRegion
Image< ElementType, Dim >::GetSubRegion(
			typename Image< ElementType, Dim >::PointType min,
			typename Image< ElementType, Dim >::PointType max
			)const
{
	//TODO - check parameters
	SizeType size;
	PointType strides;

	ElementType * pointer = GetPointer( size, strides );

	SizeType dimOrder;
	for( unsigned i = 0; i < Dimension; ++i ) {
		pointer += (min[i] - this->GetDimensionExtents(i).minimum) * strides[i];
		dimOrder[i] = i;
	}

	size = max - min;
	PointType pointerCoordinatesInSource = min;

	return CreateImageRegion( pointer, size, strides, dimOrder, pointerCoordinatesInSource );
}


///////////////////////////////////////////////////////////////////////////////

template< typename ElementType, unsigned Dim >
void
Image< ElementType, Dim >::Dump(void)
{	
	PointType stride;
	SizeType size;

	ElementType *pointer = GetPointer( size, stride );

  D_PRINT( "Type: 2D Image (" << size[0] << "x" << size[1] << "):" << std::endl);

  for( uint32 j = 0; j < size[1]; ++j ) {
    for( uint32 k = 0; k < size[0]; ++k ) {
      D_PRINT_NOENDL( *pointer << ",");
      pointer += stride[0];
    }
    D_PRINT( std::endl );
  }
}

///////////////////////////////////////////////////////////////////////////////

template< typename ElementType, unsigned Dim >
void
Image< ElementType, Dim >::SerializeClassInfo(OutStream &stream)
{
	// header
	stream.Put<uint8>( (uint8) DATASET_IMAGE);
	// template properties
	stream.Put<uint16>( this->GetElementTypeID());
	stream.Put<uint16>( this->GetDimension());
}
///////////////////////////////////////////////////////////////////////////////
template< typename ElementType, unsigned Dim >
void
Image< ElementType, Dim >::SerializeProperties(OutStream &stream)
{
	// other properties
	for( uint8 i = 0; i < Dimension; ++i ) 
	{
		stream.Put<int32>(_dimExtents[i].minimum);
		stream.Put<int32>(_dimExtents[i].maximum);
		stream.Put<float32>(_dimExtents[i].elementExtent);
	}
}
///////////////////////////////////////////////////////////////////////////////
template< typename ElementType, unsigned Dim >
void
Image< ElementType, Dim >::DeSerializeProperties(InStream &stream)
{		
	for( unsigned i = 0; i < Dimension; ++i ) {
		stream.Get<int32>(_dimExtents[ i ].minimum);		
		stream.Get<int32>(_dimExtents[ i ].maximum);
		stream.Get<float32>(_dimExtents[ i ].elementExtent);
	}
}
///////////////////////////////////////////////////////////////////////////////
template< typename ElementType, unsigned Dim >
void
Image< ElementType, Dim >::SerializeData(OutStream &stream)
{
	// actual data
	//PointType stride;
	//SizeType size;

	//ElementType *pointer = GetPointer( size, stride );

/*	DataBuff buff;
	size_t sliceSize = size[0] * size[1];

	for( uint32 k = 0; k < depth; ++k ) 
	{
	    buff.data = (void*) pointer;
	    buff.len = sliceSize * sizeof( ElementType);
	    stream.PutDataBuf( buff);
	
	    pointer += sliceSize; // move on next slice
	}*/
}
	
///////////////////////////////////////////////////////////////////////////////

template< typename ElementType, unsigned Dim >
void
Image< ElementType, Dim >::DeSerializeData(InStream &stream)
{
	//PointType stride;
	//SizeType size;

	//ElementType *pointer = GetPointer( size, stride );

	/*uint32 width;
	uint32 height;
	int32 xStride;
	int32 yStride;
	
	// get one buffer with whole slice this image represents
	ElementType *pointer = GetPointer( width, height, xStride, yStride );

	// put whole array at once
	DataBuff buff;
	buff.data = (void *) pointer;
	buff.len = width * height * sizeof( ElementType);

	stream.GetDataBuf( buff);*/

	
}

///////////////////////////////////////////////////////////////////////////////


} /*namespace Imaging*/
} /*namespace M4D*/

#ifdef WARNING_4355_DISABLED 
	#pragma warning (default : 4355)
	#undef WARNING_4355_DISABLED
#endif

#endif /*_IMAGE__H*/

/** @} */

