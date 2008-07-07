#ifndef _IMAGE__H
#error File Image.tcc cannot be included directly!
#else

namespace M4D
{
namespace Imaging
{

template< typename ElementType >
Image< ElementType, 2 >::Image()
: AbstractImage( Dimension, this->_dimExtents )
{
	for( unsigned i = 0; i < Dimension; ++i ) {
		_dimExtents[i].minimum = 0;
		_dimExtents[i].maximum = 0;
		_dimExtents[i].elementExtent = 1.0f;
	}
	_imageData = typename ImageDataTemplate< ElementType >::Ptr();
}

template< typename ElementType >
Image< ElementType, 2 >::Image( AbstractImageData::APtr imageData )
: AbstractImage( Dimension, this->_dimExtents )
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

template< typename ElementType >
Image< ElementType, 2 >::Image( typename ImageDataTemplate< ElementType >::Ptr imageData )
: AbstractImage( Dimension, this->_dimExtents )
{
	_imageData = imageData;
	
	FillDimensionInfo();
	//TODO handle exceptions
}
	
template< typename ElementType >
void
Image< ElementType, 2 >::FillDimensionInfo()
{
	if( _imageData->GetDimension() != Dimension ) {
			//TODO throw exception
	}
		
	for( unsigned i = 0; i < Dimension; ++i ) {
		_dimExtents[i].minimum = 0;
		_dimExtents[i].maximum = _imageData->GetDimensionInfo( i ).size;
		_dimExtents[i].elementExtent = _imageData->GetDimensionInfo( i ).elementExtent;
	}
}

template< typename ElementType >
void
Image< ElementType, 2 >::ReallocateData( typename ImageDataTemplate< ElementType >::Ptr imageData )
{
	if( imageData->GetDimension() != Dimension ) {
			//TODO throw exception
	}

	_imageData = imageData;

	FillDimensionInfo();
}

template< typename ElementType >
Image< ElementType, 2 >::~Image()
{

}

template< typename ElementType >
Image< ElementType, 2 > &
Image< ElementType, 2 >::CastAbstractImage( AbstractImage & image )
{
	//TODO - handle exception well
	return dynamic_cast< Image< ElementType, Dimension > & >( image );
}

template< typename ElementType >
const Image< ElementType, 2 > &
Image< ElementType, 2 >::CastAbstractImage( const AbstractImage & image )
{
	//TODO - handle exception well
	return dynamic_cast< const Image< ElementType, Dimension > & >( image );
}


template< typename ElementType >
typename Image< ElementType, 2 >::Ptr 
Image< ElementType, 2 >::CastAbstractImage( AbstractImage::AImagePtr & image )
{
	if( dynamic_cast< Image< ElementType, Dimension > * >( image.get() ) == NULL ) {
		//TODO throw exception
	}

	return boost::static_pointer_cast< Image< ElementType, Dimension > >( image );
}

template< typename ElementType >
ElementType &
Image< ElementType, 2 >::GetElement( size_t x, size_t y )
{
	if( 	x < GetDimensionExtents( 0 ).minimum || 
		x >= GetDimensionExtents( 0 ).maximum 	) 
	{
		//TODO throw exception
	}
	if( 	y < GetDimensionExtents( 1 ).minimum || 
		y >= GetDimensionExtents( 1 ).maximum 	) 
	{
		//TODO throw exception
	}
	
	return _imageData->Get( x, y );
}

template< typename ElementType >
const ElementType &
Image< ElementType, 2 >::GetElement( size_t x, size_t y )const
{
	if( 	x < GetDimensionExtents( 0 ).minimum || 
		x >= GetDimensionExtents( 0 ).maximum 	) 
	{
		//TODO throw exception
	}
	if( 	y < GetDimensionExtents( 1 ).minimum || 
		y >= GetDimensionExtents( 1 ).maximum 	) 
	{
		//TODO throw exception
	}
	
	return _imageData->Get( x, y );
}

template< typename ElementType >
ElementType *
Image< ElementType, 2 >::GetPointer( 
			size_t &width,
			size_t &height,
			int &xStride,
			int &yStride
		  )const
{
	width = _dimExtents[0].maximum - _dimExtents[0].minimum;
	height = _dimExtents[1].maximum - _dimExtents[1].minimum;

	xStride = _imageData->GetDimensionInfo( 0 ).stride;
	yStride = _imageData->GetDimensionInfo( 1 ).stride;

	return _imageData->Get( _dimExtents[0].minimum, _dimExtents[1].minimum );
}


template< typename ElementType >
typename Image< ElementType, 2 >::Ptr
Image< ElementType, 2 >::GetRestricted2DImage( 
			size_t x1, 
			size_t y1, 
			size_t x2, 
			size_t y2 
			)
{

}

//*****************************************************************************

template< typename ElementType >
Image< ElementType, 3 >::Image()
: AbstractImage( Dimension, this->_dimExtents )
{
	for( unsigned i = 0; i < Dimension; ++i ) {
		_dimExtents[i].minimum = 0;
		_dimExtents[i].maximum = 0;
		_dimExtents[i].elementExtent = 1.0f;
	}
	_imageData = typename ImageDataTemplate< ElementType >::Ptr();
}

template< typename ElementType >
Image< ElementType, 3 >::Image( AbstractImageData::APtr imageData )
: AbstractImage( 3, _dimExtents )
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

template< typename ElementType >
Image< ElementType, 3 >::Image( typename ImageDataTemplate< ElementType >::Ptr imageData )
: AbstractImage( 3, _dimExtents )
{
	_imageData = imageData;
	
	FillDimensionInfo();
	//TODO handle exceptions
}

template< typename ElementType >
void
Image< ElementType, 3 >::FillDimensionInfo()
{
	if( _imageData->GetDimension() != Dimension ) {
			//TODO throw exception
	}
		
	for( unsigned i = 0; i < Dimension; ++i ) {
		_dimExtents[i].minimum = 0;
		_dimExtents[i].maximum = _imageData->GetDimensionInfo( i ).size;
		_dimExtents[i].elementExtent = _imageData->GetDimensionInfo( i ).elementExtent;
	}
}

template< typename ElementType >
void
Image< ElementType, 3 >::ReallocateData( typename ImageDataTemplate< ElementType >::Ptr imageData )
{
	if( imageData->GetDimension() != Dimension ) {
			//TODO throw exception
	}

	_imageData = imageData;

	FillDimensionInfo();
}

template< typename ElementType >
Image< ElementType, 3 >::~Image()
{

}

template< typename ElementType >
Image< ElementType, 3 > &
Image< ElementType, 3 >::CastAbstractImage( AbstractImage & image )
{
	//TODO - handle exception well
	return dynamic_cast< Image< ElementType, Dimension > & >( image );
}

template< typename ElementType >
const Image< ElementType, 3 > &
Image< ElementType, 3 >::CastAbstractImage( const AbstractImage & image )
{
	//TODO - handle exception well
	return dynamic_cast< const Image< ElementType, Dimension > & >( image );
}

template< typename ElementType >
typename Image< ElementType, 3 >::Ptr 
Image< ElementType, 3 >::CastAbstractImage( AbstractImage::AImagePtr & image )
{
	if( dynamic_cast< Image< ElementType, Dimension > * >( image.get() ) == NULL ) {
		//TODO throw exception
	}

	return boost::static_pointer_cast< Image< ElementType, Dimension > >( image );
}

template< typename ElementType >
ElementType &
Image< ElementType, 3 >::GetElement( size_t x, size_t y, size_t z )
{
	if( 	x < GetDimensionExtents( 0 ).minimum || 
		x >= GetDimensionExtents( 0 ).maximum 	) 
	{
		//TODO throw exception
	}
	if( 	y < GetDimensionExtents( 1 ).minimum || 
		y >= GetDimensionExtents( 1 ).maximum 	) 
	{
		//TODO throw exception
	}
	if( 	z < GetDimensionExtents( 2 ).minimum || 
		z >= GetDimensionExtents( 2 ).maximum 	) 
	{
		//TODO throw exception
	}

	return _imageData->Get( x, y, z );
}

template< typename ElementType >
const ElementType &
Image< ElementType, 3 >::GetElement( size_t x, size_t y, size_t z )const
{
	if( 	x < GetDimensionExtents( 0 ).minimum || 
		x >= GetDimensionExtents( 0 ).maximum 	) 
	{
		//TODO throw exception
	}
	if( 	y < GetDimensionExtents( 1 ).minimum || 
		y >= GetDimensionExtents( 1 ).maximum 	) 
	{
		//TODO throw exception
	}
	if( 	z < GetDimensionExtents( 2 ).minimum || 
		z >= GetDimensionExtents( 2 ).maximum 	) 
	{
		//TODO throw exception
	}

	return _imageData->Get( x, y, z );
}

template< typename ElementType >
typename Image< ElementType, 2 >::Ptr
Image< ElementType, 3 >::GetRestricted2DImage( 
		size_t x1, 
		size_t y1, 
		size_t z1, 
		size_t x2, 
		size_t y2, 
		size_t z2 
		)
{

}

template< typename ElementType >
ElementType *
Image< ElementType, 3 >::GetPointer( 
			size_t &width,
			size_t &height,
			size_t &depth,
			int &xStride,
			int &yStride,
			int &zStride
		  )const
{
	width = _dimExtents[0].maximum - _dimExtents[0].minimum;
	height = _dimExtents[1].maximum - _dimExtents[1].minimum;
	depth = _dimExtents[2].maximum - _dimExtents[2].minimum;

	xStride = _imageData->GetDimensionInfo( 0 ).stride;
	yStride = _imageData->GetDimensionInfo( 1 ).stride;
	zStride = _imageData->GetDimensionInfo( 2 ).stride;

	return &_imageData->Get( _dimExtents[0].minimum, _dimExtents[1].minimum, _dimExtents[2].minimum );
}

template< typename ElementType >
typename Image< ElementType, 3 >::Ptr
Image< ElementType, 3 >::GetRestricted3DImage( 
		size_t x1, 
		size_t y1, 
		size_t z1, 
		size_t x2, 
		size_t y2, 
		size_t z2 
		)
{
}
 
template< typename ElementType >
WriterBBoxInterface &
Image< ElementType, 3 >::SetDirtyBBox( 
		size_t x1, 
		size_t y1, 
		size_t z1, 
		size_t x2, 
		size_t y2, 
		size_t z2 
		)
{
	ModificationManager & modManager = _imageData->GetModificationManager();

	return modManager.AddMod3D( 
				x1,
				y1,
				z1,
				x2,
				y2,
				z2
			);
}

template< typename ElementType >
ReaderBBoxInterface::Ptr
Image< ElementType, 3 >::GetDirtyBBox( 
		size_t x1, 
		size_t y1, 
		size_t z1, 
		size_t x2, 
		size_t y2, 
		size_t z2 
		)
{
	ModificationManager & modManager = _imageData->GetModificationManager();

	return modManager.GetMod3D( 
				x1,
				y1,
				z1,
				x2,
				y2,
				z2
			);
}

//*****************************************************************************


template< typename ElementType >
Image< ElementType, 4 >::Image()
{
	_imageData = ImageDataTemplate< ElementType >::Ptr();
}

} /*namespace Imaging*/
} /*namespace M4D*/

#endif /*_IMAGE__H*/
