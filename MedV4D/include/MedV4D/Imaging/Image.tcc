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

#define DEBUG_DATASET_SERIALIZATION 12

namespace M4D
{
namespace Imaging {



template< typename ElementType, size_t Dim >
Image< ElementType, Dim >::Image()
		: AImageDim< Dim > ( this->_dimExtents ), _isDataContinuous ( false )
{
	for ( unsigned i = 0; i < Dimension; ++i ) {
		this->_minimum[i] = _dimExtents[i].minimum = 0;
		this->_maximum[i] = _dimExtents[i].maximum = 0;
		this->_size[i] = this->_maximum[i] - this->_minimum[i];
		this->_elementExtents[i] = _dimExtents[i].elementExtent = 1.0f;
		_dimOrder[i] = 0;
		_strides[i] = 0;
	}
	_imageData = typename ImageDataTemplate< ElementType >::Ptr();
	_pointer = NULL;
	_sourceDimension = 0;
	_pointerCoordinatesInSource = NULL;

	_minimalValue = TypeTraits< ElementType >::Min;
	_maximalValue = TypeTraits< ElementType >::Max;
}

template< typename ElementType, size_t Dim >
Image< ElementType, Dim >::Image ( AImageData::APtr imageData )
		: AImageDim< Dim > ( this->_dimExtents ), _isDataContinuous ( true )
{
	try {
		_imageData = ImageDataTemplate< ElementType >::CastAbstractPointer ( imageData );

		FillDimensionInfo();

		_minimalValue = TypeTraits< ElementType >::Min;
		_maximalValue = TypeTraits< ElementType >::Max;
	} catch ( ... ) {
		//TODO
		throw;
	}
}

template< typename ElementType, size_t Dim >
Image< ElementType, Dim >::Image ( typename ImageDataTemplate< ElementType >::Ptr imageData )
		: AImageDim< Dim > ( this->_dimExtents ), _isDataContinuous ( true )
{
	_imageData = imageData;

	FillDimensionInfo();

	_minimalValue = TypeTraits< ElementType >::Min;
	_maximalValue = TypeTraits< ElementType >::Max;
	//TODO handle exceptions
}

template< typename ElementType, size_t Dim >
Image< ElementType, Dim >::Image ( typename ImageDataTemplate< ElementType >::Ptr imageData, typename Image< ElementType, Dim >::SubRegion region )
		: AImageDim< Dim > ( this->_dimExtents ), _isDataContinuous ( false ) //TODO check if region contains whole buffer
{
	_imageData = imageData;
	if ( _imageData->GetDimension() < Dimension ) {
		_THROW_ ErrorHandling::EBadParameter ( "Creating image from buffer of wrong dimension." );
	}

	_sourceDimension = region.GetSourceDimension();
	_pointerCoordinatesInSource = new int32[_sourceDimension];
	for ( unsigned i = 0; i < _sourceDimension; ++i ) {
		_pointerCoordinatesInSource[i] = region.GetPointerSourceCoordinates ( i );
	}

	for ( unsigned i = 0; i < Dimension; ++i ) {
		_dimOrder[i] = region.GetDimensionOrder ( i );
		//TODO improve for mirrored dimension
		this->_minimum[i] = _dimExtents[i].minimum = _pointerCoordinatesInSource[ _dimOrder[i] ];
		this->_maximum[i] = _dimExtents[i].maximum = _dimExtents[i].minimum + ( int32 ) region.GetSize ( i );
		this->_size[i] = this->_maximum[i] - this->_minimum[i];
		this->_elementExtents[i] = _dimExtents[i].elementExtent = _imageData->GetDimensionInfo ( _dimOrder[i] ).elementExtent;
		_strides[i] = _imageData->GetDimensionInfo ( _dimOrder[i] ).stride;
	}
	_pointer = region.GetPointer();

	_minimalValue = TypeTraits< ElementType >::Min;
	_maximalValue = TypeTraits< ElementType >::Max;
}

template< typename ElementType, size_t Dim >
Image< ElementType, Dim >::Image (
	typename ImageDataTemplate< ElementType >::Ptr	imageData,
	typename Image< ElementType, Dim >::PointType	minimum,
	typename Image< ElementType, Dim >::PointType	maximum
)
		: AImageDim< Dim > ( this->_dimExtents ), _isDataContinuous ( true )
{
	if ( imageData->GetDimension() != Dimension ) {
		_THROW_ ErrorHandling::EBadParameter ( "Creating image from buffer of wrong dimension." );
	}
	_imageData = imageData;

	_sourceDimension = Dimension;
	_pointerCoordinatesInSource = new int32[_sourceDimension];
	for ( unsigned i = 0; i < _sourceDimension; ++i ) {
		_pointerCoordinatesInSource[i] = 0;
	}
	this->_minimum = minimum;
	this->_maximum = maximum;
	for ( unsigned i = 0; i < Dimension; ++i ) {
		_dimOrder[i] = i;
		_dimExtents[i].minimum = this->_minimum[i];
		_dimExtents[i].maximum = this->_maximum[i];
		this->_size[i] = this->_maximum[i] - this->_minimum[i];
		this->_elementExtents[i] = _dimExtents[i].elementExtent = _imageData->GetDimensionInfo ( i ).elementExtent;
		_strides[i] = _imageData->GetDimensionInfo ( i ).stride;
	}
	_pointer = &_imageData->Get ( 0 );

	_minimalValue = TypeTraits< ElementType >::Min;
	_maximalValue = TypeTraits< ElementType >::Max;
}

template< typename ElementType, size_t Dim >
void
Image< ElementType, Dim >::FillDimensionInfo()
{
	if ( _imageData->GetDimension() != Dimension ) {
		_THROW_ ErrorHandling::EBadParameter ( "Creating image from buffer of wrong dimension." );
	}

	_sourceDimension = Dimension;
	_pointerCoordinatesInSource = new int32[Dimension];
	for ( unsigned i = 0; i < Dimension; ++i ) {
		this->_minimum[i] = _dimExtents[i].minimum = 0;
		this->_maximum[i] = _dimExtents[i].maximum = ( int32 ) _imageData->GetDimensionInfo ( i ).size;
		this->_size[i] = this->_maximum[i] - this->_minimum[i];
		this->_elementExtents[i] = _dimExtents[i].elementExtent = _imageData->GetDimensionInfo ( i ).elementExtent;
		_dimOrder[i] = i;
		_pointerCoordinatesInSource[i] = 0;
		_strides[i] = _imageData->GetDimensionInfo ( _dimOrder[i] ).stride;
	}
	_pointer = &_imageData->Get ( 0 );
}

template< typename ElementType, size_t Dim >
void
Image< ElementType, Dim >::ReallocateData ( typename ImageDataTemplate< ElementType >::Ptr imageData )
{
	if ( imageData->GetDimension() != Dimension ) {
		//TODO _THROW_ exception
	}

	//We have to inform about changes in structure.
	this->IncStructureTimestamp();

	_imageData = imageData;

	FillDimensionInfo();

	_minimalValue = TypeTraits< ElementType >::Min;
	_maximalValue = TypeTraits< ElementType >::Max;
}

template< typename ElementType, size_t Dim >
void
Image< ElementType, Dim >::ReallocateData (
	typename ImageDataTemplate< ElementType >::Ptr imageData,
	typename Image< ElementType, Dim >::PointType	minimum,
	typename Image< ElementType, Dim >::PointType	maximum	)
{
	if ( imageData->GetDimension() != Dimension ) {
		_THROW_ ErrorHandling::EBadParameter ( "Creating image from buffer of wrong dimension." );
	}
	//We have to inform about changes in structure.
	this->IncStructureTimestamp();
	_imageData = imageData;

	_sourceDimension = Dimension;
	_pointerCoordinatesInSource = new int32[_sourceDimension];
	for ( unsigned i = 0; i < _sourceDimension; ++i ) {
		_pointerCoordinatesInSource[i] = 0;
	}
	this->_minimum = minimum;
	this->_maximum = maximum;
	for ( unsigned i = 0; i < Dimension; ++i ) {
		_dimOrder[i] = i;
		_dimExtents[i].minimum = this->_minimum[i];
		_dimExtents[i].maximum = this->_maximum[i];
		this->_size[i] = this->_maximum[i] - this->_minimum[i];
		this->_elementExtents[i] = _dimExtents[i].elementExtent = _imageData->GetDimensionInfo ( i ).elementExtent;
		_strides[i] = _imageData->GetDimensionInfo ( i ).stride;
	}
	_pointer = &_imageData->Get ( 0 );

	_minimalValue = TypeTraits< ElementType >::Min;
	_maximalValue = TypeTraits< ElementType >::Max;
}

template< typename ElementType, size_t Dim >
Image< ElementType, Dim >::~Image()
{
	if ( _pointerCoordinatesInSource ) {
		delete [] _pointerCoordinatesInSource;
	}

}

template< typename ElementType, size_t Dim >
inline ElementType &
Image< ElementType, Dim >::GetElement ( const typename Image< ElementType, Dim >::PointType &pos )
{
	if ( ! ( pos >= this->_minimum && pos < this->_maximum ) ) {
		_THROW_ ErrorHandling::EBadIndex (
			TO_STRING ( "Parameter 'pos = [" << pos << "]' pointing outside of the image. Min = [" <<
				    this->_minimum << "]; Max = [" << this->_maximum << "]" ) );
	}
	return * ( _pointer + ( ( pos - this->_minimum ) * this->_strides ) );
}

template< typename ElementType, size_t Dim >
inline const ElementType &
Image< ElementType, Dim >::GetElement ( const typename Image< ElementType, Dim >::PointType &pos ) const
{
	if ( ! ( pos >= this->_minimum && pos < this->_maximum ) ) {
		_THROW_ ErrorHandling::EBadIndex (
			TO_STRING ( "Parameter 'pos = [" << pos << "]' pointing outside of the image. Min = [" <<
				    this->_minimum << "]; Max = [" << this->_maximum << "]" ) );
	}
	return * ( _pointer + ( ( pos - this->_minimum ) * this->_strides ) );
}


template< typename ElementType, size_t Dim >
inline ElementType
Image< ElementType, Dim >::GetElementWorldCoords ( const Vector< float32, Dim > &pos ) const
{
	return GetElement ( GetElementCoordsFromWorldCoords( pos ) );
}

template< typename ElementType, size_t Dim >
inline ElementType &
Image< ElementType, Dim >::GetElementWorldCoords ( const Vector< float32, Dim > &pos )
{
	return GetElement ( GetElementCoordsFromWorldCoords( pos ) );
}

template< typename ElementType, size_t Dim >
Vector< int32, Dim >
Image< ElementType, Dim >::GetElementCoordsFromWorldCoords ( const Vector< float32, Dim > &pos )const
{
	Vector< int32, Dim > coords;
	for ( unsigned i = 0; i < Dim; ++i ) {
		coords[i] = ROUND ( pos[i] / this->_dimExtents[i].elementExtent - 0.5f);
	}
	return coords;
}

template< typename ElementType, size_t Dim >
Vector< float32, Dim >
Image< ElementType, Dim >::GetWorldCoordsFromElementCoords ( const Vector< int32, Dim > &coords)const
{
	Vector< float32, Dim > pos;
	for ( unsigned i = 0; i < Dim; ++i ) {
		pos[i] = coords[i] * this->_dimExtents[i].elementExtent;
	}
	return pos;
}

template< typename ElementType, size_t Dim >
ElementType *
Image< ElementType, Dim >::GetPointer (
	typename Image< ElementType, Dim >::SizeType &size,
	typename Image< ElementType, Dim >::PointType &strides
) const
{
	if ( _imageData ) {
		for ( unsigned i = 0; i < Dimension; ++i ) {
			size[i] = _dimExtents[i].maximum - _dimExtents[i].minimum;
			strides[i] = _imageData->GetDimensionInfo ( i ).stride;
		}

		return _pointer;
	} else {
		size = SizeType ( ( uint32 ) 0 );
		strides = PointType ( ( int32 ) 0 );

		return NULL;
	}
}


template< typename ElementType, size_t Dim >
ElementType *
Image< ElementType, Dim >::GetPointer() const
{
	if ( _imageData ) {
		return _pointer;
	} else {
		return NULL;
	}
}


template< typename ElementType, size_t Dim >
template< size_t NewDim >
typename Image< ElementType, NewDim >::Ptr
Image< ElementType, Dim >::GetRestrictedImage (
	ImageRegion< ElementType, NewDim > region
) const
{
	Image< ElementType, NewDim > *image = new Image< ElementType, NewDim > ( this->_imageData, region );
	return typename Image< ElementType, NewDim >::Ptr ( image );
}

template< typename ElementType, size_t Dim >
ElementType
Image< ElementType, Dim >::GetLowBand() const
{
	return _minimalValue;
}

template< typename ElementType, size_t Dim >
ElementType
Image< ElementType, Dim >::GetHighBand() const
{
	return _maximalValue;
}


template< typename ElementType, size_t Dim >
WriterBBoxInterface &
Image< ElementType, Dim >::SetDirtyBBox (
	typename Image< ElementType, Dim >::PointType min,
	typename Image< ElementType, Dim >::PointType max
)
{
	ModificationManager & modManager = _imageData->GetModificationManager();

	WriterBBoxInterface * wbbox = NULL;
	DIMENSION_TEMPLATE_SWITCH_MACRO ( _sourceDimension, {
		Vector< int32, DIM > pmin = PosInSource< DIM > ( min );
		Vector< int32, DIM > pmax = PosInSource< DIM > ( max );
		/*return modManager.AddMod( pmin, pmax );*/
		wbbox = & ( modManager.AddMod<DIM> ( pmin, pmax ) );
	} );

	ASSERT ( wbbox );//TODO prevent warnings
	return *wbbox;
}

template< typename ElementType, size_t Dim >
ReaderBBoxInterface::Ptr
Image< ElementType, Dim >::GetDirtyBBox (
	typename Image< ElementType, Dim >::PointType min,
	typename Image< ElementType, Dim >::PointType max) const
{
	ModificationManager & modManager = _imageData->GetModificationManager();
	ReaderBBoxInterface::Ptr bbox;

	DIMENSION_TEMPLATE_SWITCH_MACRO ( _sourceDimension, {
		Vector< int32, DIM > pmin = PosInSource< DIM > ( min );
		Vector< int32, DIM > pmax = PosInSource< DIM > ( max );
		bbox = modManager.GetMod<DIM> ( pmin, pmax );
	} );
	return bbox;
}

template< typename ElementType, size_t Dim >
WriterBBoxInterface &
Image< ElementType, Dim >::SetWholeDirtyBBox()
{
	return SetDirtyBBox (
		       this->GetMinimum(),
		       this->GetMaximum()
	       );
}

template< typename ElementType, size_t Dim >
ReaderBBoxInterface::Ptr
Image< ElementType, Dim >::GetWholeDirtyBBox() const
{
	return GetDirtyBBox (
		       this->GetMinimum(),
		       this->GetMaximum()
	       );
}

template< typename ElementType, size_t Dim >
void
Image< ElementType, Dim >::getChangedRegionSinceTimestamp(
		typename Image< ElementType, Dim >::PointType &aMinimum,
		typename Image< ElementType, Dim >::PointType &aMaximum,
		const Common::TimeStamp &aTimestamp
		) const
{
	const ModificationManager & manager = GetModificationManager();
	ModificationManager::ConstChangeIterator it = manager.GetChangeBBox( aTimestamp );
	if ( it == manager.ChangesEnd() ) {
		aMinimum = this->GetMinimum();
		aMaximum = this->GetMaximum();
		return;
	}
	//PointType tmpMin = it->
	ModificationBBox bbox = (*it)->GetBoundingBox();
	++it;
	while( it != manager.ChangesEnd() ) {
		bbox.merge( (*it)->GetBoundingBox() );
		++it;
	}
	DIMENSION_TEMPLATE_SWITCH_MACRO ( _sourceDimension, {
		aMinimum = SourcePosInImage< DIM > ( Vector<int,DIM>( bbox.getMinimum() ) );
		aMaximum = SourcePosInImage< DIM > ( Vector<int,DIM>( bbox.getMaximum() ) );
	} );
	aMinimum = M4D::maxVect<int,Dimension>( aMinimum, this->GetMinimum() );
	aMaximum = M4D::minVect<int,Dimension>( aMaximum, this->GetMaximum() );
/*	aMinimum = PointType( bbox.getMinimum() );
	aMaximum = PointType( bbox.getMaximum() );*/
}

template< typename ElementType, size_t Dim >
const ModificationManager &
Image< ElementType, Dim >::GetModificationManager() const
{
	if ( ! _imageData ) {
		_THROW_ ErrorHandling::EObjectUnavailable ( "Image data buffer unavailable." );
	}
	return _imageData->GetModificationManager();
}

template< typename ElementType, size_t Dim >
typename Image< ElementType, Dim >::Iterator
Image< ElementType, Dim >::GetIterator() const
{
	SizeType size;
	PointType strides;

	ElementType * pointer = GetPointer ( size, strides );
	/*uint32 width;
	uint32 height;
	int32 xStride;
	int32 yStride;

	ElementType * pointer = GetPointer( width, height, xStride, yStride );*/

	return Iterator ( pointer, this->_minimum, this->_maximum, strides, this->_minimum );
}

template< typename ElementType, size_t Dim >
typename Image< ElementType, Dim >::SubRegion
Image< ElementType, Dim >::GetRegion() const
{
	return GetSubRegion ( this->_minimum, this->_maximum );
}

template< typename ElementType, size_t Dim >
typename Image< ElementType, Dim >::SubRegion
Image< ElementType, Dim >::GetSubRegion (
	typename Image< ElementType, Dim >::PointType min,
	typename Image< ElementType, Dim >::PointType max
) const
{
	if ( ! ( min >= this->_minimum ) ) {
		_THROW_ ErrorHandling::EBadParameter ( TO_STRING ( "Parameter 'min = [" << min << "]' pointing outside of image!" ) );
	}
	if ( ! ( max <= this->_maximum ) ) {
		_THROW_ ErrorHandling::EBadParameter ( TO_STRING ( "Parameter 'max = [" << max << "]' pointing outside of image!" ) );
	}


	ElementType * pointer = _pointer;
	PointType size = max - min;

	pointer += ( min - this->_minimum ) * _strides;

	PointType pointerCoordinatesInSource = min;

	return CreateImageRegion<ElementType, Dim, Dim > ( pointer, size, _strides, this->_elementExtents, _dimOrder, pointerCoordinatesInSource );
}

template< typename ElementType, size_t Dim >
typename Image< ElementType, Dim >::SliceRegion
Image< ElementType, Dim >::GetSlice ( int32 slice, uint32 perpAxis ) const
{
	SubRegion region = GetRegion();
	return region.GetSlice ( slice, perpAxis );
}


///////////////////////////////////////////////////////////////////////////////

template< typename ElementType, size_t Dim >
void
Image< ElementType, Dim >::Dump ( std::ostream &s ) const
{
	PointType stride;
	SizeType size;

	ElementType *pointer = GetPointer ( size, stride );

	s << "Type: 2D Image (" << size[0] << "x" << size[1] << "):" << std::endl;

	for ( uint32 j = 0; j < size[1]; ++j ) {
		for ( uint32 k = 0; k < size[0]; ++k ) {
			s << *pointer << ",";
			pointer += stride[0];
		}
		s << std::endl;
	}
}

///////////////////////////////////////////////////////////////////////////////

template< typename ElementType, size_t Dim >
void
Image< ElementType, Dim >::SerializeClassInfo ( M4D::IO::OutStream &stream )
{
	// header
	stream.Put<uint8> ( ( uint8 ) DATASET_IMAGE );
	// template properties
	stream.Put<uint16> ( this->GetElementTypeID() );
	stream.Put<uint16> ( this->GetDimension() );
}
///////////////////////////////////////////////////////////////////////////////
template< typename ElementType, size_t Dim >
void
Image< ElementType, Dim >::SerializeProperties ( M4D::IO::OutStream &stream )
{
	// other properties
	for ( uint8 i = 0; i < Dimension; ++i ) {
		stream.Put<int32> ( _dimExtents[i].minimum );
		stream.Put<int32> ( _dimExtents[i].maximum );
		stream.Put<float32> ( _dimExtents[i].elementExtent );
	}
}
///////////////////////////////////////////////////////////////////////////////
template< typename ElementType, size_t Dim >
void
Image< ElementType, Dim >::SerializeData ( M4D::IO::OutStream &stream ) const
{
	// actual data
	PointType stride;
	SizeType size;

	ElementType *pointer = GetPointer ( size, stride );

	M4D::IO::DataBuff buff;

	DL_PRINT ( DEBUG_DATASET_SERIALIZATION, "SerializeData begin" );
	// note: this expects image that represent the WHOLE buffer NOT only window
	switch ( this->GetDimension() ) {
	case 3:
		for ( register uint32 i = 0; i < size[2]; i++ ) {
			for ( register uint32 j = 0; j < size[1]; j++ ) {
				buff.data = ( void* ) pointer;
				buff.len = size[0] * sizeof ( ElementType );	// 1 row
				stream.PutDataBuf ( buff );

				DL_PRINT ( DEBUG_DATASET_SERIALIZATION, j << "th row of " << i <<"th slice" );
				pointer += stride[1]; // move on next row
			}
		}
	}

	DL_PRINT ( DEBUG_DATASET_SERIALIZATION, "SerializeData done" );
}

///////////////////////////////////////////////////////////////////////////////

template< typename ElementType, size_t Dim >
void
Image< ElementType, Dim >::DeSerializeData ( M4D::IO::InStream &stream )
{
	PointType stride;
	SizeType size;

	ElementType *pointer = GetPointer ( size, stride );

	DL_PRINT ( DEBUG_DATASET_SERIALIZATION, "DeSerializeData begin" );
	M4D::IO::DataBuff buff;
	// note: this expects image that represent the WHOLE buffer NOT only window
	switch ( this->GetDimension() ) {
	case 3:
		for ( register uint32 i = 0; i < size[2]; i++ ) {
			for ( register uint32 j = 0; j < size[1]; j++ ) {
				buff.data = ( void* ) pointer;
				buff.len = size[0] * sizeof ( ElementType ); // 1 row
				stream.GetDataBuf<ElementType> ( buff );

				pointer += stride[1]; // move on next slice

				DL_PRINT ( DEBUG_DATASET_SERIALIZATION, j << "th row of " << i <<"th slice" );
			}
		}
	}
	DL_PRINT ( DEBUG_DATASET_SERIALIZATION, "DeSerializeData done" );
}

///////////////////////////////////////////////////////////////////////////////


template< typename ElementType, size_t Dim >
void
Image< ElementType, Dim >::fill(ElementType aValue)
{
	auto it = GetIterator();
	while (!it.IsEnd()) {
		*it = aValue;
		++it;
	}
}

} /*namespace Imaging*/
} /*namespace M4D*/

#ifdef WARNING_4355_DISABLED
#pragma warning (default : 4355)
#undef WARNING_4355_DISABLED
#endif

#endif /*_IMAGE__H*/

/** @} */

