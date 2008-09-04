/**
 * @ingroup cellbe 
 * @author Vaclav Klecanda 
 * @file ImageSerializer.tcc 
 * @{ 
 **/

#ifndef IMAGE_SERIALIZER_H
#error File ImageSerializer.tcc cannot be included directly!
#else

namespace M4D
{
namespace CellBE
{

///////////////////////////////////////////////////////////////////////////////

// Common for all types of images
template< typename ElementType, uint8 dim>
void
ImageSerializerBase<ElementType, dim>
  ::SerializeProperties(M4D::CellBE::NetStream &s)
{
	M4D::Imaging::AbstractImage *im = (M4D::Imaging::AbstractImage *) this->m_dataSet; // cast to sucessor

	// serialize common properties
	s << (uint16) im->GetDimension() << (uint16) im->GetElementTypeID();

	for( unsigned i = 0; i < im->GetDimension(); ++i ) {
		const M4D::Imaging::DimensionExtents &dimExtents = im->GetDimensionExtents( i );

		s << dimExtents.minimum;
		s << dimExtents.maximum;
		s << dimExtents.elementExtent;
	}
}

///////////////////////////////////////////////////////////////////////////////

template< typename ElementType, uint8 dim>
M4D::Imaging::AbstractDataSet::ADataSetPtr
ImageSerializerBase<ElementType, dim>
  ::DeSerializeProperties(M4D::CellBE::NetStream &s)
{	
	int32 minimums[ dim ];
	int32 maximums[ dim ];
	float32 elExtents[ dim ];

	for( unsigned i = 0; i < dim; ++i ) {

		s >> minimums[ i ];
		s >> maximums[ i ];
		s >> elExtents[ i ];
	}

	NUMERIC_TYPE_TEMPLATE_SWITCH_MACRO( GetNumericTypeID< ElementType >(),
		return M4D::Imaging::ImageFactory::CreateEmptyImageFromExtents< TTYPE >( dim, minimums, maximums, elExtents )
		);

	return M4D::Imaging::AbstractDataSet::ADataSetPtr(); // TODO initialize

}

///////////////////////////////////////////////////////////////////////////////

template< typename ElementType, uint8 dim>
void
ImageSerializerBase<ElementType, dim>
  ::OnDataSetEndRead( void)
{
  // TODO unlock dataSet to start execution
}

///////////////////////////////////////////////////////////////////////////////
// 2D version
///////////////////////////////////////////////////////////////////////////////

template< typename ElementType>
void
ImageSerializer< ElementType, 2>
  ::Serialize( M4D::CellBE::iPublicJob *job)
{  
	M4D::Imaging::Image<ElementType, 2> *im = (M4D::Imaging::Image<ElementType, 2> *) this->m_dataSet;

	uint32 width;
	uint32 height;
	int32 xStride;
	int32 yStride;
	ElementType *pointer = im->GetPointer( width, height, xStride, yStride );

	// put whole array at once
	DataBuff buff;
	buff.data = (void *) pointer;
	buff.len = width * height * sizeof( ElementType);

	job->PutDataPiece( buff);
}

///////////////////////////////////////////////////////////////////////////////

template< typename ElementType>
void
ImageSerializer< ElementType, 2>
  ::OnDataPieceReadRequest( DataPieceHeader *header, DataBuffs &bufs)
{
  M4D::Imaging::Image<ElementType, 2> *im = (M4D::Imaging::Image<ElementType, 2> *) this->m_dataSet;

  uint32 width;
	uint32 height;
	int32 xStride;
	int32 yStride;
	ElementType *pointer = im->GetPointer( width, height, xStride, yStride );

  size_t sliceSize = width * height;

// whole 2D image at once
  DataBuff buf( pointer, sliceSize * sizeof( ElementType));
  bufs.push_back( buf);
}

///////////////////////////////////////////////////////////////////////////////

template< typename ElementType>
void
ImageSerializer< ElementType, 2>
  ::Reset( void)
{
//nothing to do currenlty
}

///////////////////////////////////////////////////////////////////////////////
// 3D version
///////////////////////////////////////////////////////////////////////////////

template< typename ElementType>
void
ImageSerializer< ElementType, 3>
  ::Serialize( M4D::CellBE::iPublicJob *job)
{
	M4D::Imaging::Image<ElementType, 3> *im = (M4D::Imaging::Image<ElementType, 3> *) this->m_dataSet;
	
	uint32 width;
	uint32 height;
	uint32 depth;
	int32 xStride;
	int32 yStride;
	int32 zStride;
	ElementType *pointer = im->GetPointer( width, height, depth, xStride, yStride, zStride );

  // put slices as dataPieces. Suppose whole DS is serialized. Not only window part
  DataBuff buff;

  size_t sliceSize = width * height;

	for( uint32 k = 0; k < depth; ++k ) {
    buff.data = (void*) pointer;
    buff.len = sliceSize * sizeof( ElementType);
    job->PutDataPiece( buff);

    pointer += sliceSize; // move on next slice
	}
}

///////////////////////////////////////////////////////////////////////////////

template< typename ElementType>
void
ImageSerializer< ElementType, 3>
  ::OnDataPieceReadRequest( DataPieceHeader *header, DataBuffs &bufs)
{
  M4D::Imaging::Image<ElementType, 3> *im = (M4D::Imaging::Image<ElementType, 3> *) this->m_dataSet;

  uint32 width;
	uint32 height;
	uint32 depth;
	int32 xStride;
	int32 yStride;
	int32 zStride;
	ElementType *pointer = im->GetPointer( width, height, depth, xStride, yStride, zStride );

  size_t sliceSize = width * height;

  DataBuff buf;
  buf.data = pointer + ( sliceSize * m_currSlice);
  buf.len = sliceSize * sizeof( ElementType);

  bufs.push_back( buf);

  m_currSlice++;
}

///////////////////////////////////////////////////////////////////////////////

template< typename ElementType>
void
ImageSerializer< ElementType, 3>
  ::Reset( void)
{
  m_currSlice = 0;
}

///////////////////////////////////////////////////////////////////////////////

} /*namespace Imaging*/
} /*namespace M4D*/

#endif

/** @} */

