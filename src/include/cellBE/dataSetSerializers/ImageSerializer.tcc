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
	AbstractImage *im = (AbstractImage *) m_dataSet; // cast to sucessor

	// serialize common properties
	s << (uint16) im->GetDimension() << (uint16) im->GetElementTypeID();

	for( unsigned i = 0; i < im->GetDimension(); ++i ) {
		const DimensionExtents &dimExtents = im->GetDimensionExtents( i );

		s << (int32)dimExtents.minimum;
		s << (int32)dimExtents.maximum;
		s << (float32)dimExtents.elementExtent;
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

	NUMERIC_TYPE_TEMPLATE_SWITCH_MACRO( ElementType,
		return ImageFactory::CreateEmptyImageFromExtents< TTYPE >( dim, minimums, maximums, elExtents )
		);

	return M4D::Imaging::AbstractDataSet::ADataSetPtr(); // TODO initialize

}

///////////////////////////////////////////////////////////////////////////////

template< typename ElementType, uint8 dim>
void
ImageSerializerBase<ElementType, dim>
  ::OnDataSetEndRead( void)
{
}

///////////////////////////////////////////////////////////////////////////////

template< typename ElementType, uint8 dim>
void
ImageSerializerBase<ElementType, dim>
  ::Reset( void)
{
}

///////////////////////////////////////////////////////////////////////////////
// 2D version
///////////////////////////////////////////////////////////////////////////////

template< typename ElementType>
void
ImageSerializer< typename ElementType, 2>
  ::OnDataPieceReadRequest( DataPieceHeader *header, DataBuffs &bufs)
{
}

///////////////////////////////////////////////////////////////////////////////
// 2D version
///////////////////////////////////////////////////////////////////////////////

template< typename ElementType>
void
ImageSerializer< typename ElementType, 2>
  ::Serialize( M4D::CellBE::iPublicJob *job)
{  
	Image<ElementType, 2> *im = (Image<ElementType, 2> *) m_dataSet;

  uint32 width;
	uint32 height;
	int32 xStride;
	int32 yStride;
	ElementType *pointer = im->GetPointer( width, height, xStride, yStride );
	for( uint32 j = 0; j < height; ++j ) {
		ElementType *tmpPointer = pointer + j*yStride;

		for( uint32 i = 0; i < width; ++i ) {
			//tady zapsani jednoho elementu
			// << *tmpPointer;

			tmpPointer += xStride;
		}
	}
}

///////////////////////////////////////////////////////////////////////////////
// 3D version
///////////////////////////////////////////////////////////////////////////////

template< typename ElementType>
void
ImageSerializer< typename ElementType, 3>
  ::OnDataPieceReadRequest( DataPieceHeader *header, DataBuffs &bufs)
{
}

///////////////////////////////////////////////////////////////////////////////
// 3D version
///////////////////////////////////////////////////////////////////////////////

template< typename ElementType>
void
ImageSerializer< typename ElementType, 3>
  ::Serialize( M4D::CellBE::iPublicJob *job)
{
  uint32 width;
	uint32 height;
	uint32 depth;
	int32 xStride;
	int32 yStride;
	int32 zStride;
	ElementType *pointer = in.GetPointer( width, height, depth, xStride, yStride, zStride );

	for( uint32 k = 0; k < depth; ++k ) {
		for( uint32 j = 0; j < height; ++j ) {
			ElementType *tmpPointer = pointer + j*yStride + k*zStride;

			for( uint32 i = 0; i < width; ++i ) {
				//tady zapsani jednoho elementu
				// << *tmpPointer;

				tmpPointer += xStride;
			}
		}
	}
}

///////////////////////////////////////////////////////////////////////////////

} /*namespace Imaging*/
} /*namespace M4D*/

#endif
