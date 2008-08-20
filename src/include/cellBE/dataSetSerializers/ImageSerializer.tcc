#ifndef IMAGE_SERIALIZER_H
#error File ImageSerializer.tcc cannot be included directly!
#else

namespace M4D
{
namespace CellBE
{

///////////////////////////////////////////////////////////////////////////////

// Common for all types of images
void
template< typename ElementType, uint8 dim>ImageSerializer
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

M4D::Imaging::AbstractDataSet *
template< typename ElementType, uint8 dim>ImageSerializer
  ::DeSerializeProperties(M4D::CellBE::NetStream &s)
{	
	int32 minimums[ dim ];
	int32 maximums[ dim ];
	float32 elExtents[ dim ];

	for( unsigned i = 0; i < dim; ++i ) {
		const DimensionExtents &dimExtents = im->GetDimensionExtents( i );

		s >> minimums[ i ];
		s >> maximums[ i ];
		s >> elExtents[ i ];
	}

	NUMERIC_TYPE_TEMPLATE_SWITCH_MACRO( elemType,
		ImageFactory::CreateEmptyImageFromExtents< TTYPE >( dim, minimums, maximums, elExtents );
		);

	return m_dataSet; // TODO initialize

}

///////////////////////////////////////////////////////////////////////////////
// 2D version
void
template< typename ElementType, 2>ImageSerializer
  ::OnDataPieceReadRequest( DataPieceHeader *header, DataBuffs &bufs)
{
}

///////////////////////////////////////////////////////////////////////////////
// 3D version
void
template< typename ElementType, 3>ImageSerializer
  ::OnDataPieceReadRequest( DataPieceHeader *header, DataBuffs &bufs)
{
}

///////////////////////////////////////////////////////////////////////////////

void
template< typename ElementType, uint8 dim>ImageSerializer
  ::OnDataSetEndRead( void)
{
}

///////////////////////////////////////////////////////////////////////////////
// 2D version
void
template< typename ElementType, 2>ImageSerializer
  ::Serialize( M4D::CellBE::iPublicJob *job)
{
  
	AbstractImage *im = (AbstractImage *) m_dataSet;

  uint32 width;
	uint32 height;
	int32 xStride;
	int32 yStride;
	ElementType *pointer = in.GetPointer( width, height, xStride, yStride );
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

void
template< typename ElementType, 3>ImageSerializer
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

void
ImageSerializer::Reset( void)
{
}

///////////////////////////////////////////////////////////////////////////////

} /*namespace Imaging*/
} /*namespace M4D*/

#endif
