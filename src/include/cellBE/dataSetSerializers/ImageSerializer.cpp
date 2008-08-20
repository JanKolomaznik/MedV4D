
#include "ImageSerializer.h"
#include "Imaging/Image.h"

using namespace M4D::CellBE;
using namespace M4D::Imaging;


template< typename ElementType >
void
SerializeImage( Image< ElementType, 2 > &image )
{
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


template< typename ElementType >
void
SerializeImage( Image< ElementType, 3 > &image )
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
ImageSerializer::SerializeProperties(M4D::CellBE::NetStream &s)
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
ImageSerializer::DeSerializeProperties(M4D::CellBE::NetStream &s)
{

	uint16 dim, elemType;
	s >> dim >> elemType;   // get common properties
	
	int32 *minimums = new int32[ dim ];
	int32 *maximums = new int32[ dim ];
	float32 *elExtents = new float32[ dim ];

	for( unsigned i = 0; i < dim; ++i ) {
		const DimensionExtents &dimExtents = im->GetDimensionExtents( i );

		s >> minimums[ i ];
		s >> maximums[ i ];
		s >> elExtents[ i ];
	}

	//TODO create dataset

	delete [] minimums;
	delete [] maximums;
	delete [] elExtents;

	return m_dataSet; // TODO initialize

}

///////////////////////////////////////////////////////////////////////////////

void
ImageSerializer::OnDataPieceReadRequest( 
                                      DataPieceHeader *header, DataBuffs &bufs)
{
}

///////////////////////////////////////////////////////////////////////////////

void
ImageSerializer::OnDataSetEndRead( void)
{
}

///////////////////////////////////////////////////////////////////////////////

void
ImageSerializer::Serialize( M4D::CellBE::iPublicJob *job)
{
	AbstractImage *im = (AbstractImage *) m_dataSet;

	NUMERIC_TYPE_TEMPLATE_SWITCH_MACRO( image->GetElementTypeID(), 
		DIMENSION_TEMPLATE_SWITCH_MACRO( 
			image->GetDimension(), 
			SerializeImage( (const Image< TTYPE, DIM >&) *image ) 
			)
		);
}

///////////////////////////////////////////////////////////////////////////////

void
ImageSerializer::Reset( void)
{
}

///////////////////////////////////////////////////////////////////////////////
