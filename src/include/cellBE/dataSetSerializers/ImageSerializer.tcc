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
ImageSerializer< typename ElementType, 2>
  ::Serialize( M4D::CellBE::iPublicJob *job)
{  
	Image<ElementType, 2> *im = (Image<ElementType, 2> *) m_dataSet;

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
ImageSerializer< typename ElementType, 2>
  ::OnDataPieceReadRequest( DataPieceHeader *header, DataBuffs &bufs)
{
  Image<ElementType, 2> *im = (Image<ElementType, 2> *) m_dataSet;

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
ImageSerializer< typename ElementType, 2>
  ::Reset( void)
{
//nothing to do currenlty
}

///////////////////////////////////////////////////////////////////////////////
// 3D version
///////////////////////////////////////////////////////////////////////////////

template< typename ElementType>
void
ImageSerializer< typename ElementType, 3>
  ::Serialize( M4D::CellBE::iPublicJob *job)
{
	Image<ElementType, 3> *im = (Image<ElementType, 3> *) m_dataSet;
	
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
ImageSerializer< typename ElementType, 3>
  ::DumpDataSet( void)
{
  Image<ElementType, 3> *im = (Image<ElementType, 3> *) m_dataSet;
	
	uint32 width;
	uint32 height;
	uint32 depth;
	int32 xStride;
	int32 yStride;
	int32 zStride;
	ElementType *pointer = im->GetPointer( width, height, depth, xStride, yStride, zStride );

  for( uint32 i = 0; i < depth; ++i ) {
    D_PRINT("Slice (" << i << "):" << endl);
    for( uint32 j = 0; j < depth; ++j ) {
      for( uint32 k = 0; k < depth; ++k ) {
        D_PRINT_NOENDL( *pointer << ",");
        pointer += xStride;
      }
      D_PRINT( endl );
    }
  }
}

///////////////////////////////////////////////////////////////////////////////

template< typename ElementType>
void
ImageSerializer< typename ElementType, 3>
  ::OnDataPieceReadRequest( DataPieceHeader *header, DataBuffs &bufs)
{
  Image<ElementType, 3> *im = (Image<ElementType, 3> *) m_dataSet;

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
ImageSerializer< typename ElementType, 3>
  ::Reset( void)
{
  m_currSlice = 0;
}

///////////////////////////////////////////////////////////////////////////////

} /*namespace Imaging*/
} /*namespace M4D*/

#endif
