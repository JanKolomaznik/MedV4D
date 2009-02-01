/**
 * @ingroup cellbe 
 * @author Vaclav Klecanda 
 * @file ImageSerializer.h 
 * @{ 
 **/

#ifndef IMAGE_SERIALIZER_H
#define IMAGE_SERIALIZER_H

#include "cellBE/AbstractDataSetSerializer.h"
#include "Imaging/Image.h"
#include "Imaging/ImageFactory.h"

namespace M4D
{
namespace CellBE
{

// commons for every dimension cases
template< typename ElementType, uint8 dim>
class ImageSerializerBase
  : public AbstractDataSetSerializer
{
public:

  ImageSerializerBase( M4D::Imaging::AbstractDataSet *dSet)
    : AbstractDataSetSerializer( dSet)
  {
  }

  ImageSerializerBase() {}

  /**
   *  Each final sucessor has to implement this functions to allow
   *  sending all properties of that particular sucessor to server.
   */
  void SerializeProperties( M4D::CellBE::NetStream &s);
  M4D::Imaging::AbstractDataSet::Ptr
    DeSerializeProperties( M4D::CellBE::NetStream &s);

  void OnDataSetEndRead( void);
};

//Tady se musi pouzit parcialni specializace
template< typename ElementType, unsigned dim>  
class ImageSerializer;

// special case for 3D images
template< typename ElementType>
class ImageSerializer< ElementType, 3 >
  : public ImageSerializerBase< ElementType, 3 >
{
	uint16 m_currSlice;

public:
	typedef ImageSerializerBase< ElementType, 3 > PredecessorType;

	ImageSerializer() {}
	ImageSerializer( M4D::Imaging::AbstractDataSet *dSet)
		: PredecessorType( dSet) {}

	/**
	 * Each special succesor should implement this functions in
	 * its own manner.
	 **/
	void Serialize( M4D::CellBE::iPublicJob *job);

	void OnDataPieceReadRequest( DataPieceHeader *header, DataBuffs &bufs);

private:
	void Reset( void);
};

// special case for 2D images
template< typename ElementType>
class ImageSerializer< ElementType, 2 >
  : public ImageSerializerBase< ElementType, 2>
{
public:
	typedef ImageSerializerBase< ElementType, 2 > PredecessorType;

	ImageSerializer() {}
	ImageSerializer( M4D::Imaging::AbstractDataSet *dSet)
	: PredecessorType( dSet) {}
	/**
	 * Each special succesor should implement this functions in
	 * its own manner.
	 **/
	void Serialize( M4D::CellBE::iPublicJob *job);

	void OnDataPieceReadRequest( DataPieceHeader *header, DataBuffs &bufs);

private:
	void Reset( void);
};

}
}

//include implementation
#include "ImageSerializer.tcc"

#endif


/** @} */

