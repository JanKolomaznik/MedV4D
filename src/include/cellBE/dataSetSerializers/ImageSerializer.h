#ifndef IMAGE_SERIALIZER_H
#define IMAGE_SERIALIZER_H

#include "../AbstractDataSetSerializer.h"
#include "Imaging/Image.h"

namespace M4D
{
namespace CellBE
{

// commons for every dimension cases
template< typename ElementType, uint8 dim>
class ImageSerializerBase		//musi se jmenovat jinak nez ta dlasi trida
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
  M4D::Imaging::AbstractDataSet::ADataSetPtr
    DeSerializeProperties( M4D::CellBE::NetStream &s);

  void OnDataSetEndRead( void);

private:
  void Reset( void);

};

//Tady se musi pouzit parcialni specializace
template< typename ElementType, unsigned dim>  
class ImageSerializer;

// special case for 3D images
template< typename ElementType>  // TADY nevim, jestli jde dedit od obecne templejty ... aby az se zkonstruuje ImageSerializer<uint8, 3> tak se zkontruuje tadle trida, ktera bude mit ty metody Image<typename ElemType, uint8> (predka)
class ImageSerializer< ElementType, 3 >	//provedu parcialni specializaci
  : public ImageSerializerBase< ElementType, 3>
{
public:
  ImageSerializer() {}
  ImageSerializer( M4D::Imaging::AbstractDataSet *dSet)
    : ImageSerializerBase( dSet) {}

  /**
	 * Each special succesor should implement this functions in
	 * its own manner.
	 **/
  void Serialize( M4D::CellBE::iPublicJob *job);
  
  void OnDataPieceReadRequest( DataPieceHeader *header, DataBuffs &bufs);
};

// special case for 2D images
template< typename ElementType>
class ImageSerializer< ElementType, 2 >
  : public ImageSerializerBase< ElementType, 2>
{
public:
  ImageSerializer() {}
  ImageSerializer( M4D::Imaging::AbstractDataSet *dSet)
    : ImageSerializerBase( dSet) {}
  /**
	 * Each special succesor should implement this functions in
	 * its own manner.
	 **/
  void Serialize( M4D::CellBE::iPublicJob *job);
  
  void OnDataPieceReadRequest( DataPieceHeader *header, DataBuffs &bufs);
};

}
}

//include implementation
#include "ImageSerializer.tcc"

#endif

