#ifndef IMAGE_SERIALIZER_H
#define IMAGE_SERIALIZER_H

#include "../AbstractDataSetSerializer.h"

namespace M4D
{
namespace CellBE
{

class ImageSerializer
  : public AbstractDataSetSerializer
{
public:

  ImageSerializer( M4D::Imaging::AbstractDataSet *dSet)
    : AbstractDataSetSerializer( dSet)
  {
  }

  ImageSerializer() {}

  /**
   *  Each final sucessor has to implement this functions to allow
   *  sending all properties of that particular sucessor to server.
   */
  void SerializeProperties( M4D::CellBE::NetStream &s);
  M4D::Imaging::AbstractDataSet *
    DeSerializeProperties( M4D::CellBE::NetStream &s);


	/**
	 * Each special succesor should implement this functions in
	 * its own manner.
	 **/
  void Serialize( M4D::CellBE::iPublicJob *job);
  
  void OnDataPieceReadRequest( DataPieceHeader *header, DataBuffs &bufs);
  void OnDataSetEndRead( void);
};

}
}

#endif

