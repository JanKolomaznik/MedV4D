/**
 * @ingroup cellbe 
 * @author Vaclav Klecanda 
 * @file iPublicJob.h 
 * @{ 
 **/

#ifndef I_PUBLICJOB_H
#define I_PUBLICJOB_H

#include <vector>
namespace M4D
{
namespace CellBE
{

struct DataBuff 
{
  DataBuff( void *dat, size_t lenght)
    : data( dat)
    , len(lenght)
  {}

  DataBuff() {}

  DataBuff( const DataBuff &b)  // copy ctor
    : data( b.data)
    , len( b.len)
  {}

  void *data;
  size_t len;
};

class DataBuffs : public std::vector< DataBuff >
{
};

/**
 *  Interface that is given to Imaging library user as an abstraction of Job.
 *  It has sending and retrival ability in scatter gather manner.
 *  Used to send and read dataSets.
 */

class iPublicJob
{
public:

  // Serialization support for DataSetSerializators
  virtual void PutDataPiece( const DataBuffs &bufs) = 0;
  virtual void PutDataPiece( const DataBuff &buf) = 0;

  // Deserialization is made throug DataSetSerializator directly
  
  /**
   *  Returns NetStream pointer. When the stream is created endian of the
   *  arrived data is taken into account as well as endian of the mashine
   *  this code is run onto (host). So the data got from the stream will be 
   *  in right byte order respect to the host mashine.
   *  Remember to call SetBuffer before you start useing the NetStream !!!
   */
  virtual NetStream * GetNetStream( void) = 0;

};

}
}

#endif


/** @} */

