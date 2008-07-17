#ifndef I_PUBLICJOB_H
#define I_PUBLICJOB_H

#include <vector>
namespace M4D
{
namespace CellBE
{

/**
 *  Interface that is given to Imaging library user as an abstraction of Job.
 *  It has sending and retrival ability in scatter gather manner.
 *  Used to send and read dataSets.
 */

class iPublicJob
{
public:

  // data buffer to send
  //template< class T>
  struct DataBuff {
    void *data;
    size_t len;
  };

  // vector of databuffers to be send in one turn
  //template< class T>
  class DataBuffs : public std::vector< DataBuff >
  {
  };

  // Serialization
  //template<class T>
  virtual void PutDataPiece( const DataBuffs &bufs) = 0;

  //template<class T>
  virtual void PutDataPiece( const DataBuff &buf) = 0;

  // Deserialization
  //template<class T>
  virtual void GetDataPiece( DataBuffs &bufs) = 0;

  //template<class T>
  virtual void GetDataPiece( DataBuff &buf) = 0;

  /**
   *  Returns NetStream pointer. When the stream is created endian of the
   *  arrived data is taken into account as well as endian of the mashine
   *  this code is run onto (host). So the data got from the stream will be 
   *  in right byte order respect to the host mashine.
   *  Remember to call SetBuffer before you start useing the NetStream !!!
   */
  virtual NetStream * GetNetStream( void) = 0;

  /**
   *  Each no longer used netstream should be returned back or it will remain
   *  inaccesable to next usage.
   */
  //virtual void ReturnNetStream( NetStream &ns) = 0;

};

}
}

#endif

