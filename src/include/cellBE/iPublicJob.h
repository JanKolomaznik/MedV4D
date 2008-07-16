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

};

}
}

#endif

