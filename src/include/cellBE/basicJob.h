#ifndef JOBBASE_H
#define JOBBASE_H

#include <vector>
#include "filterProperties.h"
#include "messageHeaders.h"
#include "dataSetProperties.h"

#include "basicSocket.h"

namespace M4D
{
namespace CellBE
{


class BasicJob
  : public BasicSocket
{

protected:
  BasicJob(boost::asio::io_service &service) : BasicSocket(service) {}

  enum Action {
    CREATE,
    REEXEC,
    DESTROY,
    PING
  };

  typedef std::vector<FilterSetting *> FilterVector;
  FilterVector m_filters;

  PrimaryJobHeader primHeader;
  SecondaryJobHeader secHeader;

  // data buffer to send
  template< class T>
  struct DataBuff {
    T *data;
    size_t len;
  };

public:
  // vector of databuffers to be send in one turn
  template< class T>
  class DataBuffs : public std::vector< DataBuff<T> >
  {
  };

  // Serialization
  template<class T>
  void PutDataPiece( DataBuffs<T> &bufs) = 0;

  template<class T>
  void PutDataPiece( DataBuff<T> &buf) = 0;

  // Deserialization
  template<class T>
  void GetDataPiece( DataBuffs<T> &bufs) = 0;

  template<class T>
  void GetDataPiece( DataBuff<T> &buf) = 0;
  

};

} // CellBE namespace
} // M4D namespace

#endif