#ifndef CELLCLIENT_HPP
#define CELLCLIENT_HPP

#include "cellBE/clientJob.h"

namespace M4D
{
namespace CellBE
{

class CellClient
{
public:
  CellClient();

  ClientJob *
    CreateJob(ClientJob::FilterVector &filters, DataSetProperties *props);

  inline void Run( void) { m_io_service.run(); }

private:
  typedef std::map<uint16, std::string> AvailServersMap;
  AvailServersMap m_servers;

  boost::asio::io_service m_io_service;

  void FindNonCommentLine( std::ifstream &f, std::string &line);

  /**
   *	Returns string reference containing address of least loaded available
   *	server able doing specified job
   */
  const std::string & FindAvailableServer( const ClientJob::FilterVector &filters);

};

} // CellBE namespace
} // M4D namespace

#endif