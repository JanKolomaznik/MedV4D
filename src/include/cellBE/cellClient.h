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

  ClientJob *CreateJob(
    M4D::Imaging::FilterVector &filters, 
    M4D::Imaging::AbstractDataSet *dataSet);

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
  const std::string & FindAvailableServer( const M4D::Imaging::FilterVector &filters);

};

} // CellBE namespace
} // M4D namespace

#endif