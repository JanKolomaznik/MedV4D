#ifndef CELLCLIENT_HPP
#define CELLCLIENT_HPP

#include "serverConn.h"

namespace M4D
{
namespace CellBE
{

  class CellClient
  {
  public:
    typedef std::map<uint16, std::string> AvailServersMap;

    CellClient();

    void SendJob( ClientJob &job);

    boost::asio::io_service m_io_service;

  private:
    AvailServersMap m_servers;    

    void FindNonCommentLine( std::ifstream &f, std::string &line);

	/**
	 *	Returns string reference containing address of least loaded available
	 *	server able doing specified job
	 */
	const std::string & FindAvailableServer( const ClientJob &job);

  };

} // CellBE namespace
} // M4D namespace

#endif