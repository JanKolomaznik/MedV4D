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

    void SendJob( uint16 serverID, Job &job);

    inline AvailServersMap &GetAvailableServers(void) { return m_servers; }

    boost::asio::io_service m_io_service;

  private:
    AvailServersMap m_servers;    

    void FindNonCommentLine( std::ifstream &f, std::string &line);

  };

} // CellBE namespace
} // M4D namespace

#endif