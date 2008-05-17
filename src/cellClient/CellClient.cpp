
// Implements functions for communication with CellBE server

#include <fstream>
#include <map>

#include "Common.h"
#include "CellClient.hpp"

using namespace M4D::CellBE;
using namespace std;

///////////////////////////////////////////////////////////////////////

CellClient::CellClient()
{
  
  ifstream cfgFile("cellServers.cfg");
  if( cfgFile.fail() )
  {
    LOG("Cell Server Config file not found!");
    throw M4D::ErrorHandling::ExceptionBase("Cell Server Config file not found!");
  }

  string line;
  uint16 count = 0;

  try {
    do {
      FindNonCommentLine( cfgFile, line);
      if( line.length() > 0)
      {
        m_servers.insert( AvailServersMap::value_type(count, line));
        count++;
      }
    } while( ! cfgFile.eof() );

  } catch (...) {
  }
}

///////////////////////////////////////////////////////////////////////

#define MAX_LINE_LEN 128
void
CellClient::FindNonCommentLine( ifstream &f, string &line)
{
	char tmp[MAX_LINE_LEN];

	f >> line;

	while( !f.eof() && line.find("//") == 0)
	{
		// throw away the rest of line since it is comment
		f.getline( tmp, MAX_LINE_LEN);
		f >> line;
	}
}

///////////////////////////////////////////////////////////////////////

void
CellClient::SendJob( uint16 serverID, Job &job)
{
  AvailServersMap::iterator i = m_servers.find( serverID);

  if( i == m_servers.end() )
  {
    D_PRINT( "server ID not found!");
    throw ExceptionBase("server ID not found!");
  }

  ServerConnection *conn = new ServerConnection(i->second, m_io_service);

  // send the job
  conn->SendJob( job);
}

///////////////////////////////////////////////////////////////////////