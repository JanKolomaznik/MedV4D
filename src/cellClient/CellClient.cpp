
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
  ServerConnection *conn;
  uint16 count = 0;

  try {
    do {
      FindNonCommentLine( cfgFile, line);
      if( line.length() > 0)
      {
        conn = new ServerConnection( line);
        m_servers.insert( AvailServersMap::value_type(count, conn));
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
    throw BaseException("server ID not found!");
  }

  ServerConnection *conn = *i;
  if( conn == NULL)
  {
    D_PRINT( "server address is NULL");
    throw BaseException("server address is NULL");
  }

  if( ! conn->IsConnected())
    conn->Connect();

  // send the job
  conn->SendJob( job);
}

///////////////////////////////////////////////////////////////////////