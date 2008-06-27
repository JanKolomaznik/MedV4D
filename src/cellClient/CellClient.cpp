
// Implements functions for communication with CellBE server

#include <fstream>
#include <map>

#include "Common.h"

#include "cellBE/CellClient.h"

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

ClientJob *
CellClient::CreateJob( ClientJob::FilterVector &filters,
                       DataSetProperties *props)
{
  ClientJob *newJob = new ClientJob(
    filters,
    props,
    FindAvailableServer( filters),
    m_io_service);
}

///////////////////////////////////////////////////////////////////////

const string &
CellClient::FindAvailableServer( const Job::FilterVector &filters)
{
	// find available server with least load (load balancing)
	return m_servers[0];
}

///////////////////////////////////////////////////////////////////////