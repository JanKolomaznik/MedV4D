/**
 *  @defgroup cellbe Remote computing
 */
/**
 * @ingroup cellbe 
 * @author Vaclav Klecanda 
 * @file remoteNodesManager.h
 */

/**
 * @addtogroup cellbe
 * @{ 
 *
 * Library used to send some parts of pipeline to remote mashines to be executed
 * and result sent back. This library is not bound to specific architecture. 
 * It was one of primary request to be platform indemendet.
 **/

#ifndef CELLCLIENT_HPP
#define CELLCLIENT_HPP

#include <fstream>
#include <string>

namespace M4D
{
namespace RemoteComputing
{

/**
 *  This class represent the gate to remote computing. Parts of computed pipeline can be send to a server to computed remotely. Then the results are sent back. There can be variety of server and architectures. Primarily this is done for Cell Broadbend Engine (CBE). We have some Play Stations 3 where is CBE with 8 SPE stream processors. But it can be ported to some other supercomputing server like Blade, or some normal servers based on desktop processors.
 *  Main idea is in creation entity called job that represents remote computation. Job is send to server, there is build his instance and stored in job container for later rerun. Then aproprite part of pipeline is created and run. The results are sent back to client. The client lives on ther server until explicit termination message is recieved. Job can be rerun with different settings (through RERUN message, entire data is not resent, instead only changes are sent). Each filter has its own ID through that is identified on the server.
 *  Next item is filter settings vector. It is pair of filterID and filter parameters. Based on filter IDs is created pipeline on server side.
 *  The last item is pointer to dataSet. It is pointer to abstract base class that each dataSet is derived from. Through virtual functions this dataSet can serialize and reSerialize and write its parameters.
 *  Each job on client side is created throuh this class by CreateJob function.
 *  This class also contains container of available servers that can be used for remote computing. It can also load it from config file.
 */
class RemoteNodesManager
{
public:
  RemoteNodesManager();
  
  /**
 *	Returns string reference containing address of least loaded available
 *	server able doing specified job. Here can be implemented some load balancing functionality.
 */
  const std::string & FindAvailableServer(void);

private:
  typedef std::map<uint16, std::string> AvailServersMap;
  AvailServersMap m_servers;

  // loading from file support
  void FindNonCommentLine( std::ifstream &f, std::string &line); 

};

} // CellBE namespace
} // M4D namespace

#endif
/** @} */

