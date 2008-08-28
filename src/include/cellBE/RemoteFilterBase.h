/**
 * @ingroup cellbe 
 * @author Vaclav Klecanda 
 * @file RemoteFilterBase.h 
 * @{ 
 **/

#ifndef _REMOTE_FILTER_BASE_H
#define _REMOTE_FILTER_BASE_H

#include "cellBE/cellClient.h"

namespace M4D
{

namespace CellBE
{

/**
 *  Used as base class for every remote filters. Contains CellClient
 *  static object that is used for job creation in derived classes.
 *  All needed CellClients methodes are thread safe so no other
 *  synchronization is needed while the cellClient is static.
 */
class RemoteFilterBase
{
protected:
  // gate to remote computing. Shared instance of cell client.
  static CellClient s_cellClient;

public:
  void Run( void) { s_cellClient.Run(); }
};

}}

#endif

/** @} */

