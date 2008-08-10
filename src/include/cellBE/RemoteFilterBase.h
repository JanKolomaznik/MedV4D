#ifndef _REMOTE_FILTER_BASE_H
#define _REMOTE_FILTER_BASE_H

#include "cellBE/cellClient.h"

namespace M4D
{

namespace CellBE
{

class RemoteFilterBase
{
protected:
  // gate to remote computing. Shared instance of cell client.
  static CellClient s_cellClient;

};

}}

#endif
