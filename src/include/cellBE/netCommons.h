#ifndef TRANSDEFS_H
#define TRANSDEFS_H

#include "Common.h"

//////////////////////////////////////////////////
// Contains some common declarations for communicaton
// between client (M4DApplication) & server (Cell)
//////////////////////////////////////////////////

#define SERVER_PORT 44433

#define ENDING_PECESIZE ((uint32)-1)

namespace M4D
{
namespace CellBE
{

class NetException
  : public ExceptionBase
{
public:
  NetException()
    : ExceptionBase("Neco spatne na siti") {}
  NetException( const std::string & what)
    : ExceptionBase(what) {}
};

class DisconnectedException
  : public NetException
{
public:
  DisconnectedException()
    : NetException("Spojeni rozpojeno") {}
};
  
}}
#endif
