#ifndef TRANSDEFS_HPP
#define TRANSDEFS_HPP

//////////////////////////////////////////////////
// Contains some common declarations for communicaton
// between client (M4DApplication) & server (Cell)
//////////////////////////////////////////////////

#include "Common.h"
#include <boost/system/error_code.hpp>

namespace M4D
{
namespace CellBE
{

const uint16 SERVER_PORT = 44433;

class NetException
  : public M4D::ErrorHandling::ExceptionBase
{
public :
  NetException( const std::string &s)
    : ExceptionBase( s) {}
};

inline void HandleErrorFlag( const boost::system::error_code &err)
{
  throw NetException( "chyba na siti");
}

}}
#endif
