#ifndef TRANSDEFS_HPP
#define TRANSDEFS_HPP

//////////////////////////////////////////////////
// Contains some common declarations for communicaton
// between client (M4DApplication) & server (Cell)
//////////////////////////////////////////////////

#include "Common.h"
#include "messHeader.h"

namespace M4D
{
namespace CellBE
{

const uint16 SERVER_PORT = 44433;

// messages identifications
enum ReqService {
  Mess_Ping,
	Mess_Job
};

typedef std::basic_ostream<
  uint8, 
  std::char_traits<uint8> > uint8_stream;

//class uint8stream : std::basic_ostream<uint8>
//{
//  uin
//}

}}
#endif
