#ifndef TRANSDEFS_HPP
#define TRANSDEFS_HPP

//////////////////////////////////////////////////
// Contains some common declarations for communicaton
// between client (M4DApplication) & server (Cell)
//////////////////////////////////////////////////

namespace M4D
{
namespace CellBE
{

const uint16 SERVER_PORT = 44433;

// messages identifications
enum ReqService {
  Ping,
	Threshold		    // client wants some data
};

}}
#endif
