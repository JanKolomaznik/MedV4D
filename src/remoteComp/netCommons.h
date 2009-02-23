/**
 * @ingroup cellbe 
 * @author Vaclav Klecanda 
 * @file netCommons.h 
 * @{ 
 **/

#ifndef TRANSDEFS_H
#define TRANSDEFS_H

#include "Common.h"


/**
 * Contains some common declarations for communicaton
 * between client (M4DApplication) & server (Cell)
 */

/// predefined server port
#define SERVER_PORT 44433

/// tag, that is put to DataPiece header indicating end of dataSet
#define ENDING_PECESIZE ((uint32)-1)

namespace M4D
{
namespace RemoteComputing
{

/**
*  Definition of basic command IDs.
*/
enum eCommand {
CREATE,       // request for job create
EXEC,         // request for job execute
DATASET,      // sending job's dataSet
};

/**
 *  Definition of some other exceptions used in networking
 */
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

/** @} */

