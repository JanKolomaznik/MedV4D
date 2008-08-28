/**
 * @ingroup cellbe 
 * @author Vaclav Klecanda 
 * @file iSerializable.h 
 * @{ 
 **/

#ifndef ISERIALIZABLE_H
#define ISERIALIZABLE_H

#include "cellBE/netStream.h"

namespace M4D
{
namespace CellBE
{

/**
 *  Serializable interface. 2 main function prototypes is within:
 *  Serialize put data into stream and DeSerialize retrieve and construct the data
 */
class iSerializable
{
public:
  virtual void Serialize( NetStream &) = 0;
  virtual void DeSerialize( NetStream &s) = 0;
};

}
}

#endif


/** @} */

