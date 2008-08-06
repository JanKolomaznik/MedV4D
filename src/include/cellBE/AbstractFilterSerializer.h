#ifndef ABSTRACT_FILTER_SERIALIZER_H
#define ABSTRACT_FILTER_SERIALIZER_H

#include "filterIDEnums.h"
#include "cellBE/netStream.h"

#include <vector>

namespace M4D
{
namespace CellBE
{

class AbstractFilterSerializer
{
public:
  /**
   *  Identification of particular AbstractFilter sucessor. Each new one has 
   *  return value that is added to enumeration in filterIDEnums.h header.
   */
  virtual FilterID GetID(void) = 0;

  /**
   *  Each final sucessor has to implement this functions to allow
   *  sending all properties of that particular sucessor to server.
   */
  /**
	*  Filter's settings. Used to sending to server.
	*  This is pointer to base abstract settings class.
	*  !!! Each new filter derived from this class
	*  should declare new settings type derived from 
	*  FilterSettingTemplate class (filterProperties.h) 
	*  with template param of type FilterID (FilterIDEnums.h).
	*  This new enum item should be also added to enum with a new
	*  data set class !!!
	*/
  virtual void SerializeProperties( M4D::CellBE::NetStream &s) = 0;
  virtual void DeSerializeProperties( M4D::CellBE::NetStream &s) = 0;  
};

typedef std::vector<AbstractFilterSerializer *> FilterPropsVector;

}
}

#endif

