#ifndef FILTERID_HPP
#define FILTERID_HPP

#include <vector>
#include "cellBE/iSerializable.h"
#include "Imaging/filterIDEnums.h"

namespace M4D
{
namespace Imaging
{

///////////////////////////////////////////////////////////////////////

/**
 *  abstract class containing filter setting. New filter has to
 *  implement new class derived from this one
 */

typedef FilterID	uint32;

class AbstractFilterSettings 
  : public M4D::CellBE::iSerializable
{
public:
	virtual FilterID
	GetFilterID() const = 0;
	
protected:

	void 
	SerializeIntoStream( M4D::CellBE::NetStream &stream)
	{
		stream << this->GetFilterID();
		Serialize( stream);
	}

	virtual void 
	Serialize( M4D::CellBE::NetStream &s) = 0;

	virtual void 
	DeSerialize( M4D::CellBE::NetStream &s) = 0;

	AbstractFilterSettings() {}
};

typedef std::vector<AbstractFilterSettings *> FilterVector;

///////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////

/*struct ThresholdingSetting : public FilterSettingTemplate<Thresholding>
{
  float threshold;

  void Serialize( M4D::CellBE::NetStream &s)
  {    
    s << threshold;
  }

  void DeSerialize( M4D::CellBE::NetStream &s)
  {
    s >> threshold;
  }
};*/

}
}
#endif

