#ifndef FILTERID_HPP
#define FILTERID_HPP

#include <vector>
#include "../cellBE/iSerializable.h"
#include "filterIDEnums.h"

namespace M4D
{
namespace Imaging
{

///////////////////////////////////////////////////////////////////////

/**
 *  abstract class containing filter setting. New filter has to
 *  implement new class derived from this one
 */

class AbstractFilterSetting 
  : public M4D::CellBE::iSerializable
{
protected:
  uint8 filterID;

  void SerializeIntoStream( M4D::CellBE::NetStream &stream)
  {
    stream << filterID;
    Serialize( stream);
  }

  AbstractFilterSetting( FilterID fid) : filterID( fid) {}
};

typedef std::vector<AbstractFilterSetting *> FilterVector;

///////////////////////////////////////////////////////////////////////

template< FilterID fid>
class FilterSettingTemplate : public AbstractFilterSetting
{
protected:
  FilterSettingTemplate() : AbstractFilterSetting( fid) {}
};

///////////////////////////////////////////////////////////////////////

struct ThresholdingSetting : public FilterSettingTemplate<Thresholding>
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
};

}
}
#endif