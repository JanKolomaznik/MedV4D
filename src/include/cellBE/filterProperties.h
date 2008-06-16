#ifndef FILTERID_HPP
#define FILTERID_HPP

#include "cellBE/iSerializable.h"

namespace M4D
{
namespace CellBE
{

//class NetStream;  // forward

/** 
 *	filter identification defines. Here are to be added new ones when
 *	new filter is written
 */
enum FilterID {
  Thresholding,
};

///////////////////////////////////////////////////////////////////////

/**
 *  abstract class containing filter setting. New filter has to
 *  implement new class derived from this one
 */
class FilterSetting : public iSerializable
{
protected:
  uint8 filterID;
public:
  FilterSetting( FilterID fid) : filterID(fid) {}
};

///////////////////////////////////////////////////////////////////////

struct ThresholdingSetting : public FilterSetting
{
  float threshold;

  ThresholdingSetting()
    : FilterSetting(Thresholding)
    , threshold(0)
  {}  

  void Serialize( NetStream &s)
  {
    s << filterID;
    s << threshold;
  }

  void DeSerialize( NetStream &s)
  {
    s >> threshold;
  }

};

}
}
#endif