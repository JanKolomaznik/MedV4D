#ifndef FILTER_PROPS_H
#define FILTER_PROPS_H

#include <vector>

namespace M4D
{
namespace Imaging
{

///////////////////////////////////////////////////////////////////////

/**
 *  abstract class containing filter setting. New filter has to
 *  implement new class derived from this one
 */

class AbstractFilterSettings
{
public:

	AbstractFilterSettings() {}
};

typedef std::vector<AbstractFilterSettings *> FilterVector;

}
}
#endif

