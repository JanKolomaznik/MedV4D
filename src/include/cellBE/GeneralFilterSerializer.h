#ifndef GENERAL_FILTER_SERIALIZER_H
#define GENERAL_FILTER_SERIALIZER_H

#include "AbstractFilterSerializer.h"
#include "Imaging/AbstractFilter.h"

namespace M4D
{
namespace CellBE
{

/**
 *  Interface that is given to Imaging library user as an abstraction of Job.
 *  It has sending and retrival ability in scatter gather manner.
 *  Used to send and read dataSets.
 */

class GeneralFilterSerializer
{
public:
  static M4D::Imaging::AbstractPipeFilter *
  DeSerialize( M4D::CellBE::NetStream &s);
  
};



}
}

#endif

