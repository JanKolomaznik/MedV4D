/**
 *  @ingroup cellbe
 *  @file FilterSerializerArray.cpp
 *  @author Vaclav Klecanda
 */
#include "Common.h"
#include "cellBE/FilterSerializerArray.h"

using namespace M4D::Imaging;

namespace M4D {
namespace CellBE {

///////////////////////////////////////////////////////////////////////////////

FilterSerializerArray::FilterSerializerArray()
{
  // here is to be put each new filterSerializer instance ...

  // thresholding
  m_serializerArray[ (uint32) FID_Thresholding] = 
    new FilterSerializer< ThresholdingFilter< Image<uint8, 3> > >( 
      NULL, 0 );

  // median
  m_serializerArray[ (uint32) FID_Median] = 
    new FilterSerializer< MedianFilter2D< Image<uint8, 2> > >( 
      NULL, 0 );
  
  // max intensity projection
  m_serializerArray[ (uint32) FID_SimpleProjection] = 
    new FilterSerializer< SimpleProjection< Image<uint8, 3> > >( 
      NULL, 0 );
  
  // ...
}

///////////////////////////////////////////////////////////////////////////////

}
}
