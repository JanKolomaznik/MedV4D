
#include "Common.h"
#include "cellBE/FilterSerializerArray.h"

using namespace M4D::Imaging;
using namespace M4D::CellBE;

///////////////////////////////////////////////////////////////////////////////

FilterSerializerArray::FilterSerializerArray()
{
  // here is to be put each new filterSerializer instance ...

  // thresholding
  m_serializerArray[ (uint32) FID_Thresholding] = 
    new FilterSerializer< typename ThresholdingFilter< Image<uint8, 3> > >( 
      NULL, 0 );

  // median
  m_serializerArray[ (uint32) FID_Median] = 
    new FilterSerializer< typename MedianFilter2D< Image<uint8, 2> > >( 
      NULL, 0 );
  
  // ...
}

///////////////////////////////////////////////////////////////////////////////
