
#include "cellBE/GeneralFilterSerializer.h"

using namespace M4D::Imaging;
using namespace M4D::CellBE;

///////////////////////////////////////////////////////////////////////////////

FilterSerializerArray::FilterSerializerArray()
{
  // here is to be put each new filterSerializer instance ...
  m_serializerArray[ (uint32) FID_Thresholding] = 
    new FilterSerializer< typename ThresholdingFilter< Image<uint8, 3> > >( 
      NULL, 0 );
  
  // ...
}

///////////////////////////////////////////////////////////////////////////////
