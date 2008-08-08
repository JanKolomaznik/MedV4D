// testing main for cell client

#include "Common.h"
#include "cellBE/cellClient.h"
#include "Imaging/ImageFactory.h"
#include "Imaging/filters/ThresholdingFilter.h"
#include "Imaging/filters/RemoteFilter.h"

using namespace M4D::CellBE;
using namespace M4D::Imaging;

int main()
{
  try
  {
    CellClient client;

    // prepare pipeline
    ThresholdingFilterMask::Settings<uint16> thresh;
    thresh.bottom = 12;
    thresh.top = 34;
    
    FilterPropsVector filters;
    filters.push_back( &thresh);

    // prepare dataSet
    Image<uint16, 3>::Ptr inImage = 
      ImageFactory::CreateEmptyImage3DTyped<uint16>(15, 15, 15);

    Image<uint8, 3>::Ptr outImage = 
      ImageFactory::CreateEmptyImage3DTyped<uint8>(15, 15, 15);

    ClientJob *job = client.CreateJob( 
      filters
      , (AbstractDataSet *) inImage
      , (AbstractDataSet *) outImage );

    client.Run();
  }
  catch (std::exception& e)
  {
    std::cout << e.what() << std::endl;
    return -1;
  }

  return 0;
}
