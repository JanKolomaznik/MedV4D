// testing main for cell client

#include "Common.h"
#include "cellBE/cellClient.h"
#include "Imaging/ImageFactory.h"
#include "cellBE/remoteFilters/BoneSegmentationRemote.h"

using namespace M4D::CellBE;
using namespace M4D::Imaging;

int main()
{
  try
  {
    CellClient client;

    // prepare dataSet
    Image<uint16, 3>::Ptr inImage = 
      ImageFactory::CreateEmptyImage3DTyped<uint16>(15, 15, 15);

    Image<uint8, 3>::Ptr outImage = 
      ImageFactory::CreateEmptyImage3DTyped<uint8>(15, 15, 15);

    BoneSegmentationRemote< Image<uint8, 3> > remFilter;

    //client.Run();
  }
  catch (std::exception& e)
  {
    std::cout << e.what() << std::endl;
    return -1;
  }

  return 0;
}
