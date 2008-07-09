// testing main for cell client

#include "Common.h"
#include "cellBE/cellClient.h"
#include "Imaging/ImageFactory.h"

using namespace M4D::CellBE;
using namespace M4D::Imaging;

int main()
{
  try
  {
    CellClient client;

    // prepare filters (pipeline)
    FilterVector filters;
    ThresholdingSetting *s = new ThresholdingSetting();
    s->threshold = 3.14f;
    filters.push_back( s);

    // prepare dataSet
    Image<uint16, 3>::Ptr image = ImageFactory::CreateEmptyImage3DTyped<uint16>(15, 15, 15);

    ClientJob *job = client.CreateJob( filters, (AbstractDataSet *) image );

    client.Run();
  }
  catch (std::exception& e)
  {
    std::cerr << e.what() << std::endl;
    return -1;
  }

  return 0;
}
