// testing main for cell client

#include "Common.h"
#include "cellBE/cellClient.h"

using namespace M4D::CellBE;

int main()
{
  try
  {
    CellClient client;

    ClientJob::FilterVector filters;

    ThresholdingSetting *s = new ThresholdingSetting();
    s->threshold = 3.14f;

    filters.push_back( s);

    Image3DProperties props;
    props.x = props.y = props.z = 233;

    ClientJob *job = client.CreateJob( filters, &props);

    client.Run();
  }
  catch (std::exception& e)
  {
    std::cerr << e.what() << std::endl;
    return -1;
  }

  return 0;
}
