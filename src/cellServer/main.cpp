
#include "Common.h"
#include "server.h"

using namespace M4D::CellBE;

int main( void)
{
  try
  {
    boost::asio::io_service io_service;

    Server server(io_service);
    io_service.run();
  }
  catch (std::exception& e)
  {
    std::cerr << e.what() << std::endl;
    return -1;
  }

  return 0;
}