//
// server.cpp
// ~~~~~~~~~~
//
// Copyright (c) 2003-2008 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#include "cellBE/netCommons.h"
#include "server.h"

using namespace M4D::CellBE;

struct pok {
  uint32 p;
  uint8 j;
  uint8 k;
};

int main( void)
{
  try
  {
    boost::asio::io_service io_service;

    pok p;
    int i = sizeof(p);

    Server server(io_service, SERVER_PORT);
    io_service.run();
  }
  catch (std::exception& e)
  {
    std::cerr << e.what() << std::endl;
  }

  return 0;
}