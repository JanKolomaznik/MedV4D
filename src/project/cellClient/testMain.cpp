//
// client.cpp kokoda
// ~~~~~~~~~~
//
// Copyright (c) 2003-2008 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#include "Common.h"
#include "clientJob.h"
#include "cellBE/CellClient.h"

using namespace M4D::CellBE;

int main()
{
  try
  {

    ClientJob job(false);

    ThresholdingSetting *s = new ThresholdingSetting();
    s->threshold = 3.14f;

    job.AddFilter( "thresh", s);

    CellClient c;

    c.SendJob(job);

    c.m_io_service.run();
  }
  catch (std::exception& e)
  {
    std::cerr << e.what() << std::endl;
    return -1;
  }

  return 0;
}
