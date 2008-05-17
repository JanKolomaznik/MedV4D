#ifndef MESS_HEADER_H
#define MESS_HEADER_H

#include "Common.h"
#include "serializer.h"
#include <sstream>

namespace M4D
{
namespace CellBE
{

#define MESS_HEADER_LEN 2 * sizeof(uint16)

  class MessageHeader
  {
  public:
    char data[MESS_HEADER_LEN];

    MessageHeader( uint16 messID, uint16 len)
    {
      data[0] = 0;
      data[1] = 0;
      data[3] = 0;
      data[2] = 0;

      std::stringstream s( data);

      Serializer::ToStream( s, messID);
      Serializer::ToStream( s, len);
      s.flush();
    }
    MessageHeader() {}

    void Decode( uint16 &messID, uint16 &messLen)
    {
      std::istringstream s( data);

      Serializer::FromStream( s, messID);
      Serializer::FromStream( s, messLen);
    }

  };

} // CellBE namespace
} // M4D namespace

#endif