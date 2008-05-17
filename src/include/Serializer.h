#ifndef SERIALIZER_H
#define SERIALIZER_H

#include <ostream>

/**
 * Only for integral datatypes
 */
class Serializer
{
  static const uint8 m_mask = 0xFF;
  static int i;
  static char m_charArray[8];
  static uint64 val;
public:
  
  template< typename T >
  static void ToStream( std::ostream &s, const typename T &what)
  {
    uint64 tmp = (uint64)what;
    int i = sizeof(what);
    while( i > 0, i--)
    {
      s << (char)(tmp & m_mask);
      tmp = (uint64) (tmp >> 8);
    }
  }

  template< typename T >
  static void FromStream( std::istream &s, typename T &what)
  {
    uint64 val = 0;
    int i = sizeof(what);

    char c[8];

    // read the whole num
    s.read( c, i);

    // parse it in oposite order
    i--;
    val += (uint8) c[i];
    

    while( i>0, i--)
    {
      val = val << 8;      
      val += (uint8) c[i];
    }

    what = (T)val;
  }
};

//#ifndef SERIALIZER_STATICS
//#define SERIALIZER_STATICS
//  char Serializer::m_charArray[8];
//  int Serializer::i;
//  uint64 Serializer::val;
//#endif

#endif
