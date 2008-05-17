#ifndef JOB_H
#define JOB_H

#include <vector>
#include "messHeader.h"

namespace M4D
{
namespace CellBE
{

#define MESSLEN 32

  class Job
  {
    
    void *m_data;

  public:
    uint16 m_filterID;

    float32 m_f1;
    std::string m_str;


    Job( uint16 filterID) : m_filterID(filterID) {}
    Job( void) {}

    enum State {
      Complete,
      Incomplete,
      Failed,
    };
    State state;

    std::vector<uint8> response;
    typedef void (*JobCompletitionCallback)(void);

    MessageHeader messageHeader;
    std::string sendedMessage;

    JobCompletitionCallback onComplete;

    template <typename Archive>
    void serialize(Archive& ar, const unsigned int version)
    {
      ar & m_filterID;
      ar & m_f1;
      ar & m_str;
      //ar & m_data;
    }

    static const int messLen = 32;

  };

} // CellBE namespace
} // M4D namespace

#endif