#ifndef JOB_H
#define JOB_H

namespace M4D
{
namespace CellBE
{

  class Job
  {
    uint16 m_filterID;
    void *m_data;

  public:

#define RESP_HEADER_LEN 2
    struct ResponseHeader {
      char data[RESP_HEADER_LEN];
    };

    enum State {
      Complete,
      Incomplete,
      Failed,
    };
    State state;

    ResponseHeader respHeader;

    std::vector<uint8> response;
    typedef void (*JobCompletitionCallback)(void);

    JobCompletitionCallback onComplete;

    template <typename Archive>
    void serialize(Archive& ar, const unsigned int version)
    {
      ar & m_filterID;
      ar & m_data;
    }

  };

} // CellBE namespace
} // M4D namespace

#endif