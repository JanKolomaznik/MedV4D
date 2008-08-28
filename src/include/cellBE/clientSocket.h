/**
 * @ingroup cellbe 
 * @author Vaclav Klecanda 
 * @file clientSocket.h 
 * @{ 
 **/

#ifndef CLIENT_SOCKET_H
#define CLIENT_SOCKET_H

#include "cellBE/basicJob.h"

namespace M4D
{
namespace CellBE
{

/**
 *  Advances BasicSocket functionality for connecting ability
 */
class ClientSocket
  : public BasicJob
{
public:

  //void SendJob( ClientJob *job);
  //void Resend( ClientJob *job);
  //void QuitJob( ClientJob *job);

  /*typedef void (*OnXCallback)(void);

  void Send( void *what, size_t size, OnXCallback onOk, OnXCallback onError)*/

protected:
  ClientSocket( const std::string &address, boost::asio::io_service &service);

  std::string m_address;

  //////////////////////////////////////////////
  //void SendData( ClientJob *j);
  //void ReadResponseHeader( ClientJob *job);
  //////////////////////////////////////////////

  void Connect( boost::asio::io_service &service);
  // send callbacks
  //void EndSendJobHeader( const boost::system::error_code& e, ClientJob *j);
  //void EndSendData( const boost::system::error_code& e, ClientJob *j);

  //void OnJobResponseHeaderRead( const boost::system::error_code& e,
  //  const size_t bytesRead, ClientJob *j);

  //void OnJobResponseBodyRead( const boost::system::error_code& e,
  //  const size_t bytesRead, ClientJob *j);

};

} // CellBE namespace
} // M4D namespace

#endif
/** @} */

