#ifndef PER_CONN_THREAD_H
#define PER_CONN_THREAD_H

namespace dataStorage {

	class PerConnectionThread
	{
	public:
		PerConnectionThread( void);
		~PerConnectionThread( void);
		
		CrsPlatfrmServerSocket client;

	public:
		void Recieved( int8* data);
		void Send( int8* data);
	};

}

#endif