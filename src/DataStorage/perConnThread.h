#ifndef PER_CONN_THREAD_H
#define PER_CONN_THREAD_H

namespace dataStorage {

	class PerConnectionThread
	{
	private:
		CrsPlatfrmClientSocket clientSocket;
		void OnRecievedCommand( int8 *data, size_t len);
		void OnRecieveDataRequest( int8 *data, size_t len);
		//SendResult( int8 *data, size_t len);

	public:
		PerConnectionThread();
		~PerConnectionThread( void);		

		void Start( void *threadData);	// thread function

		inline
		CrsPlatfrmClientSocket* GetClientSocket( void) { return &clientSocket; }
	};

}

#endif