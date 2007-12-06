#ifndef PER_CONN_THREAD_H
#define PER_CONN_THREAD_H

namespace dataStorage {

	class PerConnectionThread {
		M4D_SOCKET client;

		public void Recieved( byte* data);
		public void Send( byte* data);
	}

}

#endif