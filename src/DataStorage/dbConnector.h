#ifndef DB_CONNECTOR_H
#define DB_CONNECTOR_H


#include <string>

namespace DB {

	/**
	 * Manages access to database
	 */
	class DBConnector {

		bool ConnectToDB( void);
		bool SendQuery( string query);
	}
}

#endif