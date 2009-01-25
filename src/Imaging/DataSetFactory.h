#ifndef DATASETFACTORY_H_
#define DATASETFACTORY_H_

#include "iAccessStream.h"
#include "AbstractDataSet.h"

namespace M4D
{
namespace Imaging
{

/**
 * Factory class that creates data sets of all kind. 
 */
class DataSetFactory
{
public:
	/**
	 * Creates data set based on atributes that reads from stream
	 */
	static AbstractDataSet::ADataSetPtr CreateDataSet(iAccessStream &stream);
	
private:	// helpers
	static AbstractDataSet::ADataSetPtr CreateImage(iAccessStream &stream);
};

}
}
#endif /*DATASETFACTORY_H_*/
