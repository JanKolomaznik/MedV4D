
#ifndef TF_XMLREADER

#include <QtCore/QXmlStreamReader>
#include <QtCore/QString>
#include <QtCore/QFile>

#include <TF/TFScheme.h>

#include <fstream>

using namespace std;

class TFXmlReader: public QXmlStreamReader{

public:
	TFXmlReader();
	~TFXmlReader();

	bool read(QIODevice* device, TFScheme** storage);

private:
	bool readScheme(TFScheme** scheme);
	bool readFunction(TFFunction** function);
	bool readPoint(TFPoint** point);
};

#endif //TF_XMLREADER