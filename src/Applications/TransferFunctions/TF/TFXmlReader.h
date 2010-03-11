
#ifndef TF_XMLREADER
#define TF_XMLREADER

#include <QtCore/QXmlStreamReader>
#include <QtCore/QString>
#include <QtCore/QFile>

#include <TF/TFScheme.h>
#include <TF/TFSchemePainter.h>
#include <TF/TFSchemeTools.h>

#include <fstream>

using namespace std;

class TFXmlReader: public QXmlStreamReader{

public:
	TFXmlReader();
	~TFXmlReader();

	bool read(QIODevice* device, TFScheme** storage);

private:
	bool readScheme(TFScheme** scheme);
	bool readFunction(TFSchemeFunction** function);
	bool readPoint(TFSchemePoint** point);
};

#endif //TF_XMLREADER