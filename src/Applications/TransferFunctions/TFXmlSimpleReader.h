#ifndef TF_XMLSIMPLEREADER
#define TF_XMLSIMPLEREADER

#include <QtCore/QXmlStreamReader>
#include <QtCore/QString>

#include <TFSimpleFunction.h>

class TFXmlSimpleReader: public QXmlStreamReader{

public:
	TFXmlSimpleReader();
	~TFXmlSimpleReader();

	TFSimpleFunction read(QIODevice* device, bool &error);

private:
	TFSimpleFunction readFunction(bool &error);
	TFPoint readPoint(bool &error);
};

#endif //TF_XMLSIMPLEREADER