#ifndef TF_XMLSIMPLEREADER
#define TF_XMLSIMPLEREADER

#include <QtCore/QXmlStreamReader>
#include <QtCore/QString>
#include <QtGui/QMessageBox>

#include <TFSimpleFunction.h>

namespace M4D {
namespace GUI {

class TFXmlSimpleReader: public QXmlStreamReader{

public:
	TFXmlSimpleReader();
	~TFXmlSimpleReader();

	void read(QIODevice* device, TFSimpleFunction* function, bool &error);
	void readTestData(TFSimpleFunction* function);

private:
	void readFunction(TFSimpleFunction* function, bool &error);
	float readPoint(bool &error);
};

} // namespace GUI
} // namespace M4D

#endif //TF_XMLSIMPLEREADER