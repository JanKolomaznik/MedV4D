#ifndef TF_GRAYSCALE_XMLREADER
#define TF_GRAYSCALE_XMLREADER

#include <QtCore/QXmlStreamReader>
#include <QtCore/QString>
#include <QtGui/QMessageBox>

#include <TFAbstractFunction.h>

namespace M4D {
namespace GUI {

class TFXmlReader: public QXmlStreamReader{

public:
	TFXmlReader();
	~TFXmlReader();
/*
	void read(QIODevice* device, TFAbstractFunction<1>::Ptr function, bool &error);
	void readTestData(TFAbstractFunction<1>* function);

private:
	void readFunction_(TFAbstractFunction<1>::Ptr function, bool &error);
	void readPoint_(TF::Color* point, bool &error);*/
};

} // namespace GUI
} // namespace M4D

#endif //TF_GRAYSCALE_XMLREADER