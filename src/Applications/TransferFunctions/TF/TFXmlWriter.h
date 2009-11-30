 #ifndef TF_XMLWRITER
 #define TF_XMLWRITER

 #include <QtCore/QXmlStreamWriter>
 #include <QtCore/QString>

 #include <TF/TFScheme.h>

 class TFXmlWriter : public QXmlStreamWriter
 {
 public:
	 TFXmlWriter();
     bool write(QIODevice* device, TFScheme** data);

 private:
     void writeScheme(TFScheme** scheme);
     void writeFunction(TFFunction** function);
     void writePoint(TFPoint** point);
 };


 #endif	//TF_XMLWRITER