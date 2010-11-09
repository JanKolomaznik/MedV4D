#ifndef TF_HSVA_HOLDER
#define TF_HSVA_HOLDER

#include "common/Types.h"

#include <TFAbstractHolder.h>
#include <TFRGBaFunction.h>
#include <TFRGBaPainter.h>
//#include <TFGrayscaleXmlREADER.h>
//#include <TFGrayscaleXmlWriter.h>

#include <string>
#include <map>
#include <vector>

namespace M4D {
namespace GUI {

class TFHSVaHolder: public TFAbstractHolder{

public:
	TFHSVaHolder(QWidget* window);
	~TFHSVaHolder();

	void setUp(QWidget *parent, const QRect rect);

protected:
	void save_(QFile &file);
	bool load_(QFile &file);

	void updateFunction_();
	void updatePainter_(const QRect& rect);

	TFAbstractFunction* getFunction_();

private:

	enum ConversionType{
		CONVERT_HSV_TO_RGB,
		CONVERT_RGB_TO_HSV
	};

	TFRGBaFunction function_;
	TFRGBaPainter painter_;

	void convert_(const TFFunctionMapPtr sourceComponent1,
				  const TFFunctionMapPtr sourceComponent2,
				  const TFFunctionMapPtr sourceComponent3,
				  TFFunctionMapPtr outcomeComponent1,
				  TFFunctionMapPtr outcomeComponent2,
				  TFFunctionMapPtr outcomeComponent3,
				  ConversionType type);
};

} // namespace GUI
} // namespace M4D

#endif //TF_HSVA_HOLDER