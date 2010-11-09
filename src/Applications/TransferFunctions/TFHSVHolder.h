#ifndef TF_HSV_HOLDER
#define TF_HSV_HOLDER

#include "common/Types.h"

#include <TFAbstractHolder.h>
#include <TFRGBFunction.h>
#include <TFRGBPainter.h>
//#include <TFGrayscaleXmlREADER.h>
//#include <TFGrayscaleXmlWriter.h>

#include <string>
#include <map>
#include <vector>

namespace M4D {
namespace GUI {

class TFHSVHolder: public TFAbstractHolder{

public:
	TFHSVHolder(QWidget* window);
	~TFHSVHolder();

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

	TFRGBFunction function_;
	TFRGBPainter painter_;

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

#endif //TF_HSV_HOLDER