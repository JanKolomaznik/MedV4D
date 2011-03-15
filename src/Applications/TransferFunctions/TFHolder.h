#ifndef TF_ABSTRACTHOLDER
#define TF_ABSTRACTHOLDER

#include "common/Types.h"
#include "Imaging/Histogram.h"
#include "GUI/utils/TransferFunctionBuffer.h"

#include "ui_TFHolder.h"

#include <QtGui/QMainWindow>
#include <QtGui/QDockWidget>
#include <QtGui/QFileDialog>
#include <QtGui/QMessageBox>
#include <QtGui/QPainter>
#include <QtGui/QMouseEvent>
#include <QtGui/QWheelEvent>
#include <QtGui/QPaintEvent>

#include <QtCore/QString>

#include <TFTypes.h>
#include <TFXmlWriter.h>
#include <TFXmlReader.h>
#include <TFPaletteButton.h>

#include <TFAbstractFunction.h>
#include <TFAbstractModifier.h>
#include <TFAbstractPainter.h>
#include <TFWorkCopy.h>

namespace M4D {
namespace GUI {

namespace Apply 
{
	template<typename ElementIterator>
	bool applyTransferFunction(
		ElementIterator begin,
		ElementIterator end,
		TFAbstractFunction::Ptr function_){

		tfAbort("unsupported buffer type");
		return false;
	}

	template<>
	inline bool applyTransferFunction<TransferFunctionBuffer1D::Iterator>(
		TransferFunctionBuffer1D::Iterator begin,
		TransferFunctionBuffer1D::Iterator end,
		TFAbstractFunction::Ptr function_
		){

		TFSize index = 0;
		for(TransferFunctionBuffer1D::Iterator it = begin; it!=end; ++it)
		{
			tfAssert(index < function_->getDomain());
			TFColor color = function_->getMappedRGBfColor(index);
			
			*it = TransferFunctionBuffer1D::ValueType(
				color.component1, color.component2, color.component3, color.alpha);

			++index;
		}

		return true;
	}

}

class TFHolder : public QWidget{

	Q_OBJECT
	
	friend class TFHolderFactory;

public:

	typedef boost::shared_ptr<TFHolder> Ptr;

	enum Type{
		TFHolderGrayscale,
		TFHolderGrayscaleAlpha,
		TFHolderRGB,
		TFHolderRGBa,
		TFHolderHSV,
		TFHolderHSVa,
		TFHolderPolygonRGBa,
		TFHolderUnknown
	};

	TFHolder(QMainWindow* mainWindow,
		TFAbstractFunction::Ptr function,
		TFAbstractModifier::Ptr modifier,
		TFAbstractPainter::Ptr painter,
		TFHolder::Type);

	~TFHolder();

	void save();
	void activate();
	void deactivate();

	void setup(const TFSize index);
	void setHistogram(TFHistogramPtr histogram);
	//void setDomain(const TFSize domain);

	bool connectToTFPalette(QObject* tfPalette);	//	tfPalette has to be TFPalette instance
	bool createPaletteButton(QWidget* parent);
	void createDockWidget(QWidget* parent);

	TFSize getIndex();
	TFHolder::Type getType() const;

	TFPaletteButton* getButton() const;
	QDockWidget* getDockWidget() const;

	M4D::Common::TimeStamp getLastChangeTime();
	
	template<typename ElementIterator>
	bool applyTransferFunction(
		ElementIterator begin,
		ElementIterator end){

		modifier_->getWorkCopy()->updateFunction(function_);

		return M4D::GUI::Apply::applyTransferFunction< ElementIterator >( begin, end, function_ );
	}

signals:

	void Close(const TFSize index);
	void Activate(const TFSize index);

protected slots:

	void on_closeButton_clicked();
	void on_saveButton_clicked();
	void on_activateButton_clicked();
	void refresh_view();

private:

	TFHolder::Type type_;
	Ui::TFHolder* ui_;
	QMainWindow* holder_;
	QDockWidget* dockHolder_;
	QDockWidget* dockTools_;
	
	QString qTitle_;

	TFAbstractFunction::Ptr function_;
	TFAbstractModifier::Ptr modifier_;
	TFAbstractPainter::Ptr painter_;
	TFPaletteButton* button_;

	M4D::Common::TimeStamp lastChange_;

	TFSize index_;

	const TFPoint<TFSize, TFSize> painterLeftTopMargin_;
	const TFPoint<TFSize, TFSize> painterRightBottomMargin_;

	bool setup_;
	bool active_;

	void paintEvent(QPaintEvent*);
	void resizeEvent(QResizeEvent*);
	void mousePressEvent(QMouseEvent *e);
	void mouseReleaseEvent(QMouseEvent *e);
	void mouseMoveEvent(QMouseEvent *e);
	void wheelEvent(QWheelEvent *e);

	bool load_(QFile &file);
	void save_(QFile &file);

	void resizePainter_();

	TFPaintingPoint correctCoords_(const TFPaintingPoint &point);
	TFPaintingPoint correctCoords_(int x, int y);

};

template<>
inline std::string convert<TFHolder::Type, std::string>(const TFHolder::Type &holderType){

	switch(holderType){
		case TFHolder::TFHolderGrayscale:
		{
			return "Grayscale";
		}
		case TFHolder::TFHolderGrayscaleAlpha:
		{
			return "Grayscale-alpha";
		}
		case TFHolder::TFHolderRGB:
		{
			return "RGB";
		}
		case TFHolder::TFHolderRGBa:
		{
			return "RGBa";
		}
		case TFHolder::TFHolderHSV:
		{
			return "HSV";
		}
		case TFHolder::TFHolderHSVa:
		{
			return "HSVa";
		}
		case TFHolder::TFHolderPolygonRGBa:
		{
			return "Polygon RGBa";
		}
	}
	return "Unknown";
}

template<>
inline TFHolder::Type convert<std::string, TFHolder::Type>(const std::string &holderType){

	if(holderType == "Grayscale"){
		return TFHolder::TFHolderGrayscale;
	}
	if(holderType == "Grayscale-alpha"){
		return TFHolder::TFHolderGrayscaleAlpha;
	}
	if(holderType == "RGB"){
		return TFHolder::TFHolderRGB;
	}
	if(holderType == "RGBa"){
		return TFHolder::TFHolderRGBa;
	}
	if(holderType == "HSV"){
		return TFHolder::TFHolderHSV;
	}
	if(holderType == "HSVa"){
		return TFHolder::TFHolderHSVa;
	}
	if(holderType == "Polygon RGBa"){
		return TFHolder::TFHolderPolygonRGBa;
	}
	return TFHolder::TFHolderUnknown;
}

} // namespace GUI
} // namespace M4D

#endif //TF_ABSTRACTHOLDER
