#ifndef TF_ABSTRACTHOLDER
#define TF_ABSTRACTHOLDER

#include "common/Types.h"
#include "Imaging/Histogram.h"

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

#include <TFCommon.h>
#include <TFXmlWriter.h>
#include <TFXmlReader.h>
#include <TFPaletteButton.h>

#include <TFAbstractFunction.h>
#include <TFAbstractModifier.h>
#include <TFAbstractPainter.h>
#include <TFWorkCopy.h>
#include <TFAdaptation.h>

namespace M4D {
namespace GUI {

class TFHolder : public QWidget{

	Q_OBJECT
	
	friend class TFCreator;

public:

	typedef boost::shared_ptr<TFHolder> Ptr;

	TFHolder(QMainWindow* mainWindow,
		TFAbstractPainter::Ptr painter,
		TFAbstractModifier::Ptr modifier,
		std::string title);

	~TFHolder();

	void save();
	void activate();
	void deactivate();

	void setup(const TF::Size index);
	void setHistogram(TF::Histogram::Ptr histogram);
	void setDomain(const TF::Size domain);

	bool connectToTFPalette(QObject* tfPalette);	//	tfPalette has to be TFPalette instance
	bool createPaletteButton(QWidget* parent);
	void createDockWidget(QWidget* parent);

	TF::Size getIndex();

	TFPaletteButton* getButton() const;
	QDockWidget* getDockWidget() const;

	M4D::Common::TimeStamp getLastChangeTime();
	
	template<typename ElementIterator>
	bool applyTransferFunction(
		ElementIterator begin,
		ElementIterator end){

		return TF::Adaptation::applyTransferFunction< ElementIterator >( begin, end,
			modifier_->getWorkCopy()->getFunction());
	}

signals:

	void Close(const TF::Size index);
	void Activate(const TF::Size index);

protected slots:

	void on_closeButton_clicked();
	void on_saveButton_clicked();
	void on_activateButton_clicked();
	void refresh_view();

private:

	TFHolder(QMainWindow* mainWindow);

	Ui::TFHolder* ui_;
	QMainWindow* holder_;
	QDockWidget* dockHolder_;
	QDockWidget* dockTools_;
	
	std::string title_;

	TFAbstractModifier::Ptr modifier_;
	TFAbstractPainter::Ptr painter_;
	TFPaletteButton* button_;

	M4D::Common::TimeStamp lastChange_;

	TF::Size index_;

	const TF::Point<TF::Size, TF::Size> painterLeftTopMargin_;
	const TF::Point<TF::Size, TF::Size> painterRightBottomMargin_;

	bool blank_;
	bool active_;

	void paintEvent(QPaintEvent*);
	void resizeEvent(QResizeEvent*);
	void mousePressEvent(QMouseEvent *e);
	void mouseReleaseEvent(QMouseEvent *e);
	void mouseMoveEvent(QMouseEvent *e);
	void wheelEvent(QWheelEvent *e);

	bool load(QFile &file);
	void save_(QFile &file);

	void resizePainter_();

	TF::PaintingPoint correctCoords_(const TF::PaintingPoint &point);
	TF::PaintingPoint correctCoords_(int x, int y);

};

} // namespace GUI
} // namespace M4D

#endif //TF_ABSTRACTHOLDER
