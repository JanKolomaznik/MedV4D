#ifndef TF_BASICHOLDER
#define TF_BASICHOLDER

#include "common/Types.h"

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

#include <TFPredefined.h>

#include <TFHolderInterface.h>

#include <ui_TFBasicHolder.h>

namespace M4D {
namespace GUI {

class TFBasicHolder : public QWidget, public TFHolderInterface{

	Q_OBJECT
	
	friend class TFCreator;

public:

	typedef boost::shared_ptr<TFBasicHolder> Ptr;

	TFBasicHolder(QMainWindow* mainWindow,
		TFAbstractPainter<1>::Ptr painter,
		TFAbstractModifier<1>::Ptr modifier,
		TF::Types::Structure structure);

	~TFBasicHolder();

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

	bool changed();

signals:

	void Close(const TF::Size index);
	void Activate(const TF::Size index);

protected slots:

	void on_closeButton_clicked();
	void on_saveButton_clicked();
	void on_activateButton_clicked();
	void refresh_view();

private:

	TFBasicHolder(QMainWindow* mainWindow);

	Ui::TFBasicHolder* ui_;
	QMainWindow* holder_;
	QDockWidget* dockHolder_;
	QDockWidget* dockTools_;
	
	std::string title_;
	TF::Types::Structure structure_;

	TFAbstractModifier<1>::Ptr modifier_;
	TFAbstractPainter<1>::Ptr painter_;
	TFPaletteButton* button_;

	TF::Size index_;

	const TF::Point<TF::Size, TF::Size> painterLeftTopMargin_;
	const TF::Point<TF::Size, TF::Size> painterRightBottomMargin_;

	bool blank_;
	bool active_;

	TFApplyFunctionInterface::Ptr functionToApply_();

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

#endif //TF_BASICHOLDER
