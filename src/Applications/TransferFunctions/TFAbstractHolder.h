#ifndef TF_HOLDERINTERFACE
#define TF_HOLDERINTERFACE

#include <QtGui/QDockWidget>
#include <QtGui/QMainWindow>

#include <QtCore/QString>
#include <QtGui/QMessageBox>
#include <QtGui/QFileDialog>

#include <TFCommon.h>
#include <TFHistogram.h>
#include <TFPaletteButton.h>

#include <TFXmlReader.h>
#include <TFXmlWriter.h>

#include <TFAbstractFunction.h>
#include <TFAbstractModifier.h>
#include <TFAbstractPainter.h>
#include <TFWorkCopy.h>

#include <TFPredefined.h>

#include <TFAdaptation.h>

#include <set>

#include <ui_TFHolderUI.h>

namespace M4D {
namespace GUI {	

class TFAbstractHolder: public QWidget{

	Q_OBJECT

public:

	typedef boost::shared_ptr<TFAbstractHolder> Ptr;

	enum Attribute{
		Composition
	};
	typedef std::set<Attribute> Attributes;

	virtual ~TFAbstractHolder(){}

	virtual bool loadData(TFXmlReader::Ptr reader, bool& sideError) = 0;

	virtual void setup(QMainWindow* mainWindow, const int index = -1);
	virtual void setHistogram(TF::Histogram::Ptr histogram) = 0;
	virtual void setDomain(const TF::Size domain) = 0;

	TF::Size getIndex();
	std::string getName();
	virtual Attributes getAttributes();
	QDockWidget* getDockWidget() const;

	virtual void activate();
	virtual void deactivate();

	virtual bool changed() = 0;

	template<typename BufferIterator>
	bool applyTransferFunction(
		BufferIterator begin,
		BufferIterator end){

		return TF::Adaptation::applyTransferFunction<BufferIterator>(begin, end, functionToApply_());
	}

signals:

	void Close(const TF::Size index);
	void Activate(const TF::Size index);

public slots:

	void save();
	virtual void close();

protected slots:

	virtual void refresh_view();

	void activate_clicked();

protected:

	const TF::Point<TF::Size, TF::Size> painterLeftTopMargin_;
	const TF::Point<TF::Size, TF::Size> painterRightBottomMargin_;

	QDockWidget* holderDock_;
	QMainWindow* holderMain_;

	Ui::TFHolderUI* ui_;

	TF::Types::Structure structure_;

	QString fileName_;
	std::string name_;
	TF::Size index_;
	Attributes attributes_;

	bool saved_;
	bool active_;
	
	TFAbstractHolder();

	virtual void paintEvent(QPaintEvent*) = 0;
	virtual void resizeEvent(QResizeEvent*) = 0;

	virtual TFApplyFunctionInterface::Ptr functionToApply_() = 0;
	
	virtual void saveData_(TFXmlWriter::Ptr writer) = 0;
};

} // namespace GUI
} // namespace M4D

#endif //TF_HOLDERINTERFACE
