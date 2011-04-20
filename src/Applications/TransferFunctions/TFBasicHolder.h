#ifndef TF_BASICHOLDER
#define TF_BASICHOLDER

#include <QtGui/QDockWidget>
#include <QtGui/QMainWindow>

#include <QtCore/QString>
#include <QtGui/QMessageBox>
#include <QtGui/QFileDialog>
#include <QtGui/QPainter>

#include <TFCommon.h>
#include <TFHistogram.h>

#include <TFFunctionInterface.h>
#include <TFAbstractModifier.h>
#include <TFWorkCopy.h>

#include <TFPredefined.h>

#include <TFAdaptation.h>

#include <set>

#include <ui_TFHolderUI.h>

namespace M4D {
namespace GUI {	

class TFBasicHolder: public QWidget{

	Q_OBJECT

public:

	typedef boost::shared_ptr<TFBasicHolder> Ptr;

	enum Attribute{
		Composition,
		Dimension1
	};
	typedef std::set<Attribute> Attributes;
	
	TFBasicHolder(TFAbstractModifier::Ptr modifier,
		TF::Types::Structure structure,
		Attributes attributes,
		std::string name);

	virtual ~TFBasicHolder();

	virtual bool loadData(TF::XmlReaderInterface* reader, bool& sideError);

	virtual void setup(QMainWindow* mainWindow, const int index = -1);
	virtual void setHistogram(TF::Histogram::Ptr histogram);
	virtual void setDataStructure(const std::vector<TF::Size>& dataStructure);

	TF::Size getIndex();
	std::string getName();
	Attributes getAttributes();
	bool hasAttribute(const Attribute attribute);

	TF::Size getDimension();
	TFFunctionInterface::Const getFunction();

	QDockWidget* getDockWidget() const;

	virtual void setActive(const bool active);
	void setAvailable(const bool available);

	virtual Common::TimeStamp lastChange();

	template<typename BufferIterator>
	bool applyTransferFunction(
		BufferIterator begin,
		BufferIterator end){

			return TF::Adaptation::applyTransferFunction<BufferIterator>(
				begin,
				end,
				TFFunctionInterface::Const(modifier_->getFunction())
			);
	}

signals:

	void Close(const TF::Size index);
	void Activate(const TF::Size index);

public slots:

	void save();
	virtual bool close();

protected slots:

	void activate_clicked();
	void on_nameEdit_editingFinished();

protected:

	QDockWidget* holderDock_;
	QMainWindow* holderMain_;

	Ui::TFHolderUI* ui_;
	QDockWidget* toolsDock_;

	TF::XmlWriterInterface* writer_;

	TF::Types::Structure structure_;

	QString fileName_;
	std::string name_;
	TF::Size index_;
	Attributes attributes_;

	M4D::Common::TimeStamp lastSave_;
	M4D::Common::TimeStamp lastChange_;
	bool active_;

	TFAbstractModifier::Ptr modifier_;

	virtual void resizeEvent(QResizeEvent*);
	
	virtual void saveData_(TF::XmlWriterInterface* writer);

	virtual void saveSettings_(TF::XmlWriterInterface* writer);
	virtual bool loadSettings_(TF::XmlReaderInterface* reader);
};

} // namespace GUI
} // namespace M4D

#endif //TF_BASICHOLDER
