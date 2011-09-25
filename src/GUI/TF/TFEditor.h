#ifndef TF_EDITOR
#define TF_EDITOR

#include <QtGui/QDockWidget>
#include <QtGui/QMainWindow>

#include <QtCore/QString>
#include <QtGui/QMessageBox>
#include <QtGui/QFileDialog>

#include "GUI/TF/TFCommon.h"
#include "GUI/TF/TFHistogram.h"

#include "GUI/TF/TFFunctionInterface.h"
#include "GUI/TF/TFAbstractModifier.h"
#include "GUI/TF/TFWorkCopy.h"

#include "GUI/TF/TFPredefined.h"

#include <set>

namespace M4D {
namespace GUI {	

class TFEditor: public QMainWindow{

	Q_OBJECT

public:

	typedef boost::shared_ptr<TFEditor> Ptr;

	enum Attribute{
		Composition
	};
	typedef std::set<Attribute> Attributes;
	
	TFEditor(TFAbstractModifier::Ptr modifier,
		TF::Types::Structure structure,
		Attributes attributes,
		std::string name);

	virtual ~TFEditor();

	bool load(TF::XmlReaderInterface* reader, bool& sideError);
	bool loadFunction(TF::XmlReaderInterface* reader);
	bool save();
	bool saveFunction();
	bool close();

	virtual void setup(QMainWindow* mainWindow, const int index = -1) = 0;
	void setHistogram(TF::HistogramInterface::Ptr histogram);
	void setDataStructure(const std::vector<TF::Size>& dataStructure);

	TF::Size getIndex();
	std::string getName();
	Attributes getAttributes();
	bool hasAttribute(const Attribute attribute);

	TF::Size getDimension();
	TFFunctionInterface::Const getFunction();

	QDockWidget* getDockWidget() const;

	virtual void setActive(const bool active);
	virtual void setAvailable(const bool available){}

	Common::TimeStamp lastChange();

signals:

	void Close(const TF::Size index);
	void Activate(const TF::Size index);

protected:

	QDockWidget* editorDock_;
	QDockWidget* toolsDock_;

	TF::XmlWriterInterface* writer_;

	TF::Types::Structure structure_;
	QString fileName_;
	QString fileNameFunction_;
	std::string name_;
	TF::Size index_;
	Attributes attributes_;

	M4D::Common::TimeStamp lastSave_;
	M4D::Common::TimeStamp lastChange_;
	bool active_;

	TFAbstractModifier::Ptr modifier_;
	
	virtual void saveSettings_(TF::XmlWriterInterface* writer);

	virtual bool loadSettings_(TF::XmlReaderInterface* reader);
};

} // namespace GUI
} // namespace M4D

#endif //TF_EDITOR
