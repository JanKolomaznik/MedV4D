#ifndef TF_EDITOR
#define TF_EDITOR

#include <QtWidgets/QDockWidget>
#include <QtWidgets/QMainWindow>

#include <QtCore/QString>
#include <QtWidgets/QMessageBox>
#include <QtWidgets/QFileDialog>

#include "MedV4D/GUI/TF/Common.h"
#include "MedV4D/GUI/TF/Histogram.h"

#include "MedV4D/GUI/TF/FunctionInterface.h"
#include "MedV4D/GUI/TF/AbstractModifier.h"
#include "MedV4D/GUI/TF/WorkCopy.h"

#include "MedV4D/GUI/TF/Predefined.h"

#include"MedV4D/Imaging/Histogram.h"

#include <set>

namespace M4D {
namespace GUI {

class Editor: public QMainWindow
{
	Q_OBJECT
public:

	typedef std::shared_ptr<Editor> Ptr;

	enum Attribute{
		Composition
	};
	typedef std::set<Attribute> Attributes;

	Editor(AbstractModifier::Ptr modifier,
		TF::Types::Structure structure,
		Attributes attributes,
		std::string name);

	virtual ~Editor();

	bool
	load(TF::XmlReaderInterface* reader, bool& sideError);

	bool
	loadFunction(TF::XmlReaderInterface* reader);

	bool
	save();

	bool
	saveFunction();

	bool
	close();

	virtual void
	setup(QMainWindow* mainWindow, const int index = -1) = 0;

	void
	setHistogram(TF::HistogramInterface::Ptr histogram);

	void
	setStatistics(M4D::Imaging::Statistics::Ptr aStatistics);

	void
	setDataStructure(const std::vector<TF::Size>& dataStructure);

	TF::Size
	getIndex();

	std::string
	getName();

	Attributes
	getAttributes();

	bool
	hasAttribute(const Attribute attribute);

	TF::Size
	getDimension();

	TransferFunctionInterface::Const
	getFunction();

	QDockWidget*
	getDockWidget() const;

	virtual void
	setActive(const bool active);

	virtual void
	setAvailable(const bool available){}

	Common::TimeStamp
	lastChange();

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

	AbstractModifier::Ptr modifier_;

	virtual void saveSettings_(TF::XmlWriterInterface* writer);

	virtual bool loadSettings_(TF::XmlReaderInterface* reader);
};

} // namespace GUI
} // namespace M4D

#endif //TF_EDITOR
