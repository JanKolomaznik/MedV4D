#ifndef TF_WINDOW
#define TF_WINDOW

#include "MedV4D/Common/IDGenerator.h"

#include <map>

#include <QtWidgets/QWidget>
#include <QtWidgets/QDockWidget>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QFileDialog>
#include <QtWidgets/QMessageBox>
#include <QtWidgets/QMenuBar>
#include <QKeySequence>
#include <QKeyEvent>
#include <QtWidgets/QGridLayout>
#include <QtCore/QTimer>

#include "MedV4D/GUI/TF/Creator.h"
#include "MedV4D/GUI/TF/PaletteButton.h"

#include "MedV4D/GUI/TF/Common.h"
#include "MedV4D/GUI/TF/Color.h"
#include "MedV4D/GUI/TF/Histogram.h"
#include "MedV4D/GUI/TF/Editor.h"

#include "MedV4D/generated/ui_Palette.h"

namespace M4D {
namespace GUI {

class Palette : public QMainWindow
{
    Q_OBJECT;
public:

	typedef boost::shared_ptr<Palette> Ptr;
	typedef std::map<TF::Size, Editor*> Editors;

	Palette(QMainWindow* parent, const std::vector<TF::Size>& dataStructure);
	~Palette();

	void
	setupDefault();

	void
	setDataStructure(const std::vector<TF::Size>& dataStructure);
	bool
	setHistogram(const TF::HistogramInterface::Ptr histogram);

	void
	setPreview(const QImage& preview, const int index = -1);
	QImage
	getPreview(const int index = -1);
	QSize
	getPreviewSize();

	TF::Size
	getDomain(const TF::Size dimension);
	TF::Size
	getDimension();

	Editors
	getEditors();

	TransferFunctionInterface::Const
	getTransferFunction(const int index = -1);

	Common::IDNumber
	getActiveEditorId()const
	{
		return activeEditor_;
	}

	Common::TimeStamp
	lastChange();

	Common::TimeStamp
	lastPaletteChange();

public slots:
	void
	selectTransferFunction( int idx )
	{
		change_activeHolder( idx );
	}

	void
	loadFromFile( QString fileName, bool showGui = true )
	{
		Editor* created = creator_.loadEditorFromFile( fileName );

		if(!created) return;

		addToPalette_(created, showGui);
	}

signals:

	void
	UpdatePreview(M4D::GUI::TF::Size index);

	void
	transferFunctionAdded( int idx );

	void
	transferFunctionModified( int idx );

	void
	changedTransferFunctionSelection( int index );

private slots:

	void
	close_triggered(TF::Size index);

	void
	on_addButton_clicked();

	void
	change_activeHolder(TF::Size index);

	void
	update_previews();

	void
	on_previewsCheck_toggled(bool enable);

	void
	detectChanges();

protected:

	void
	resizeEvent(QResizeEvent*);
	void
	closeEvent(QCloseEvent *);

private:

	struct EditorInstance
	{
		Editor* editor;
		PaletteButton* button;
		M4D::Common::TimeStamp change;
		M4D::Common::TimeStamp previewUpdate;

		M4D::Common::TimeStamp lastDetectedChange;

		EditorInstance(): editor(NULL),	button(NULL)
		{ }

		EditorInstance(Editor* editor, PaletteButton* button): editor(editor), button(button)
		{ }

		void
		updatePreview()
		{
			previewUpdate = M4D::Common::TimeStamp();
		}
	};

	typedef std::map<TF::Size, EditorInstance*> HolderMap;
	typedef HolderMap::iterator HolderMapIt;

	static const int noFunctionAvailable = -1;
	static const int emptyPalette = -2;

	Ui::Palette* ui_;
	QMainWindow* mainWindow_;
	QGridLayout* layout_;
	TF::Size colModulator_;

	TF::HistogramInterface::Ptr histogram_;
	std::vector<TF::Size> dataStructure_;

	M4D::Common::TimeStamp lastPaletteChange_;
	M4D::Common::TimeStamp lastChange_;
	bool activeChanged_;

	M4D::Common::IDGenerator idGenerator_;
	int activeEditor_;
	HolderMap palette_;

	Creator creator_;

	QTimer previewUpdater_;
	bool previewEnabled_;

	QTimer mChangeDetectionTimer;

	void
	addToPalette_(Editor* editor, bool visible = true );

	void
	removeFromPalette_(const TF::Size index);

	void
	reformLayout_(bool forceReform = false);

	void
	activateNext_(HolderMapIt it);

	void
	changeDomain_(const TF::Size dimension);
};

} // namespace GUI
} // namespace M4D

#endif //TF_WINDOW
