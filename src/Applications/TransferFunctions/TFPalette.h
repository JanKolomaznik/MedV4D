#ifndef TF_WINDOW
#define TF_WINDOW

#include "common/IDGenerator.h"

#include <map>

#include <QtGui/QWidget>
#include <QtGui/QDockWidget>
#include <QtGui/QMainWindow>
#include <QtGui/QFileDialog>
#include <QtGui/QMessageBox>
#include <QtGui/QMenuBar>
#include <QtGui/QKeySequence>
#include <QtGui/QKeyEvent>
#include <QtGui/QGridLayout>
#include <QtCore/QTimer>

#include <TFCreator.h>
#include <TFPaletteButton.h>

#include <TFCommon.h>
#include <TFColor.h>
#include <TFHistogram.h>
#include <TFEditor.h>

#include "ui_TFPalette.h"

namespace M4D {
namespace GUI {

class TFPalette : public QMainWindow{

    Q_OBJECT

public:

	typedef boost::shared_ptr<TFPalette> Ptr;
	typedef std::map<TF::Size, TFEditor*> Editors;

	TFPalette(QMainWindow* parent, const std::vector<TF::Size>& dataStructure);
    ~TFPalette();

	void setupDefault();

	void setDataStructure(const std::vector<TF::Size>& dataStructure);
	bool setHistogram(const TF::HistogramInterface::Ptr histogram);

	void setPreview(const QImage& preview, const int index = -1);
	QImage getPreview(const int index = -1);
	QSize getPreviewSize();

	TF::Size getDomain(const TF::Size dimension);	
	TF::Size getDimension();	

	Editors getEditors();	
	TFFunctionInterface::Const getTransferFunction(const int index = -1);

	Common::TimeStamp lastChange();
	Common::TimeStamp lastPaletteChange();

signals:

	void UpdatePreview(M4D::GUI::TF::Size index);

private slots:

    void close_triggered(TF::Size index);

	void on_addButton_clicked();

	void change_activeHolder(TF::Size index);

	void update_previews();
	void on_previewsCheck_toggled(bool enable);

protected:

	void resizeEvent(QResizeEvent*);
	void closeEvent(QCloseEvent *);

private:	

	struct Editor{
		TFEditor* editor;
		TFPaletteButton* button;
		M4D::Common::TimeStamp change;
		M4D::Common::TimeStamp previewUpdate;

		Editor():
			editor(NULL),
			button(NULL){
		}

		Editor(TFEditor* editor, TFPaletteButton* button):
			editor(editor),
			button(button){
		}

		void updatePreview(){

			previewUpdate = M4D::Common::TimeStamp();
		}
	};

	typedef std::map<TF::Size, Editor*> HolderMap;
	typedef HolderMap::iterator HolderMapIt;

	static const int noFunctionAvailable = -1;
	static const int emptyPalette = -2;

    Ui::TFPalette* ui_;
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

	TFCreator creator_;

	QTimer previewUpdater_;
	bool previewEnabled_;

	void addToPalette_(TFEditor* editor);
	void removeFromPalette_(const TF::Size index);

	void reformLayout_(bool forceReform = false);

	void activateNext_(HolderMapIt it);

	void changeDomain_(const TF::Size dimension);
};

} // namespace GUI
} // namespace M4D

#endif //TF_WINDOW