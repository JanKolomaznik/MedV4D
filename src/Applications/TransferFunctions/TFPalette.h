#ifndef TF_WINDOW
#define TF_WINDOW

#include <map>

#include <QtGui/QWidget>
#include <QtGui/QDockWidget>
#include <QtGui/QMainWindow>
#include <QtGui/QFileDialog>
#include <QtGui/QMessageBox>
#include <QtGui/QMenuBar>
#include <QtGui/QKeySequence>
#include <QtGui/QKeyEvent>
#include <QtGui/QVBoxLayout>

#include <TFCreator.h>
#include <TFPaletteButton.h>

#include <TFCommon.h>
#include <TFColor.h>
#include <TFHistogram.h>
#include <TFBasicHolder.h>
#include <TFIndexer.h>

#include "ui_TFPalette.h"

namespace M4D {
namespace GUI {

class TFPalette : public QMainWindow{

    Q_OBJECT

public:

	typedef boost::shared_ptr<TFPalette> Ptr;

	TFPalette(QMainWindow* parent);
    ~TFPalette();

	void setupDefault();

	void setDataStructure(const std::vector<TF::Size>& dataStructure);
	void setHistogram(const TF::Histogram::Ptr histogram);
	void setPreview(const QImage& preview, const int index = -1);

	TF::Size getDomain(const TF::Size dimension);	
	TF::Size getDimension();	

	std::vector<TFBasicHolder*> getEditors();	
	TFFunctionInterface::Const getTransferFunction();

	bool changed();
	Common::TimeStamp lastPaletteChange();

	template<typename BufferIterator>
	bool applyTransferFunction(BufferIterator begin, BufferIterator end){

		if(activeEditor_ == emptyPalette) on_addButton_clicked();
		if(activeEditor_ == emptyPalette || activeEditor_ == noFunctionAvailable) return false;

		return palette_.find(activeEditor_)->second.holder->applyTransferFunction<BufferIterator>(begin, end);
	}

private slots:

    void close_triggered(TF::Size index);

	void on_addButton_clicked();
	void on_removeButton_clicked();

	void change_activeHolder(TF::Size index);

protected:

	void resizeEvent(QResizeEvent*);
	void closeEvent(QCloseEvent *);

private:	

	struct Editor{
		TFBasicHolder* holder;
		TFPaletteButton* button;

		Editor():
			holder(NULL),
			button(NULL){
		}

		Editor(TFBasicHolder* holder, TFPaletteButton* button):
			holder(holder),
			button(button){
		}
	};

	typedef std::map<TF::Size, Editor> HolderMap;
	typedef HolderMap::iterator HolderMapIt;

	static const int noFunctionAvailable = -1;
	static const int emptyPalette = -2;

    Ui::TFPalette* ui_;
	QMainWindow* mainWindow_;
	QVBoxLayout* layout_;

	TF::Histogram::Ptr histogram_;
	std::vector<TF::Size> dataStructure_;

	M4D::Common::TimeStamp lastChange_;
	bool activeChanged_;

	TF::Indexer indexer_;
	int activeEditor_;
	HolderMap palette_;

	TFCreator creator_;

	void addToPalette_(TFBasicHolder* holder);
	void removeFromPalette_(const TF::Size index);

	void activateNext_(HolderMapIt it);

	void changeDomain_(const TF::Size dimension);
};

} // namespace GUI
} // namespace M4D

#endif //TF_WINDOW