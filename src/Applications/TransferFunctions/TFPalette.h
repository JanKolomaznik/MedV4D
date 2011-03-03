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

#include "Imaging/Histogram.h"

#include <TFHolderFactory.h>
#include <TFPaletteButton.h>
#include <ui_TFPalette.h>

namespace M4D {
namespace GUI {

class TFPalette : public QMainWindow{

    Q_OBJECT

	typedef std::map<TFSize, TFHolder*> HolderMap;
	typedef HolderMap::iterator HolderMapIt;
	typedef M4D::Imaging::Histogram32::Ptr HistogramPtr;

public:

	TFPalette(QMainWindow* parent, const TFSize& domain);
    ~TFPalette();

	M4D::Common::TimeStamp getTimeStamp();

	template<typename ElementIterator>
	bool applyTransferFunction(
		ElementIterator begin,
		ElementIterator end){

		if(activeHolder_ < 0) return false;
		return palette_.find(activeHolder_)->second->applyTransferFunction<ElementIterator>(begin, end);
	}

	void setupDefault();

	void setHistogram(HistogramPtr histogram);

protected slots:

    void close_triggered(TFSize index);
	void newTF_triggered(TFHolder::Type tfType);

    void on_actionLoad_triggered();

	void change_activeHolder(TFSize index);

protected:

	void resizeEvent(QResizeEvent*);

private:	

	class Indexer{

		typedef std::vector<TFSize> Indexes;
		typedef Indexes::iterator IndexesIt;

	public:

		Indexer();
		~Indexer();

		TFSize getIndex();
		void releaseIndex(TFSize index);

	private:

		TFSize nextIndex_;
		Indexes released_;
	};

    Ui::TFPalette* ui_;
	QMainWindow* mainWindow_;
	QVBoxLayout* layout_;

	HistogramPtr histogram_;
	TFSize domain_;

	Indexer indexer_;
	int activeHolder_;
	HolderMap palette_;
	
	TFActions tfActions_;	

	bool connectTFActions_();

	void addToPalette_(TFHolder* holder);
	void removeFromPalette_(TFSize index);
};

} // namespace GUI
} // namespace M4D

#endif //TF_WINDOW