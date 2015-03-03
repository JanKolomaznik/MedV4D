#ifndef FILTERING_DIALOG_BASE_H_
#define FILTERING_DIALOG_BASE_H_

#include <qdialog.h>

 template<typename EigenvalueType = float>
class FilteringDialogBase : public QDialog
{
public:
  virtual std::vector<EigenvalueType> GetValues() = 0;
};

#endif // FILTERING_DIALOG_BASE_H_