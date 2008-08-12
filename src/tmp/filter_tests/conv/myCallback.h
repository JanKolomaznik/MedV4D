class myCallback : public vtkCommand
{
public:
  vtkImageViewer *IV;
//  vtkImageIdealLowPass *Filter;
  vtkImageConvolve *Filter;

  static myCallback *New() { return new myCallback; }

  void Execute(vtkObject *caller,
               unsigned long event,
               void *vtkNotUsed(callData))
    {
      if (this->IV->GetInput() == NULL) { return; }

      // Reset

      if (event == vtkCommand::ResetWindowLevelEvent)
        {
        this->IV->GetInput()->UpdateInformation();
        this->IV->GetInput()->SetUpdateExtent(this->IV->GetInput()->GetWholeExtent());
        this->IV->GetInput()->Update();

//        this->Filter->SetXCutOff(0);
//        this->Filter->SetYCutOff(0);
  	    double k[9] =
  	    { 0,  0,  0,
	      0,  1,  0,
		  0,  0,  0
	    };
	    this->Filter->SetKernel3x3(k);

		this->IV->Render();
        return;
        }

      // Start

      if (event == vtkCommand::StartWindowLevelEvent)
        {
          return;
        }

      // Adjust the window level here

      vtkInteractorStyleImage *isi = static_cast<vtkInteractorStyleImage *>(caller);

      int *size = this->IV->GetRenderWindow()->GetSize();

      // Compute normalized delta

      double dx = 4*(isi->GetWindowLevelCurrentPosition()[0] - isi->GetWindowLevelStartPosition()[0]) / (double)size[0];
      double dy = 4*(isi->GetWindowLevelStartPosition()[1] - isi->GetWindowLevelCurrentPosition()[1]) / (double)size[1];

      // Scale by current values

//      this->Filter->SetXCutOff(dx);
//      this->Filter->SetYCutOff(dy);

	  double h = dy/9 - dx/4,
		     d = dy/9,
			 c = dx + dy/9 + 1-dx-dy;

	  double k[9] =
	  { d,h,d,
	    h,c,h,
		d,h,d
	  };
/*
      dx*0 -1/4 0  + dy*1/9 1/9 1/9 + (1-dx-dy)*0 0 0
		-1/4 1 -1/4     1/9 1/9 1/9             0 1 0
		 0 -1/4 0       1/9 1/9 1/9             0 0 0
*/


	  this->Filter->SetKernel3x3(k);

      this->IV->Render();
    }
};

