#ifndef EIGENVALUES_FILTER_POLICIES_H_
#define EIGENVALUES_FILTER_POLICIES_H_

#include <algorithm>
#include <itkVector.h>

#include "MedV4D/Common/Vector.h"

#include <vector>

namespace M4D
{
  namespace GUI
  {
    namespace Viewer
    {

      template<typename PixelType = unsigned short, typename EigenvalueType = float, size_t Dimension = 3>
      class MethodPolicy
      {
      public:
        typedef itk::Vector<EigenvalueType, Dimension> EigenvaluesVectorType;
        typedef EigenvalueType InputValueType;

        virtual PixelType ComputePixelValue(EigenvaluesVectorType eigenvalues) const = 0;

        virtual PixelType GetHessianSigma() const = 0;

      protected:
        static const PixelType RANGE_NORMALIZATION_CONSTANT = 1000;
      };

      template<typename PixelType = unsigned short, typename EigenvalueType = float, size_t Dimension = 3>
      class FranghiVesselness : public MethodPolicy < PixelType, EigenvalueType, Dimension >
      {
      public:
        typedef itk::Vector<EigenvalueType, Dimension> EigenvaluesVectorType;

        FranghiVesselness(PixelType hessianSigma, EigenvalueType alpha, EigenvalueType beta, EigenvalueType gamma)
          : hessianSigma(hessianSigma), alpha(alpha), beta(beta), gamma(gamma)
        {
        }

        FranghiVesselness(const std::vector<EigenvalueType>& constants)
          : hessianSigma(constants[0]), alpha(constants[1]), beta(constants[2]), gamma(constants[3])
        {
        }

        FranghiVesselness(const FranghiVesselness& other)
          : hessianSigma((PixelType)other.hessianSigma), alpha(other.alpha), beta(other.beta), gamma(other.gamma)
        {
        }

        virtual PixelType ComputePixelValue(EigenvaluesVectorType eigenvalues) const override
        {
          this->SortEigenValuesAbsoluteValue(eigenvalues);

          EigenvalueType R_A = abs(eigenvalues[1]) / abs(eigenvalues[2]);
          EigenvalueType R_B = abs(eigenvalues[0]) / abs(eigenvalues[1] * eigenvalues[2]);
          EigenvalueType S = sqrt(eigenvalues[0] * eigenvalues[0] + eigenvalues[1] * eigenvalues[1] + eigenvalues[2] * eigenvalues[2]);

          if (eigenvalues[1] < 0 && eigenvalues[2] < 0)
          {
            EigenvalueType retval = (1.0 - ExponencialFormula(R_A, alpha)) * ExponencialFormula(R_B, beta) * (1.0 - ExponencialFormula(S, gamma));

            return retval * MethodPolicy::RANGE_NORMALIZATION_CONSTANT;
          }
          else
          {
            return 0;
          }
        }

        virtual PixelType GetHessianSigma() const override
        {
          return this->hessianSigma;
        }

      private:
        struct AbsoluteValueComparer
        {
          bool operator() (EigenvalueType lesser, EigenvalueType bigger) const
          {
            return abs(lesser) < abs(bigger);
          }
        };

        EigenvalueType ExponencialFormula(EigenvalueType a, EigenvalueType b) const
        {
          return exp(-((a*a) / (2 * b*b)));
        }

        void SortEigenValuesAbsoluteValue(EigenvaluesVectorType& eigenvalues) const
        {
          std::sort(eigenvalues.Begin(), eigenvalues.End(), this->comparer);
        }

        PixelType hessianSigma;

        EigenvalueType alpha;
        EigenvalueType beta;
        EigenvalueType gamma;

        AbsoluteValueComparer comparer;
      };

      template<typename PixelType = unsigned short, typename EigenvalueType = float, size_t Dimension = 3>
      class EigenvaluesLinearCombination : public MethodPolicy < PixelType, EigenvalueType, Dimension >
      {
      public:
        typedef itk::Vector<EigenvalueType, Dimension> EigenvaluesVectorType;

        EigenvaluesLinearCombination(const std::vector<EigenvalueType>& constants)
          : hessianSigma(constants[0]), alpha(constants[1]), beta(constants[2]), gamma(constants[3])
        {
        }

        virtual PixelType ComputePixelValue(EigenvaluesVectorType eigenvalues) const override
        {
          return (eigenvalues[0] * this->alpha + eigenvalues[1] * this->beta + eigenvalues[2] * this->gamma) * MethodPolicy::RANGE_NORMALIZATION_CONSTANT;
        }

        virtual ~EigenvaluesLinearCombination()
        {
        }

        virtual PixelType GetHessianSigma() const override
        {
          return this->hessianSigma;
        }

        PixelType hessianSigma;

        EigenvalueType alpha;
        EigenvalueType beta;
        EigenvalueType gamma;
      };

      template<typename PixelType = unsigned short, typename EigenvalueType = float, size_t Dimension = 3>
      class ParameterPolicy : public MethodPolicy < PixelType, EigenvalueType, Dimension >
      {
      public:
        typedef itk::Vector<EigenvalueType, Dimension> EigenvaluesVectorType;

        ParameterPolicy(const std::vector<EigenvalueType>& constants)
          : hessianSigma(constants[0]), alpha(constants[1]), beta(constants[2]), gamma(constants[3])
        {
        }

        virtual PixelType ComputePixelValue(EigenvaluesVectorType eigenvalues) const override
        {
          return 0;
        }

        virtual ~ParameterPolicy()
        {
        }

        virtual PixelType GetHessianSigma() const override
        {
          return this->hessianSigma;
        }

        PixelType hessianSigma;

        EigenvalueType alpha;
        EigenvalueType beta;
        EigenvalueType gamma;
      };
    }
  }
}

#endif //EIGENVALUES_FILTER_POLICIES_H_
