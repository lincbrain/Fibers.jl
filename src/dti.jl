#=
  Original Author: Anastasia Yendiki

  Copyright © 2022 The General Hospital Corporation (Boston, MA) "MGH"
 
  Terms and conditions for use, reproduction, distribution and contribution
  are found in the 'FreeSurfer Software License Agreement' contained
  in the file 'LICENSE' found in the FreeSurfer distribution, and here:
 
  https://surfer.nmr.mgh.harvard.edu/fswiki/FreeSurferSoftwareLicense
 
  Reporting: freesurfer@nmr.mgh.harvard.edu
=#

using LinearAlgebra, StaticArrays, Statistics

export DTI, adc_fit, dti_fit, dti_write


"Container for outputs of a DTI fit"
struct DTI
  s0::MRI
  eigval1::MRI
  eigval2::MRI
  eigval3::MRI
  eigvec1::MRI
  eigvec2::MRI
  eigvec3::MRI
  rd::MRI
  md::MRI
  fa::MRI
end


"""
    ADCwork{T}

Pre-allocated workspace for ADC fit computations

- `T::DataType`                : Data type for computations (default: `Float32`)
- `nvol::Int`                  : Number of volumes in DWI series
- `ib0::Vector{Bool}`          : Indicator for b=0 volumes [`nvol`]
- `ipos::Vector{Vector{Bool}}` : Indicator for volumes with positive DWI signal [`nvol`]
- `logs::Vector{Vector{T}}`    : Logarithm of DWI signal [`nvol`]
- `d::Vector{Vector{T}}`       : Linear system solution vector [2]
- `A::Matrix{T}`               : System matrix [nvol x 2]
- `pA::Matrix{T}`              : Pseudo-inverse of system matrix [2 x nvol]
"""
struct ADCwork{T}
  nvol::Int
  ib0::Vector{Bool}
  ipos::Vector{Vector{Bool}}
  logs::Vector{Vector{T}}
  d::Vector{Vector{T}}
  A::Matrix{T}
  pA::Matrix{T}

  function ADCwork(bval::Vector{Float32}, T::DataType=Float32)

    # Number of volumes in DWI series
    nvol = length(bval)

    # Indicator for b=0 volumes
    ib0  = (bval .== minimum(bval))

    # Indicator for volumes with positive DWI signal
    ipos = [Vector{Bool}(undef, nvol) for tid in 1:Threads.nthreads()]

    # Log of DWI signal
    logs = [Vector{T}(undef, nvol) for tid in 1:Threads.nthreads()]

    # Linear system solution vector
    d    = [Vector{T}(undef, 2) for tid in 1:Threads.nthreads()]

    # System matrix
    A = Matrix{T}(undef, nvol, 2)

    A[:,1] = -bval
    A[:,2] = ones(T, nvol)

    # Pseudo-inverse of system matrix
    pA = pinv(A)

    new{T}(
      nvol,
      ib0,
      ipos,
      logs,
      d,
      A,
      pA
    )
  end
end


"""
    DTIwork{T}

Pre-allocated workspace for DTI fit computations

- `T::DataType`                : Data type for computations (default: `Float32`)
- `nvol::Int`                  : Number of volumes in DWI series
- `ib0::Vector{Bool}`          : Indicator for b=0 volumes [`nvol`]
- `ipos::Vector{Vector{Bool}}` : Indicator for volumes with positive DWI signal [`nvol`]
- `logs::Vector{Vector{T}}`    : Logarithm of DWI signal [`nvol`]
- `d::Vector{Vector{T}}`       : Linear system solution vector [7]
- `A::Matrix{T}`               : System matrix [nvol x 7]
- `pA::Matrix{T}`              : Pseudo-inverse of system matrix [7 x nvol]
"""
struct DTIwork{T}
  nvol::Int
  ib0::Vector{Bool}
  ipos::Vector{Vector{Bool}}
  logs::Vector{Vector{T}}
  d::Vector{Vector{T}}
  A::Matrix{T}
  pA::Matrix{T}

  function DTIwork(bval::Vector{Float32},
                   bvec::Matrix{Float32}, T::DataType=Float32)

    # Number of volumes in DWI series
    nvol = length(bval)

    # Indicator for b=0 volumes
    ib0  = (bval .== minimum(bval))

    # Indicator for volumes with positive DWI signal
    ipos = [Vector{Bool}(undef, nvol) for tid in 1:Threads.nthreads()]

    # Log of DWI signal
    logs = [Vector{T}(undef, nvol) for tid in 1:Threads.nthreads()]

    # Linear system solution vector
    d    = [Vector{T}(undef, 7) for tid in 1:Threads.nthreads()]

    # System matrix
    A = Matrix{T}(undef, nvol, 7)

    A[:,1] = bvec[:,1].^2
    A[:,2] = 2*bvec[:,1].*bvec[:,2]
    A[:,3] = 2*bvec[:,1].*bvec[:,3]
    A[:,4] = bvec[:,2].^2
    A[:,5] = 2*bvec[:,2].*bvec[:,3]
    A[:,6] = bvec[:,3].^2

    A[:,1:6] .*= -bval

    A[:,7] = ones(T, nvol)

    # Pseudo-inverse of system matrix
    pA = pinv(A)

    new{T}(
      nvol,
      ib0,
      ipos,
      logs,
      d,
      A,
      pA
    )
  end
end


"""
    adc_fit(dwi::MRI, mask::MRI)

Fit the apparent diffusion coefficient (ADC) to DWIs and return it as an
`MRI` structure.
"""
function adc_fit(dwi::MRI, mask::MRI)

  if isempty(dwi.bval)
    error("Missing b-value table from input DWI structure")
  end

  W = ADCwork(dwi.bval)

  adc = MRI(mask, 1, Float32)
  s0  = MRI(mask, 1, Float32)

  Threads.@threads for iz in 1:size(dwi.vol, 3)
    for iy in 1:size(dwi.vol, 2)
      for ix in 1:size(dwi.vol, 1)
        mask.vol[ix, iy, iz] == 0 && continue

        adc.vol[ix, iy, iz],
        s0.vol[ix, iy, iz] = adc_fit(dwi.vol[ix, iy, iz, :], W)
      end
    end
  end

  return adc, s0
end


"""
    adc_fit(dwi::Vector{T}, W::ADCwork{T}) where T<:AbstractFloat

Fit the ADC for a single voxel
"""
function adc_fit(dwi::Vector{T}, W::ADCwork{T}) where T<:AbstractFloat

  tid = Threads.threadid()

  # Only use positive DWI values to fit the model
  W.ipos[tid] .= dwi .> 0
  npos = sum(W.ipos[tid])

  if npos == W.nvol
    W.logs[tid] .= log.(dwi)
    mul!(W.d[tid], W.pA, W.logs[tid])
  elseif npos > 6 && any(W.ipos[tid][W.ib0])
    mul!(W.d[tid], pinv(W.A[W.ipos[tid], :]), log.(dwi[W.ipos[tid]]))
  else
    return zero(T), zero(T)
  end

  return W.d[tid][1], exp(W.d[tid][2])
end


"""
    dti_fit(dwi::MRI, mask::MRI)

Fit tensors to DWIs and return a `DTI` structure.
"""
function dti_fit(dwi::MRI, mask::MRI)

  if isempty(dwi.bval)
    error("Missing b-value table from input DWI structure")
  end

  if isempty(dwi.bvec)
    error("Missing gradient table from input DWI structure")
  end

  dti_fit_ls(dwi::MRI, mask::MRI)
end


"""
    dti_fit_ls(dwi::MRI, mask::MRI)

Perform least-squares fitting of tensors from DWIs and return a `DTI` structure.

If you use this method, please cite:
Peter Basser et al. (1994). Estimation of the effective self-diffusion tensor from the NMR spin echo. Journal of Magnetic Resonance Series B, 103(3), 247–254. https://doi.org/10.1006/jmrb.1994.1037
"""
function dti_fit_ls(dwi::MRI, mask::MRI)

  W = DTIwork(dwi.bval, dwi.bvec)

  S0    = MRI(mask, 1, Float32)
  Eval1 = MRI(mask, 1, Float32)
  Eval2 = MRI(mask, 1, Float32)
  Eval3 = MRI(mask, 1, Float32)
  Evec1 = MRI(mask, 3, Float32)
  Evec2 = MRI(mask, 3, Float32)
  Evec3 = MRI(mask, 3, Float32)
  RD    = MRI(mask, 1, Float32)
  MD    = MRI(mask, 1, Float32)
  FA    = MRI(mask, 1, Float32)

  Threads.@threads for iz in 1:size(dwi.vol, 3)
    for iy in 1:size(dwi.vol, 2)
      for ix in 1:size(dwi.vol, 1)
        mask.vol[ix, iy, iz] == 0 && continue

        S0.vol[ix, iy, iz],
        Eval1.vol[ix, iy, iz],
        Eval2.vol[ix, iy, iz],
        Eval3.vol[ix, iy, iz],
        Evec1.vol[ix, iy, iz, :],
        Evec2.vol[ix, iy, iz, :],
        Evec3.vol[ix, iy, iz, :],
        RD.vol[ix, iy, iz],
        MD.vol[ix, iy, iz],
        FA.vol[ix, iy, iz] = dti_fit_ls(dwi.vol[ix, iy, iz, :], W)
      end
    end
  end

  return DTI(S0, Eval1, Eval2, Eval3, Evec1, Evec2, Evec3, RD, MD, FA)
end


"""
    dti_fit_ls(dwi::Vector{T}, W::DTIwork{T}) where T<:AbstractFloat

Perform least-squares fitting of tensor for a single voxel
"""
function dti_fit_ls(dwi::Vector{T}, W::DTIwork{T}) where T<:AbstractFloat

  tid = Threads.threadid()

  # Only use positive DWI values to fit the model
  W.ipos[tid] .= dwi .> 0
  npos = sum(W.ipos[tid])

  if npos == W.nvol
    W.logs[tid] .= log.(dwi)
    mul!(W.d[tid], W.pA, W.logs[tid])
  elseif npos > 6 && any(W.ipos[tid][W.ib0])
    mul!(W.d[tid], pinv(W.A[W.ipos[tid], :]), log.(dwi[W.ipos[tid]]))
  else
    return zero(T), zero(T), zero(T), zero(T), 
                    zeros(T, 3), zeros(T, 3), zeros(T, 3),
                    zero(T), zero(T), zero(T)
  end

  s0 = exp(W.d[tid][7])

  D = @SMatrix [ W.d[tid][1] 0           0;
                 W.d[tid][2] W.d[tid][4] 0;
                 W.d[tid][3] W.d[tid][5] W.d[tid][6] ]

  E = eigen(Symmetric(D, :L))

  return s0, E.values[3], E.values[2], E.values[1],
             E.vectors[:, 3], E.vectors[:, 2], E.vectors[:, 1],
             dti_maps(E.values[3], E.values[2], E.values[1])...
end


"""
    dti_maps(eigval1::T, eigval2::T, eigval3::T) where T<:AbstractFloat

Return the radial diffusivity (RD), mean diffusivity (MD), and fractional
anisotropy (FA) given the 3 eigenvalues the diffusion tensor.
"""
function dti_maps(eigval1::T, eigval2::T, eigval3::T) where T<:AbstractFloat

  rd = eigval2 + eigval3
  md = (eigval1 + rd) / 3
  rd /= 2

  fa = sqrt(((eigval1 - md)^2 + (eigval2 - md)^2 + (eigval3 - md)^2) /
            (eigval1^2 + eigval2^2 + eigval3^2) * T(1.5))

  return rd, md, fa
end


"""
    dti_write(dti::DTI, basename::String)

Write the volumes from a `DTI` structure that was created by `dti_fit()`
to files whose names start with the specified base name.
"""
function dti_write(dti::DTI, basename::String)

  for var in fieldnames(DTI)
    mri_write(getfield(dti, var), basename * "_" * string(var) * ".nii.gz")
  end
end


