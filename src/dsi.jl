#=
  Diffusion Spectrum Imaging (DSI) reconstruction
=#

using FFTW, Interpolations

export DSI, dsi_rec, dsi_write

"Container for outputs of a DSI reconstruction"
struct DSI
  pdf::MRI
  odf::MRI
  peak::Vector{MRI}
  qa::Vector{MRI}
end


"""
    DSIwork{T}

Pre-allocated workspace for DSI reconstruction computations

- `T::DataType`                : Data type for computations (default: `Float32`)
- `nfft::Int`                  : Size of FFT matrix
- `nvert::Int`                 : Number of ODF vertices (on the half sphere)
- `dqr::T`                     : Spacing of radial sampling for PDF integration
- `iq_ind::Vector{Int}`        : Indices of voxels in q-space sphere
- `H::Array{T, 3}`             : Hanning window [nfft^3]
- `p::Vector{Array{T, 3}}`     : Diffusion propagator [nfft^3]
- `X::Vector{Array{T, 3}}`     : DWI signals in q-space arrangement [nfft^3]
- `x::Vector{Array{Complex{T}, 3}}`   : FFT of q-space signal [nfft^3]
- `xtmp::Vector{Array{Complex{T}, 3}} : FFT of q-space signal [nfft^3]
- `F::FFTW.cFFTWPlan`          : Linear operator for FFT
- `qr2::Vector{T}              : Q-space radii squared for PDF integration
- `iq_sub_interp::Array{T, 3}  : Non-integer q-space indices for interpolation
- `isort::Vector{Vector{Int}}` : Indices of ODF peak vertices (sorted) [`nvert`]
- `o::Vector{Vector{T}}`       : ODF amplitudes [`nvert`]
- `odf_peak::Vector{Vector{T}}`: ODF amplitudes at local peaks [`nvert`]
- `faces::Matrix{Int}`         : ODF faces (on the half sphere) [nvert x 3]
"""
struct DSIwork{T}
  nfft::Int
  nvert::Int
  dqr::T
  iq_ind::Vector{Int}
  H::Array{T, 3}
  p::Vector{Array{T, 3}}
  X::Vector{Array{T, 3}}
  x::Vector{Array{Complex{T}, 3}}
  xtmp::Vector{Array{Complex{T}, 3}}
  F::FFTW.cFFTWPlan
  qr2::Vector{T}
  iq_sub_interp::Array{T, 3}
  isort::Vector{Vector{Int}}
  o::Vector{Vector{T}}
  odf_peak::Vector{Vector{T}}
  faces::Matrix{Int}

  function DSIwork(bval::Vector{Float32}, bvec::Matrix{Float32}, odf_dirs::ODF=sphere_642, hann_width::Int=32, T::DataType=Float32)

    # Find input q-space points
    q = bvec .* sqrt.(bval)

    # Assume that input q-space points are on discrete locations on a grid
    bmin = minimum(bval)
    dq = sqrt(minimum(bval[bval .> bmin]))	# Sample spacing in q-space
    iq = Int32.(round.(q / dq))		# {-5,...,5} for usual 514-point grid

    # Find corresponding indices in FFT matrix
    nfft = maximum(iq) - minimum(iq) + 1
    nfft = 2^Int32(ceil(log2(nfft)))	# Zero-pad to next power-of-two size
#    nfft = Int32(ceil(nfft/2))*2	# Zero-pad to next even size
    iq_shift = Int32(nfft/2)+1		# Shift (0,0,0) to (Nx,Ny,Nz)/2+1
    iq_sub = iq .+ iq_shift
    iq_ind = Vector{Int}(undef, size(iq_sub, 1))
    iq_ind .= mapslices(x -> LinearIndices((1:nfft,1:nfft,1:nfft))[x...],
                        iq_sub; dims=2)

    # 3D Hanning window
    if hann_width == 0
      H = ones(T, nfft, nfft, nfft)
    else
      H = zeros(T, nfft, nfft, nfft)
      H[iq_ind] = (1 .+ cos.(sqrt.(sum(iq.^2; dims=2))*(2*pi/hann_width))) * .5
    end

    # DWI signal in q-space arrangement
    X    = [zeros(T, nfft, nfft, nfft) for tid in 1:Threads.nthreads()]

    # Linear operator for FFT
    F    = plan_fft(ones(T, nfft, nfft, nfft))

    # FFT of q-space signal
    x    = [zeros(Complex{T}, nfft, nfft, nfft) for tid in 1:Threads.nthreads()]
    xtmp = [zeros(Complex{T}, nfft, nfft, nfft) for tid in 1:Threads.nthreads()]

    # Diffusion propagator
    p   = [zeros(T, nfft, nfft, nfft) for tid in 1:Threads.nthreads()]

    # Number of ODF vertices (on the half sphere)
    nvert    = div(size(odf_dirs.vertices, 1), 2)

    # Radial sampling of PDF (for discrete integration to compute ODF)
    qr = T(nfft/2-1) * collect(T, .3:.03:.9)
    dqr = qr[2] - qr[1]
    iq_sub_interp =
      cat(map(x -> x * qr' .+ iq_shift,
              eachslice(odf_dirs.vertices[nvert+1:end, :], dims=1))[:]...,
          dims=3)

    # Indices of sorted ODF peaks
    isort    = [Vector{Int}(undef, nvert) for tid in 1:Threads.nthreads()]

    # ODF amplitudes
    o        = [Vector{T}(undef, nvert) for tid in 1:Threads.nthreads()]

    # ODF amplitudes at local peaks
    odf_peak = [Vector{T}(undef, nvert) for tid in 1:Threads.nthreads()]

    # ODF faces (on the half sphere)
    faces = copy(odf_dirs.faces)
    faces[faces .> nvert] .-= nvert

    new{T}(
      nfft,
      nvert,
      dqr,
      iq_ind,
      H,
      p,
      X,
      x,
      xtmp,
      F,
      qr .^ 2,
      iq_sub_interp,
      isort,
      o,
      odf_peak,
      faces
    )
  end
end


"""
    dsi_rec(dwi::MRI, mask::MRI, odf_dirs::ODF=sphere_642, σ::Float32=Float32(1.25))

Perform diffusion spectrum imaging (DSI) reconstruction of DWIs, and return a
`DSI` structure.

If you use this method, please cite:
Wedeen, V. J., et al. (2005). Mapping complex tissue architecture with diffusion spectrum magnetic resonance imaging. Magnetic resonance in medicine, 54(6), 1377–1386. https://doi.org/10.1002/mrm.20642

# Arguments
- `dwi::MRI`: A series of DWIs, stored in an `MRI` structure with valid `.bvec`
   and `.bval` fields
- `mask::MRI`: A brain mask volume, stored in an `MRI` structure
- `odf_dirs::ODF=sphere_642`: The vertices and faces of the ODF tessellation,
  stored in an `ODF` structure
- `hann_width::Int=32`: Width of Hanning window (in q-space voxels)

# Output
In the `DSI` structure:
- `.pdf`: PDF amplitudes on the half sphere
- `.odf`: ODF amplitudes on the half sphere
- `.peak`: Orientation vectors of the 3 peak ODF amplitudes
- `.qa`: Quantitative anisotropy for each of the 3 peak orientations

"""
function dsi_rec(dwi::MRI, mask::MRI, odf_dirs::ODF=sphere_642, hann_width::Int=32)

  if isempty(dwi.bval)
    error("Missing b-value table from input DWI structure")
  end

  if isempty(dwi.bvec)
    error("Missing gradient table from input DWI structure")
  end

  npeak = 3;

  W = DSIwork(dwi.bval, dwi.bvec, odf_dirs, hann_width)

  nshift = div(W.nfft,2)

  pdf  = MRI(mask, length(W.iq_ind), Float32)
  odf  = MRI(mask, W.nvert, Float32)
  peak = Vector{MRI}(undef, npeak)
  qa   = Vector{MRI}(undef, npeak)

  for ipeak in 1:npeak
    peak[ipeak] = MRI(mask, 3, Float32)
    qa[ipeak]   = MRI(mask, 1, Float32)
  end

  Threads.@threads for iz in 1:size(dwi.vol, 3)
    for iy in 1:size(dwi.vol, 2)
      for ix in 1:size(dwi.vol, 1)
        mask.vol[ix, iy, iz] == 0 && continue

        tid = Threads.threadid()

        # Arrange DWI signals onto q-space grid
        @views W.X[tid][W.iq_ind] .= dwi.vol[ix, iy, iz, :]

        maximum(W.X[tid]) == 0 && continue

        W.X[tid] .= max.(W.X[tid], 0)

        # Multiply q-space signal with Hanning window
        W.X[tid] .= W.X[tid] .* W.H

        # FFT from q-space signal to diffusion propagator
        # Equivalent to:
        # W.x[tid] .= fftshift(W.F * fftshift(W.X[tid]))
        # but without the memory allocations
        circshift!(W.x[tid], W.X[tid], (nshift, nshift, nshift))
        mul!(W.xtmp[tid], W.F, W.x[tid])
        circshift!(W.x[tid], W.xtmp[tid], (nshift, nshift, nshift))

        # Normalize the real part to a PDF
        # (choosing not to threshold negatives here)
        W.p[tid] .= real.(W.x[tid])
        W.p[tid] .= W.p[tid] ./ sum(W.p[tid])

        @views pdf.vol[ix, iy, iz, :] .= W.p[tid][W.iq_ind]

        # Interpolate PDF values at multiple radii along each direction
        itp = interpolate(W.p[tid], BSpline(Linear()))

        # Integrate PDF along radial direction to compute ODF
        fill!(W.o[tid], 0)
        for ivert in eachindex(W.o[tid])
          for irad in eachindex(W.qr2)
            W.o[tid][ivert] += itp(W.iq_sub_interp[1,irad,ivert],
                                   W.iq_sub_interp[2,irad,ivert],
                                   W.iq_sub_interp[3,irad,ivert]) * W.qr2[irad]
          end

          W.o[tid][ivert] *= W.dqr
        end

        odfmin = minimum(W.o[tid])

        odf.vol[ix, iy, iz, :] .= W.o[tid]

        nvalid = find_peaks!(W)

        n = min(nvalid, npeak)

        for ipeak in 1:n
          peak[ipeak].vol[ix, iy, iz, :] =
            odf_dirs.vertices[W.isort[tid][ipeak], :]

          qa[ipeak].vol[ix, iy, iz] =
            odf.vol[ix, iy, iz, W.isort[tid][ipeak]] - odfmin
        end
      end
    end
  end

  odfmax = maximum(mean(odf.vol, dims=4))

  for ipeak in 1:npeak
    qa[ipeak].vol /= odfmax
  end

  return DSI(pdf, odf, peak, qa)
end


"""
    dsi_write(dsi::DSI, basename::String)

Write the volumes from a `DSI` structure that was created by `dsi_rec()`
to files whose names start with the specified base name.
"""
function dsi_write(dsi::DSI, basename::String)

  for var in fieldnames(DSI)
    vartype = fieldtype(DSI, var)

    if vartype == MRI
      fname = basename * "_" * string(var) * ".nii.gz"
      mri_write(getfield(dsi, var), fname)
    elseif vartype == Vector{MRI}
      for ivol = 1:length(getfield(dsi, var))
        fname = basename * "_" * string(var) * string(ivol) * ".nii.gz"
        mri_write(getfield(dsi, var)[ivol], fname)
      end
    end
  end
end


