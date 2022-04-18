#=
  Original Author: Anastasia Yendiki

  Copyright © 2022 The General Hospital Corporation (Boston, MA) "MGH"
 
  Terms and conditions for use, reproduction, distribution and contribution
  are found in the 'FreeSurfer Software License Agreement' contained
  in the file 'LICENSE' found in the FreeSurfer distribution, and here:
 
  https://surfer.nmr.mgh.harvard.edu/fswiki/FreeSurferSoftwareLicense

  Reporting: freesurfer@nmr.mgh.harvard.edu
=#

using LinearAlgebra, Statistics, DelimitedFiles

export RUMBASD, rumba_rec, rumba_peaks, rumba_write

"Container for outputs of a RUMBA-SD fit"
struct RUMBASD
  fodf::MRI
  fgm::MRI
  fcsf::MRI
  peak::Vector{MRI}
  gfa::MRI
  var::MRI
  snr_mean::Float32
  snr_std::Float32
end


"""
    RUMBAwork{T}

Pre-allocated workspace for RUMBA-SD reconstruction computations

- `T::DataType`         : Data type for computations (default: `Float32`)
- `nmask::Int`          : Number of voxels in brain mask
- `ndir::Int`           : Number of unique diffusion-encoding directions
- `ncomp::Int`          : Number of components in fODF signal decomposition
- `nvox::NTuple{3,Int}` : Dimensions of image volume (or nothing if TV not used)
- `dodf_mat::Matrix{T}` : Work matrices in diffusion ODF space
- `dodf_sig_mat::Matrix{T}`
- `Iratio::Matrix{T}`
- `fodf_mat::Matrix{T}` : Work matrices in fiber ODF space
- `rl_mat::Matrix{T}`
- `tv_mat::Matrix{T}`
- `tv_vol::Array{T,3}`  : Work volumes in image space
- `Gx_vol::Array{T,3}`
- `Gy_vol::Array{T,3}`
- `Gz_vol::Array{T,3}`
- `Div_vol::Array{T,3}`
- `λ::Array{T,3}`
- `σ2_vec::Matrix{T}`   : Noise variance by voxel
"""
struct RUMBAwork{T}
  nmask::Int
  ndir::Int
  ncomp::Int
  nvox::NTuple{3,Int}
  dodf_mat::Matrix{T}
  dodf_sig_mat::Matrix{T}
  Iratio::Matrix{T}
  fodf_mat::Matrix{T}
  rl_mat::Matrix{T}
  tv_mat::Matrix{T}
  tv_vol::Array{T,3}
  Gx_vol::Array{T,3}
  Gy_vol::Array{T,3}
  Gz_vol::Array{T,3}
  Div_vol::Array{T,3}
  λ::Array{T,3}
  σ2_vec::Matrix{T}

  function RUMBAwork(nmask::Int, ndir::Int, ncomp::Int,
                     nvox::Union{NTuple{3,Int}, Nothing}=nothing,
                     T::DataType=Float32)

    # Work matrices in diffusion ODF space
    dodf_mat     = Matrix{T}(undef, ndir, nmask)
    dodf_sig_mat = Matrix{T}(undef, ndir, nmask)
    Iratio       = Matrix{T}(undef, ndir, nmask)

    # Work matrices in fiber ODF space
    fodf_mat     = Matrix{T}(undef, ncomp, nmask)
    rl_mat       = Matrix{T}(undef, ncomp, nmask)
    tv_mat       = Matrix{T}(undef, ncomp, nmask)

    # Work volumes in image space
    isnothing(nvox) && (nvox = (0,0,0))
    tv_vol       = Array{T,3}(undef, nvox)
    Gx_vol       = Array{T,3}(undef, nvox)
    Gy_vol       = Array{T,3}(undef, nvox)
    Gz_vol       = Array{T,3}(undef, nvox)
    Div_vol      = Array{T,3}(undef, nvox)
    λ            = Array{T,3}(undef, nvox)

    # Noise variance by voxel
    σ2_vec       = Matrix{T}(undef, 1, nmask)

    new{T}(
      nmask::Int,
      ndir::Int,
      ncomp::Int,
      nvox::NTuple{3,Int},
      dodf_mat::Matrix{T},
      dodf_sig_mat::Matrix{T},
      Iratio::Matrix{T},
      fodf_mat::Matrix{T},
      rl_mat::Matrix{T},
      tv_mat::Matrix{T},
      tv_vol::Array{T,3},
      Gx_vol::Array{T,3},
      Gy_vol::Array{T,3},
      Gz_vol::Array{T,3},
      Div_vol::Array{T,3},
      λ::Array{T,3},
      σ2_vec::Matrix{T}
    )
  end
end


"""
    tensor_model(φ::Number, θ::Number, λ::Vector, b::Vector, g::Matrix, s0::Number)

    Compute the expected diffusion-weighted signal in a voxel, assuming that
    diffusion can be modeled by a tensor with orientation angles `φ`, `θ` and
    eigenvalues `λ` and that the signal was acquired with b-values `b` and 
    gradient vectors `g`, and that the non-diffusion-weighted signal is `s0`.
"""
function tensor_model(φ::Number, θ::Number, λ::Vector, b::Vector, g::Matrix, s0::Number)

  if length(λ) != 3
    error("Length of diffusivity vector " * string(λ) * " must be 3")
  end

  R = ang2rot(φ, θ)
  D = R * Diagonal(λ) * R'

  S = s0 .* exp.(-b .* diag(g*D*g'))

  return S
end


"""
    besseli_ratio(nu::Integer, z::Float32)

Evaluate the ratio of the modified Bessel functions of the first kind,
of orders `nu` and `nu`-1, at the points in `z`.

Instead of direct evaluation, i.e., besseli(`nu`,`z`) / besseli(`nu`-1,`z`),
compute with Perron's continued fraction equation, which is an order of
magnitude faster.

Perron's continued fraction is substantially superior than Gauss' continued
fraction when z >> nu, and only moderately inferior otherwise
(Walter Gautschi and Josef Slavik, Math. Comp., 32(143):865-875, 1978).
"""
function besseli_ratio(nu::Integer, z::T) where T <: AbstractFloat

  return z / ((2*nu + z) - 
               ((2*nu+1)*z / 
               (2*z + (2*nu+1) - 
               ((2*nu+3)*z /
               ((2*nu+2) + 2*z - ((2*nu+5) * z / ((2*nu+3) + 2*z)))))))
end


"""
    Gradient operator
"""
function sd_grad!(Gx_vol::Array{T, 3}, Gy_vol::Array{T, 3}, Gz_vol::Array{T, 3}, fODF_vol::Array{T, 3}) where T <: AbstractFloat

  @views Gx_vol .= fODF_vol[[2:end; end], :, :] .- fODF_vol
  @views Gy_vol .= fODF_vol[:, [2:end; end], :] .- fODF_vol
  @views Gz_vol .= fODF_vol[:, :, [2:end; end]] .- fODF_vol
end


"""
    Divergence operator
"""
function sd_div!(Div_vol::Array{T, 3}, Gx_vol::Array{T, 3}, Gy_vol::Array{T, 3}, Gz_vol::Array{T, 3}) where T <: AbstractFloat

  @views Div_vol[2:end-1,:,:]  .=  Gx_vol[2:end-1,:,:] .- Gx_vol[1:end-2, :, :]
  @views Div_vol[1,:,:]        .=  Gx_vol[1,:,:]	# boundaries
  @views Div_vol[end,:,:]      .= -Gx_vol[end-1,:,:]

  @views Div_vol[:,2:end-1,:] .+=  Gy_vol[:,2:end-1,:] .- Gy_vol[:, 1:end-2, :]
  @views Div_vol[:,1,:]       .+=  Gy_vol[:,1,:]	# boundaries
  @views Div_vol[:,end,:]     .+= -Gy_vol[:,end-1,:]

  @views Div_vol[:,:,2:end-1] .+=  Gz_vol[:,:,2:end-1] .- Gz_vol[:, :, 1:end-2]
  @views Div_vol[:,:,1]       .+=  Gz_vol[:,:,1]	# boundaries
  @views Div_vol[:,:,end]     .+= -Gz_vol[:,:,end-1]
end


"""
    Total variation (TV) regularization term

Read the fODF amplitudes for a single vertex from a 3D volume in the workspace
and compute the TV regularization term in place, overwriting the volume
"""
function rumba_tv!(W::RUMBAwork{T}) where T <: AbstractFloat
  # Compute spatial gradients
  sd_grad!(W.Gx_vol, W.Gy_vol, W.Gz_vol, W.tv_vol)

  # Normalize spatial gradients
  W.tv_vol .= sqrt.(W.Gx_vol.^2 .+ W.Gy_vol.^2 .+ W.Gz_vol.^2 .+ eps(T))
  W.Gx_vol ./= W.tv_vol
  W.Gy_vol ./= W.tv_vol
  W.Gz_vol ./= W.tv_vol

  # Compute divergence
  sd_div!(W.Div_vol, W.Gx_vol, W.Gy_vol, W.Gz_vol)

  # Compute TV term (abs/eps to ensure values > 0)
  W.tv_vol .= 1 ./ (abs.(1 .- W.λ .* W.Div_vol) .+ eps(T))
end


"""
Initialize RUMBA-SD estimates
"""
function rumba_sd_initialize!(W::RUMBAwork{T}, fodf_0::Vector{T}, kernel::Matrix{T}, signal_mat::AbstractArray{T}, λ::T) where T <: AbstractFloat

  fill!(W.fodf_mat, 1)
  W.fodf_mat .*= fodf_0

  fill!(W.dodf_mat, 1)
  W.dodf_mat .*= (kernel * fodf_0)

  fill!(W.λ, λ)
  fill!(W.σ2_vec, λ)

  W.dodf_sig_mat .= (signal_mat .* W.dodf_mat) ./ W.σ2_vec

  fill!(W.tv_vol, 0)
  fill!(W.tv_mat, 1)
end


"""
    fodf:     Volume fractions of anisotropic compartments
    fgm:      Volume fraction of GM isotropic component
    fcsf:     Volume fraction of CSF isotropic component
    var:      Noise variance
    snr_mean: Estimated mean snr
    snr_std:  Estimated SNR standard deviation
"""
function rumba_sd_iterate!(W::RUMBAwork{T}, signal_mat::AbstractArray{T}, kernel::Array, ind_mask::Vector{Int}, iter::Int, n_order::Integer, coil_combine::String, ipat_factor::Integer, use_tv::Bool) where T <: AbstractFloat

  fzero = Float32(0)
  ε = eps(Float32)

  ndir, ncomp = size(kernel)

  # -------------------- R-L deconvolution term ------------------------ #
  # Ratio of modified Bessel functions of order n_order and n_order-1
  W.Iratio .= besseli_ratio.(n_order, W.dodf_sig_mat)

  mul!(W.rl_mat,  kernel', signal_mat .* W.Iratio)
  W.rl_mat ./= (kernel' * W.dodf_mat .+= ε)

  # -------------------- TV regularization term ------------------------ #
  @time if use_tv
    for icomp in 1:ncomp
      # Embed fODF amplitudes in brain mask
      fill!(W.tv_vol, 0)
      W.tv_vol[ind_mask] = W.fodf_mat[icomp, :]

      # Compute TV term in place
      rumba_tv!(W)

      # Extract TV term from brain mask
      W.tv_mat[icomp, :] = W.tv_vol[ind_mask]
    end
  end

  # ------------------------- Update estimate -------------------------- #
  # Enforce positivity
  W.fodf_mat .= max.(W.fodf_mat .* W.rl_mat .* W.tv_mat, fzero)

  if iter <= 100		&& false
    # Energy preservation at each step that included the bias on the s0 image
    # Only used to stabilize recovery in early iterations
    cte = sqrt(1 + 2*n_order*mean(W.σ2_vec))
    cte = sqrt(1 + n_order*mean(W.σ2_vec))
    cte = 1
    W.fodf_mat .= cte * W.fodf_mat ./ (sum(W.fodf_mat, dims=1) .+ ε)
  end

  mul!(W.dodf_mat, kernel, W.fodf_mat)
  W.dodf_sig_mat .= (signal_mat .* W.dodf_mat) ./ W.σ2_vec

  # --------------------- Noise variance estimate ---------------------- #
  W.σ2_vec .= sum((signal_mat.^2 + W.dodf_mat.^2)/2 -
              (W.σ2_vec .* W.dodf_sig_mat) .* W.Iratio, dims=1) /
              (n_order * ndir)

  # Assume that estimate of σ is in interval [1/snr_min, 1/snr_max],
  # where snr_min = 8 and snr_max = 80
  W.σ2_vec .= min.(Float32((1/8)^2), max.(W.σ2_vec, Float32((1/80)^2)))

  snr_mean = mean(1 ./ sqrt.(W.σ2_vec))
  snr_std  =  std(1 ./ sqrt.(W.σ2_vec))

  println("Estimated mean SNR (s0/σ) = " * string(snr_mean) *
          " (+-) " * string(snr_std))

  println("Number of coils = " * string(n_order))

  println("Mean sum(fODF) = " * string(mean(sum(W.fodf_mat, dims=1))))

  # ----------------- Update regularization parameter λ ----------------- #
  if use_tv
    if ipat_factor == 1
      # Penalize all voxels equally, assuming equal variance in all voxels
      # For low levels of noise, enforce a minimum level of regularization
      fill!(W.λ, max(mean(W.σ2_vec), Float32((1/30)^2)))
                               
    elseif ipat_factor > 1
      # Adaptive spatial regularization, assuming spatially inhomogeneous
      # variance, e.g., tissue dependent or due to parallel imaging
      # (in the future, λ could be low-pass filtered for robust estimation)
      fill!(W.λ, 0)
      W.λ[ind_mask] = W.σ2_vec
    end
  end
end


"""
Find ODF peaks, given a max number of peaks and an ODF amplitude threshold.
"""
function rumba_peaks(fodf, odf_vertices, idx_neig, thr, f_iso, npeak_max)

  # Use higher threshold in voxels with high f_iso
  #thr_xyz = thr + f_iso
  thr_xyz = thr / (1 - f_iso)

  # Find local maxima of ODF within a neighborhood around each vertex
  ispeak = find_local_peaks(fodf, idx_neig, thr_xyz)

  idx_peaks = findall(ispeak)
  v_peaks   = odf_vertices[idx_peaks, :]
  f_peaks   = fodf[idx_peaks]

  # Sort peaks and keep at most npeak_max of them
  npeak = length(f_peaks)
  ind_ord = sortperm(f_peaks, rev=true)

  if npeak > npeak_max
    npeak = npeak_max
    ind_ord = ind_ord[1:npeak_max]
  end

  idx_peaks = idx_peaks[ind_ord]
  v_peaks   = v_peaks[ind_ord, :]
  f_peaks   = f_peaks[ind_ord]

  # Find the true volume fraction of each anisotropic compartment,
  # factoring in the volume fractions of the isotropic compartments
  f_peaks = f_peaks * ((1 - f_iso) / sum(f_peaks))

  return vec((f_peaks .* v_peaks)')
end


"""
    find_local_peaks(odf, idx_neig, thr)

Find local maxima of ODF given a neighborhood around each vertex and an ODF
amplitude threshold.

Returns a bool vector the size of the ODF, where peak vertices are true.
"""
function find_local_peaks(odf, idx_neig, thr)

  ispeak = fill(false, size(odf))
  thr_abs = thr * maximum(odf)

  # Compare the ODF amplitude at each vertex to the amplitudes of its neighbors
  for ivert in 1:length(odf)
    (odf[ivert] < thr_abs) && continue

    if odf[ivert] > maximum(odf[idx_neig[ivert]])
      ispeak[ivert] = true
    end
  end

  return ispeak
end


"""
    rumba_rec(dwi::MRI, mask::MRI, odf_dirs::ODF=sphere_724, niter::Integer=600, λ_para::Float32=Float32(1.7*10^-3), λ_perp::Float32=Float32(0.2*10^-3), λ_csf::Float32=Float32(3.0*10^-3), λ_gm::Float32=Float32(0.8*10^-4), ncoils::Integer=1, coil_combine::String="SMF-SENSE", ipat_factor::Integer=1, use_tv::Bool=true)

Perform robust and unbiased model-based spherical deconvolution (RUMBA-SD)
reconstruction of DWIs, and return a `RUMBASD` structure.

If you use this method, please cite:
Erick J. Canales-Rodríguez, et al. (2015). Spherical deconvolution of multichannel diffusion MRI data with non-Gaussian noise models and spatial regularization. PLoS ONE, 10(10), e0138910. https://doi.org/10.1371/journal.pone.0138910

# Arguments
- `dwi::MRI`: A series of DWIs, stored in an `MRI` structure with valid `.bvec`
  and `.bval` fields
- `mask::MRI`: A brain mask volume, stored in an `MRI` structure
- `odf_dirs::ODF=sphere_724`: The vertices and faces of the ODF tessellation,
  stored in an `ODF` structure
- `λ_para::Float32=Float32(1.7*10^-3)`: Axial diffusivity in white-matter
  voxels with a single fiber population
- `λ_perp::Float32=Float32(0.2*10^-3)`: Radial diffusivity in white-matter 
  voxels with a single fiber population
- `λ_csf::Float32=Float32(3.0*10^-3)`: Mean diffusivity in CSF voxels
- `λ_gm::Float32=Float32(0.8*10^-4)`: Mean diffusivity in gray-matter voxels
- `ncoils::Integer=1`: Number of receive coil elements, if the DWIs were
  collected with parallel imaging, or 1 otherwise
- `coil_combine::String="SMF-SENSE"`: Method that was used to combine signals
  from receive coil elements (possible values: "SMF-SENSE" or "SoS-GRAPPA";
  has no effect if `ncoils` equals 1)
- `ipat_factor::Integer=1`: Acceleration factor, if the DWIs were collected
  with parallel imaging, or 1 otherwise
- `use_tv::Bool=true`: If true, include a total-variation regularization term
  in the FOD reconstruction

# Output
In the `RUMBASD` structure:
- `.fodf`: Volume fractions of anisotropic compartments (one per vertex)
- `.fgm`: Volume fraction of GM isotropic compartment
- `.fcsf`: Volume fraction of CSF isotropic compartment
- `.peak`: Orientation vectors of the 5 peak ODF amplitudes
- `.gfa`: Generalized fractional anisotropy
- `.var`: Estimated noise variance
- `.snr_mean`: Estimated mean SNR
- `.snr_std`: Estimated SNR standard deviation

"""
function rumba_rec(dwi::MRI, mask::MRI, odf_dirs::ODF=sphere_724, niter::Integer=600, λ_para::Float32=Float32(1.7*10^-3), λ_perp::Float32=Float32(0.2*10^-3), λ_csf::Float32=Float32(3.0*10^-3), λ_gm::Float32=Float32(0.8*10^-4), ncoils::Integer=1, coil_combine::String="SMF-SENSE", ipat_factor::Integer=1, use_tv::Bool=true)
 
  if isempty(dwi.bval)
    error("Missing b-value table from input DWI structure")
  end

  if isempty(dwi.bvec)
    error("Missing gradient table from input DWI structure")
  end

  n_order = 1

  if coil_combine == "SoS-GRAPPA"
    n_order = ncoils
  elseif coil_combine != "SMF-SENSE"
    error("Unknown coil combine mode " * coil_combine)
  end

  if ipat_factor < 1
    error("iPAT factor must be a positive integer")
  end

  # Rearrange signals so that average low-b volume is first, followed by DWIs,
  # and divide by average low-b volume
  signal = copy(dwi.vol)
  signal[signal .< 0] .= eps(Float32)

  ib0 = (dwi.bval .== minimum(dwi.bval))

  s0 = mean(signal[:, :, :, ib0], dims=4)
  signal = signal[:, :, :, .!ib0] ./ s0
  signal[isnan.(signal)] .= 0
  signal = cat(Float32.(s0 .> 0), signal, dims=4)	# signal=1 when b=0
  signal[signal .> 1] .= 1

  # Normalize gradient vectors 
  g = vcat([0 0 0], dwi.bvec[.!ib0, :] ./
                    sqrt.(sum(dwi.bvec[.!ib0, :].^2, dims=2)))
  b = vec(vcat([0], dwi.bval[.!ib0, :]))

  (nx, ny, nz, ndir) = size(signal)

  # Vertices and faces on the sphere where the fODF will be sampled
  nvert = size(odf_dirs.vertices, 1)
  half_vertices  = odf_dirs.vertices[1:Int(nvert/2), :]
  Vfaces = odf_dirs.faces

  # Find all vertices within an angular neigborhood
  if odf_dirs == sphere_724 || odf_dirs == sphere_642
    ang_neig = 12.5
  elseif odf_dirs == sphere_362
    ang_neig = 16
  end

  cos_ang = half_vertices * half_vertices'
  cos_ang[cos_ang .>  1] .=  1
  cos_ang[cos_ang .< -1] .= -1

  ang = acosd.(cos_ang)
  ang = min.(ang, 180 .- ang)

  isneig = (ang .< ang_neig)
  isneig[diagind(isneig)] .= 0
  idx_neig = findall.(eachslice(isneig, dims=1))

  #
  # Generate reconstruction kernel based on multi-tensor model
  #
  Kernel = Matrix{Float32}(undef, length(b), nvert+2)

  s0 = 1		# Assuming unit (normalized) signal

  # One radially symmetric tensor compartment for each fODF vertex
  (φ, θ) = cart2sph(odf_dirs.vertices[:,1], odf_dirs.vertices[:,2],
                                            odf_dirs.vertices[:,3])
  θ .= -θ

  for ivert in 1:nvert
    Kernel[:, ivert] .= tensor_model(φ[ivert], θ[ivert],
                                     [λ_para, λ_perp, λ_perp], b, g, s0)
  end

  # One isotropic tensor compartment for CSF
  Kernel[:, nvert+1] .= tensor_model(0, 0, [λ_csf, λ_csf, λ_csf], b, g, s0)

  # One isotropic tensor compartment for GM
  Kernel[:, nvert+2] .= tensor_model(0, 0, [λ_gm, λ_gm, λ_gm], b, g, s0)

  #
  # Initialize fODFs
  #
  fodf_0 = ones(Float32, size(Kernel,2))	# Uniform distribution on the sphere
  fodf_0 = fodf_0 / sum(fodf_0)

  nvert = div(nvert,2)			# Work on half sphere
  Kernel0 = Kernel[:, nvert+1:end]
  fodf_00 = fodf_0[(nvert+1):end]
  fodf_00 = fodf_00 / sum(fodf_00)		###
					### TODO: Do half Kernel above?
					###

  # Extract diffusion signal from brain mask
  (nx, ny, nz, ndir) = size(signal)
  nxyz = nx * ny * nz

  ind_mask = findall(vec(mask.vol) .> 0)
  nmask = length(ind_mask)

  signal_mat = reshape(signal, nxyz, ndir)[ind_mask, :]'

  # Allocate workspace
  ndir, ncomp = size(Kernel0)

  W = RUMBAwork(nmask, ndir, ncomp, use_tv ? (nx,ny,nz) : (0,0,0))

  # Initialize estimates
  σ0 = Float32(1/15)
  λ0 = σ0^2
  snr_mean = snr_std = Float32(0)

  rumba_sd_initialize!(W, fodf_00, Kernel0, signal_mat, λ0)

  #
  # Reconstruct fODFs
  #
  @time for iter in 1:niter
    println("Iteration " * string(iter) * " of " * string(niter))

    rumba_sd_iterate!(W, signal_mat, Kernel0, ind_mask, iter,
                      n_order, coil_combine, ipat_factor, use_tv)
  end

  # Energy preservation
  W.fodf_mat ./= (sum(W.fodf_mat, dims=1) .+ eps(Float32))

  #
  # Embed estimates in brain mask
  #
  fodf = MRI(mask, nvert, Float32)
  fcsf = MRI(mask, 1, Float32)
  fgm  = MRI(mask, 1, Float32)
  var  = MRI(mask, 1, Float32)
  gfa  = MRI(mask, 1, Float32)

  # Volume fractions of anisotropic WM compartments
  for icomp in 1:ncomp-2
    fodf.vol[ind_mask .+ (icomp-1)*nxyz] = W.fodf_mat[icomp, :]
  end

  # Volume fraction of isotropic GM compartment
  fgm.vol[ind_mask] = W.fodf_mat[ncomp, :]

  # Volume fraction of isotropic CSF compartment
  fcsf.vol[ind_mask] = W.fodf_mat[ncomp-1, :]

  f_iso = fgm.vol + fcsf.vol

  # Add isotropic components to ODF and normalize to sum=1
  fodf.vol .= fodf.vol .+ f_iso

  fodf.vol .= fodf.vol ./ sum(fodf.vol, dims=4)
  fodf.vol[isnan.(fodf.vol)] .= 0;

  # Noise variance
  var.vol[ind_mask] = W.σ2_vec

  #
  # Compute GFA
  #
#### TODO: Can provide a pre-computed mean to std -> fill(1/size(fodf.vol,4))
  gfa.vol .= std(fodf.vol, dims=4) ./ sqrt.(mean(fodf.vol.^2, dims=4))
  gfa.vol[isnan.(gfa.vol)] .= 0

  #
  # Extract ODF peaks
  #
  npeak = 5			# Max number of peaks per voxel
  fthresh = Float32(.1)		# Min volume fraction to retain a peak

  peak_mat = zeros(Float32, nx, ny, nz, 3*npeak)

  for iz in 1:nz
    for iy in 1:ny
      for ix in 1:nx
        (mask.vol[ix, iy, iz] == 0) && continue

        vecs = rumba_peaks(fodf.vol[ix, iy, iz, :], half_vertices, idx_neig,
                            fthresh, f_iso[ix, iy, iz], npeak)
        peak_mat[ix, iy, iz, 1:length(vecs)] = vecs
      end
    end
  end

  peak = Vector{MRI}(undef, npeak);

  for ipeak in 1:npeak
    peak[ipeak] = MRI(mask, 3, Float32)
    peak[ipeak].vol = peak_mat[:, :, :, 3*ipeak .+ (-2:0)]
  end

  return RUMBASD(fodf, fgm, fcsf, peak, gfa, var, snr_mean, snr_std)
end


"""
    rumba_write(rumba::RUMBASD, basename::String)

Write the volumes from a `RUMBASD` structure that was created by `rumba_rec()`
to files whose names start with the specified base name.
"""
function rumba_write(rumba::RUMBASD, basename::String)

  for var in fieldnames(RUMBASD)
    vartype = fieldtype(RUMBASD, var)

    if vartype == MRI
      fname = basename * "_" * string(var) * ".nii.gz"
      mri_write(getfield(rumba, var), fname)
    elseif vartype == Vector{MRI}
      for ivol in 1:length(getfield(rumba, var))
        fname = basename * "_" * string(var) * string(ivol) * ".nii.gz"
        mri_write(getfield(rumba, var)[ivol], fname)
      end
    else
      fname = basename * "_" * string(var) * ".txt"
      writedlm(fname, getfield(rumba, var), ' ')
    end
  end
end


