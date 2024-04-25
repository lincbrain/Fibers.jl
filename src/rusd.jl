#=
  Robust and unbiased model-based spherical deconvolution (RUMBA-SD)
  reconstruction
=#

using LinearAlgebra, Statistics, DelimitedFiles

export RUMBASD, rumba_rec, rumba_peaks!, rumba_write

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
- `isort::Vector{Vector{Int}}`   : Fiber ODF peak vertex indices (sorted)
- `fodf_peak::Vector{Vector{T}}` : Fiber ODF amplitudes at local peaks
- `dodf_mat::Matrix{T}` : Work matrices in diffusion ODF space
- `dodf_sig_mat::Matrix{T}`
- `Iratio::Matrix{T}`
- `fodf_mat::Matrix{T}` : Work matrices in fiber ODF space
- `rl_mat::Matrix{T}`
- `rl_mat2::Matrix{T}`
- `tv_mat::Matrix{T}`
- `tv_vol::Vector{Array{T,3}}` : Work volumes in image space
- `Gx_vol::Vector{Array{T,3}}`
- `Gy_vol::Vector{Array{T,3}}`
- `Gz_vol::Vector{Array{T,3}}`
- `Div_vol::Vector{Array{T,3}}`
- `λ::Array{T,3}`
- `σ2_vec::Matrix{T}`   : Noise variance and SNR by voxel
- `snr_vec::Matrix{T}`
"""
struct RUMBAwork{T}
  nmask::Int
  ndir::Int
  ncomp::Int
  nvox::NTuple{3,Int}
  isort::Vector{Vector{Int}}
  fodf_peak::Vector{Vector{T}}
  dodf_mat::Matrix{T}
  dodf_sig_mat::Matrix{T}
  Iratio::Matrix{T}
  fodf_mat::Matrix{T}
  rl_mat::Matrix{T}
  rl_mat2::Matrix{T}
  tv_mat::Matrix{T}
  tv_vol::Vector{Array{T,3}}
  Gx_vol::Vector{Array{T,3}}
  Gy_vol::Vector{Array{T,3}}
  Gz_vol::Vector{Array{T,3}}
  Div_vol::Vector{Array{T,3}}
  λ::Array{T,3}
  σ2_vec::Matrix{T}
  snr_vec::Matrix{T}

  function RUMBAwork(nmask::Int, ndir::Int, ncomp::Int,
                     nvox::Union{NTuple{3,Int}, Nothing}=nothing,
                     T::DataType=Float32)

    # Work vectors for fiber ODF peak search
    isort        = [Vector{Int}(undef, ncomp-2) for tid in 1:Threads.nthreads()]
    fodf_peak    = [Vector{T}(undef, ncomp-2)   for tid in 1:Threads.nthreads()]

    # Work matrices in diffusion ODF space
    dodf_mat     = Matrix{T}(undef, ndir, nmask)
    dodf_sig_mat = Matrix{T}(undef, ndir, nmask)
    Iratio       = Matrix{T}(undef, ndir, nmask)

    # Work matrices in fiber ODF space
    fodf_mat     = Matrix{T}(undef, ncomp, nmask)
    rl_mat       = Matrix{T}(undef, ncomp, nmask)
    rl_mat2      = Matrix{T}(undef, ncomp, nmask)
    tv_mat       = Matrix{T}(undef, ncomp, nmask)

    # Work volumes in image space
    isnothing(nvox) && (nvox = (0,0,0))
    tv_vol       = [Array{T,3}(undef, nvox) for tid in 1:Threads.nthreads()]
    Gx_vol       = [Array{T,3}(undef, nvox) for tid in 1:Threads.nthreads()]
    Gy_vol       = [Array{T,3}(undef, nvox) for tid in 1:Threads.nthreads()]
    Gz_vol       = [Array{T,3}(undef, nvox) for tid in 1:Threads.nthreads()]
    Div_vol      = [Array{T,3}(undef, nvox) for tid in 1:Threads.nthreads()]
    λ            = Array{T,3}(undef, nvox)

    # Noise variance and SNR by voxel
    σ2_vec       = Matrix{T}(undef, 1, nmask)
    snr_vec      = Matrix{T}(undef, 1, nmask)

    new{T}(
      nmask::Int,
      ndir::Int,
      ncomp::Int,
      nvox::NTuple{3,Int},
      isort::Vector{Vector{Int}},
      fodf_peak::Vector{Vector{T}},
      dodf_mat::Matrix{T},
      dodf_sig_mat::Matrix{T},
      Iratio::Matrix{T},
      fodf_mat::Matrix{T},
      rl_mat::Matrix{T},
      rl_mat2::Matrix{T},
      tv_mat::Matrix{T},
      tv_vol::Vector{Array{T,3}},
      Gx_vol::Vector{Array{T,3}},
      Gy_vol::Vector{Array{T,3}},
      Gz_vol::Vector{Array{T,3}},
      Div_vol::Vector{Array{T,3}},
      λ::Array{T,3},
      σ2_vec::Matrix{T},
      snr_vec::Matrix{T}
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

  tid = Threads.threadid()

  # Compute spatial gradients
  sd_grad!(W.Gx_vol[tid], W.Gy_vol[tid], W.Gz_vol[tid], W.tv_vol[tid])

  # Normalize spatial gradients
  W.tv_vol[tid] .= sqrt.(W.Gx_vol[tid].^2 .+ W.Gy_vol[tid].^2 .+
                                             W.Gz_vol[tid].^2 .+ eps(T))
  W.Gx_vol[tid] ./= W.tv_vol[tid]
  W.Gy_vol[tid] ./= W.tv_vol[tid]
  W.Gz_vol[tid] ./= W.tv_vol[tid]

  # Compute divergence
  sd_div!(W.Div_vol[tid], W.Gx_vol[tid], W.Gy_vol[tid], W.Gz_vol[tid])

  # Compute TV term (abs/eps to ensure values > 0)
  W.tv_vol[tid] .= 1 ./ (abs.(1 .- W.λ .* W.Div_vol[tid]) .+ eps(T))
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

  fzero = T(0)
  ε = eps(T)

  ndir, ncomp = size(kernel)

  # -------------------- R-L deconvolution term ------------------------ #
  # Ratio of modified Bessel functions of order n_order and n_order-1
  W.Iratio .= besseli_ratio.(n_order, W.dodf_sig_mat)

  mul!(W.rl_mat,  kernel', signal_mat .* W.Iratio)
  mul!(W.rl_mat2, kernel', W.dodf_mat)
  W.rl_mat ./= (W.rl_mat2 .+= ε)

  # -------------------- TV regularization term ------------------------ #
  @time if use_tv
    Threads.@threads for icomp in 1:ncomp
      tid = Threads.threadid()

      # Embed fODF amplitudes in brain mask
      fill!(W.tv_vol[tid], 0)
      W.tv_vol[tid][ind_mask] = view(W.fodf_mat, icomp, :)

      # Compute TV term in place
      rumba_tv!(W)

      # Extract TV term from brain mask
      W.tv_mat[icomp, :] = view(W.tv_vol[tid], ind_mask)
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
    W.fodf_mat .= cte .* W.fodf_mat ./ (sum(W.fodf_mat, dims=1) .+ ε)
  end

  mul!(W.dodf_mat, kernel, W.fodf_mat)
  W.dodf_sig_mat .= (signal_mat .* W.dodf_mat) ./ W.σ2_vec

  # --------------------- Noise variance estimate ---------------------- #
  W.Iratio .= (signal_mat.^2 + W.dodf_mat.^2) ./ 2 .-
              (W.σ2_vec .* W.dodf_sig_mat) .* W.Iratio
  W.σ2_vec .= sum(W.Iratio , dims=1) ./ (n_order * ndir)

  # Assume that estimate of σ is in interval [1/snr_min, 1/snr_max],
  # where snr_min = 8 and snr_max = 80
  clamp!(W.σ2_vec, T((1/80)^2), T((1/8)^2))

  W.snr_vec .= 1 ./ sqrt.(W.σ2_vec)

  # ----------------- Update regularization parameter λ ----------------- #
  if use_tv
    if ipat_factor == 1
      # Penalize all voxels equally, assuming equal variance in all voxels
      # For low levels of noise, enforce a minimum level of regularization
      fill!(W.λ, max(mean(W.σ2_vec), T((1/30)^2)))
                               
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
Find fODF peaks, given an fODF amplitude vector and an amplitude threshold.

Return the number of peaks found.
"""
function rumba_peaks!(W::RUMBAwork{T}, fodf::AbstractVector{T}, f_iso::T, idx_neig::Vector{Vector{Int}}, thr::T) where T <: AbstractFloat

  tid = Threads.threadid()

  # Use higher threshold in voxels with high f_iso
  #thr_xyz = thr + f_iso
  thr_xyz = thr / (1 - f_iso)

  # Find local maxima of ODF within a neighborhood around each vertex
  W.fodf_peak[tid] .= fodf

  thr_abs = thr_xyz * maximum(fodf)

  # Compare the fODF amplitude at each vertex to the amplitudes of its neighbors
  for ivert in 1:length(fodf)
    if fodf[ivert] < thr_abs ||
       fodf[ivert] .<= maximum(@view fodf[idx_neig[ivert]])
      W.fodf_peak[tid][ivert] = 0
    end
  end

  # Sort peaks by decreasing amplitude
  sortperm!(W.isort[tid], W.fodf_peak[tid], rev=true)

  return length(findall(W.fodf_peak[tid] .> 0))
end


"""
    rumba_rec(dwi::MRI{T}, mask::MRI, odf_dirs::ODF=sphere_724, niter::Integer=600, λ_para::T=T(1.7*10^-3), λ_perp::T=T(0.2*10^-3), λ_csf::T=T(3.0*10^-3), λ_gm::T=T(0.8*10^-4), ncoils::Integer=1, coil_combine::String="SMF-SENSE", ipat_factor::Integer=1, use_tv::Bool=true) where T <: AbstractFloat

Perform robust and unbiased model-based spherical deconvolution (RUMBA-SD)
reconstruction of DWIs, and return a `RUMBASD` structure.

If you use this method, please cite:
Erick J. Canales-Rodríguez, et al. (2015). Spherical deconvolution of multichannel diffusion MRI data with non-Gaussian noise models and spatial regularization. PLoS ONE, 10(10), e0138910. https://doi.org/10.1371/journal.pone.0138910

# Arguments
- `dwi::MRI{T}`: A series of DWIs, stored in an `MRI` structure with valid `.bvec`
  and `.bval` fields
- `mask::MRI`: A brain mask volume, stored in an `MRI` structure
- `odf_dirs::ODF=sphere_724`: The vertices and faces of the ODF tessellation,
  stored in an `ODF` structure
- `λ_para::T=T(1.7*10^-3)`: Axial diffusivity in white-matter
  voxels with a single fiber population
- `λ_perp::T=T(0.2*10^-3)`: Radial diffusivity in white-matter 
  voxels with a single fiber population
- `λ_csf::T=T(3.0*10^-3)`: Mean diffusivity in CSF voxels
- `λ_gm::T=T(0.8*10^-4)`: Mean diffusivity in gray-matter voxels
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
function rumba_rec(dwi::MRI{Array{T,4}}, mask::MRI, odf_dirs::ODF=sphere_724, niter::Integer=600, λ_para::T=T(1.7*10^-3), λ_perp::T=T(0.2*10^-3), λ_csf::T=T(3.0*10^-3), λ_gm::T=T(0.8*10^-4), ncoils::Integer=1, coil_combine::String="SMF-SENSE", ipat_factor::Integer=1, use_tv::Bool=true) where T <: AbstractFloat
 
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

  nx, ny, nz = size(mask.vol)
  nxyz = nx * ny * nz

  ind_mask = findall(vec(mask.vol) .> 0)
  nmask = length(ind_mask)

  ib0 = (dwi.bval .== minimum(dwi.bval))
  ndir = length(findall(.!ib0)) + 1

  # Extract diffusion signal from brain mask
  # Rearrange signals so that average low-b volume is first, followed by DWIs
  signal_mat = Array{T, 2}(undef, ndir, nmask)

  @views signal_mat[1, :] .=
         mean(max.(dwi.vol[:, :, :, ib0], 0), dims=4)[ind_mask]
  @views signal_mat[2:end, :] .=
         reshape(max.(dwi.vol[:, :, :, .!ib0], 0), nxyz, ndir-1)[ind_mask, :]'

  # Divide DWIs by average low-b volume
  @views signal_mat[2:end, :] ./= signal_mat[1, :]'
  @views signal_mat[isnan.(signal_mat)] .= 0

  # signal=1 if b=0
  @views signal_mat[1, :] .= T.(signal_mat[1, :] .> 0)
  @views signal_mat[signal_mat .> 1] .= 1

  # Normalize gradient vectors 
  g = vcat([0 0 0], dwi.bvec[.!ib0, :] ./
                    sqrt.(sum(dwi.bvec[.!ib0, :].^2, dims=2)))
  b = vec(vcat([0], dwi.bval[.!ib0, :]))

  # Vertices and faces on the sphere where the fODF will be sampled
  nvert = size(odf_dirs.vertices, 1)
  nvert = div(nvert,2)			# Will work on half sphere
  half_vertices  = odf_dirs.vertices[1:nvert, :]

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
  Kernel = Matrix{T}(undef, length(b), nvert+2)

  s0 = 1		# Assuming unit (normalized) signal

  # One radially symmetric tensor compartment for each fODF vertex
  (φ, θ) = cart2sph(odf_dirs.vertices[nvert+1:end,1],
                    odf_dirs.vertices[nvert+1:end,2],
                    odf_dirs.vertices[nvert+1:end,3])
  θ .= -θ

  for ivert in 1:nvert
    Kernel[:, ivert] .= tensor_model(φ[ivert], θ[ivert],
                                     [λ_para, λ_perp, λ_perp], b, g, s0)
  end

  # One isotropic tensor compartment for CSF
  Kernel[:, nvert+1] .= tensor_model(0, 0, [λ_csf, λ_csf, λ_csf], b, g, s0)

  # One isotropic tensor compartment for GM
  Kernel[:, nvert+2] .= tensor_model(0, 0, [λ_gm, λ_gm, λ_gm], b, g, s0)

  # Total number of compartments
  ncomp = nvert+2

  #
  # Initialize fODFs (uniform distribution on the half sphere)
  #
  fodf_0 = ones(T, nvert+2)
  	fodf_0 .= fodf_0 ./ (2*nvert+2)
  fodf_0 .= fodf_0 ./ sum(fodf_0)

  # Allocate workspace
  W = RUMBAwork(nmask, ndir, ncomp, use_tv ? (nx,ny,nz) : (0,0,0))

  # Initialize estimates
  σ0 = T(1/15)
  λ0 = σ0^2
  snr_mean = snr_std = T(0)

  rumba_sd_initialize!(W, fodf_0, Kernel, signal_mat, λ0)

  #
  # Reconstruct fODFs
  #
  @time for iter in 1:niter
    println("Iteration " * string(iter) * " of " * string(niter))

    @time rumba_sd_iterate!(W, signal_mat, Kernel, ind_mask, iter,
                      n_order, coil_combine, ipat_factor, use_tv)

    snr_mean = mean(W.snr_vec)
    snr_std  =  std(W.snr_vec; mean=snr_mean)

    println("Estimated mean SNR (s0/σ) = " * string(snr_mean) *
            " (+-) " * string(snr_std))

    println("Number of coils = " * string(n_order))

    println("Mean sum(fODF) = " * string(mean(sum(W.fodf_mat, dims=1))))
  end

  # Energy preservation
  W.fodf_mat ./= (sum(W.fodf_mat, dims=1) .+ eps(T))

  #
  # Embed estimates in brain mask
  #
  fodf = MRI(mask, nvert, T)
  fcsf = MRI(mask, 1, T)
  fgm  = MRI(mask, 1, T)
  var  = MRI(mask, 1, T)
  gfa  = MRI(mask, 1, T)

  # Volume fractions of anisotropic WM compartments
  for icomp in 1:ncomp-2
    @views fodf.vol[ind_mask .+ (icomp-1)*nxyz] .= W.fodf_mat[icomp, :]
  end

  # Volume fraction of isotropic CSF compartment
  fcsf.vol[ind_mask] = W.fodf_mat[end-1, :]

  # Volume fraction of isotropic GM compartment
  fgm.vol[ind_mask] = W.fodf_mat[end, :]

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
  gfa.vol .= std(fodf.vol, dims=4) ./ sqrt.(mean(fodf.vol.^2, dims=4))
  gfa.vol[isnan.(gfa.vol)] .= 0

  #
  # Extract ODF peaks
  #
  npeak = 5			# Max number of peaks per voxel
  fthresh = T(.1)		# Min volume fraction to retain a peak

  peak = Vector{MRI}(undef, npeak);

  for ipeak in 1:npeak
    peak[ipeak] = MRI(mask, 3, T)
  end

  Threads.@threads for iz in 1:nz
    for iy in 1:ny
      for ix in 1:nx
        (mask.vol[ix, iy, iz] == 0) && continue

        tid = Threads.threadid()

        nvalid = rumba_peaks!(W, fodf.vol[ix, iy, iz, :], f_iso[ix, iy, iz],
                              idx_neig, fthresh)

        n = min(nvalid, npeak)

        fnorm = (1 - f_iso[ix, iy, iz]) /
                sum(fodf.vol[ix, iy, iz, W.isort[tid][1:n]])

        for ipeak in 1:n
          @views peak[ipeak].vol[ix, iy, iz, :] .=
                   half_vertices[W.isort[tid][ipeak], :] .*
                   (fodf.vol[ix, iy, iz, W.isort[tid][ipeak]] * fnorm)
        end
      end
    end
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


