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
    multi_tensor_model(φ, θ, f, λ::Vector, b::Vector, g::Matrix, s0::Number)

    Compute the expected diffusion signal in a voxel that contains a set of
    compartments with orientation angles `φ`, `θ` and volume fractions `f`.

    Assume that diffusion in all compartments can be modeled by a tensor with
    eigenvalues `λ`, that the signal was acquired with b-values `b` and
    gradient vectors `g`, and that the non-diffusion-weighted signal is `s0`.
"""
function multi_tensor_model(φ, θ, f, λ::Vector, b::Vector, g::Matrix, s0::Number)

  if length(λ) != 3
    error("Length of diffusivity vector " * string(λ) * " must be 3")
  end

  if length(φ) != length(θ)
    error("Lengths of polar angle (" * length(φ) * ") and " *
          "azimuthal angle (" * length(θ) * ") vectors do not match")
  end

  if length(θ) != length(f)
    error("Lengths of angle (" * length(θ) * ") and " *
          "compartment volume (" * length(f) * ") vectors do not match")
  end

  Λ = Diagonal(λ)
  fsum = sum(f)

  # Iterate over compartments
  S = 0
  for icomp in 1:length(f)
    R = ang2rot(φ[icomp], θ[icomp])
    D = R*Λ*R'
    S = S .+ f[icomp] / fsum * exp.(-b .* diag(g*D*g'))
  end
  S = s0*S

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
function besseli_ratio(nu::Integer, z::Float32)

  return z / ((2*nu + z) - 
               ((2*nu+1)*z / 
               (2*z + (2*nu+1) - 
               ((2*nu+3)*z /
               ((2*nu+2) + 2*z - ((2*nu+5) * z / ((2*nu+3) + 2*z)))))))
end


"""
    Gradient operator
"""
function sd_grad(M)

  fx = M[[2:end; end], :, :] - M
  fy = M[:, [2:end; end], :] - M
  fz = M[:, :, [2:end; end]] - M

  return cat(fx, fy, fz, dims=4)
end


"""
    Divergence operator
"""
function sd_div(P)

  Px = P[:,:,:,1]
  Py = P[:,:,:,2]
  Pz = P[:,:,:,3]

  fx = Px - Px[[1; 1:end-1], :, :]
  fx[1,:,:]   =  Px[1,:,:]	# boundary
  fx[end,:,:] = -Px[end-1,:,:]

  fy = Py - Py[:, [1; 1:end-1], :]
  fy[:,1,:]   =  Py[:,1,:]	# boundary
  fy[:,end,:] = -Py[:,end-1,:]

  fz = Pz - Pz[:, :, [1; 1:end-1]]
  fz[:,:,1]   =  Pz[:,:,1]	# boundary
  fz[:,:,end] = -Pz[:,:,end-1]

  return fx + fy + fz
end


"""
    Total variation regularization term
"""
function rumba_tv(fodf_mat::Array{Float32, 3}, λ::Float32)
  # Compute gradient and divergence
  Grad = sd_grad(fodf_mat)

  d = sqrt.(sum(Grad.^2, dims=4) .+ eps(Float32))
  Div = sd_div(Grad ./ d)

  # abs/eps to ensure values > 0
  tv_mat = 1 ./ (abs.(1 .- λ .* Div) .+ eps(Float32))

  return tv_mat
end


"""
    fodf:     Volume fractions of anisotropic compartments
    fgm:      Volume fraction of GM isotropic component
    fcsf:     Volume fraction of CSF isotropic component
    var:      Noise variance
    snr_mean: Estimated mean snr
    snr_std:  Estimated SNR standard deviation
"""
function rumba_sd_iterate(signal::Array, kernel::Array, fodf_0::Array, niter::Integer, mask::Array, n_order::Integer, coil_combine::String, ipat_factor::Integer, use_tv::Bool)

  (nx, ny, nz, ndir) = size(signal)
  nxyz = nx * ny * nz

  ind_mask = findall(vec(mask) .> 0)
  nmask = length(ind_mask)

  signal_mat = reshape(signal, nxyz, ndir)[ind_mask, :]
  signal_mat = permutedims(signal_mat, [2 1])

  fodf_mat = repeat(fodf_0, 1, nmask)

  dodf_mat = repeat(kernel * fodf_0, 1, nmask)

  fzero = Float32(0)
  λ_aux = zeros(Float32, nx, ny, nz)

  # --------------------------- Main algorithm --------------------------- #
  σ0 = Float32(1/15)
  λ = σ0^2
  σ2 = λ * ones(Float32, size(signal_mat))
  ε = 1f-7

  dodf_sig_mat = (signal_mat .* dodf_mat) ./ σ2
  σ2_i = []

  snr_mean = snr_std = Float32(0)

  @time for iter in 1:niter
    println("Iteration " * string(iter) * " of " * string(niter))

    # -------------------- R-L deconvolution term ------------------------ #
    # Ratio of modified Bessel functions of order n_order and n_order-1
    Iratio = besseli_ratio.(n_order, dodf_sig_mat)

    rl_mat = (kernel' * (signal_mat .* Iratio)) ./
             (kernel' * dodf_mat .+ eps(Float32))

    # -------------------- TV regularization term ------------------------ #
    tv_mat = ones(Float32, size(fodf_mat))

    @time if use_tv
      for idir in 1:size(fodf_mat, 1)
        # Embed in brain mask
        fodf = zeros(Float32, nx, ny, nz)
        fodf[ind_mask] = fodf_mat[idir, :]

        tv = rumba_tv(fodf, λ)
        tv_mat[idir, :] = tv[ind_mask]
      end
    end

    # ------------------------- Update estimate -------------------------- #
    fodf_mat = max.(fodf_mat .* rl_mat .* tv_mat, fzero) # Enforce positivity

    if iter <= 100		&& false
      # Energy preservation at each step that included the bias on the s0 image
      # Only used to stabilize recovery in early iterations
      cte = sqrt(1 + 2*n_order*mean(σ2))
      cte = sqrt(1 + n_order*mean(σ2))
      cte = 1
      fodf_mat = cte * fodf_mat ./ (sum(fodf_mat, dims=1) .+ eps(Float32)) 
    end

    dodf_mat = kernel * fodf_mat
    dodf_sig_mat = (signal_mat .* dodf_mat) ./ σ2

    # --------------------- Noise variance estimate ---------------------- #
    σ2_i = sum((signal_mat.^2 + dodf_mat.^2)/2 -
               (σ2 .* dodf_sig_mat) .* Iratio, dims=1) / (n_order * ndir)

    # Assume that estimate of σ is in interval [1/snr_min, 1/snr_max],
    # where snr_min = 8 and snr_max = 80
    σ2_i = min.(Float32((1/8)^2), max.(σ2_i, Float32((1/80)^2)))

    snr_mean = mean(1 ./ sqrt.(σ2_i))
    snr_std  =  std(1 ./ sqrt.(σ2_i))

    println("Estimated mean SNR (s0/σ) = " * string(snr_mean) *
            " (+-) " * string(snr_std))

    println("Number of coils = " * string(n_order))

    println("Mean sum(fODF) = " * string(mean(sum(fodf_mat, dims=1))))

    σ2 = repeat(σ2_i, ndir)

    # ----------------- Update regularization parameter λ ----------------- #
    if use_tv
      if ipat_factor == 1
        # Penalize all voxels equally, assuming equal variance in all voxels
        λ = mean(σ2_i)
                                 
        # For low levels of noise, enforce a minimum level of regularization
        λ = max(λ, Float32((1/30)^2))
      elseif ipat_factor > 1
        # Adaptive spatial regularization, assuming spatially inhomogeneous
        # variance, e.g., tissue dependent or due to parallel imaging
        # (in the future, λ could be low-pass filtered for robust estimation)
        λ = zeros(Float32, [nx ny nz])
        λ[ind_mask] = σ2_i
      end
    end
  end

  # Energy preservation
  fodf_mat = fodf_mat ./ (sum(fodf_mat, dims=1) .+ eps(Float32))

  noisevar = zeros(Float32, nx, ny, nz)
  noisevar[ind_mask] = σ2_i
  # ---------------------- End of main algorithm ------------------------- #

  # Embed ODFs into brain mask
  ncomp = length(fodf_0)

  # Volume fractions of anisotropic WM compartments
  fodf = zeros(Float32, nxyz, ncomp-2)
  for icomp in 1:ncomp-2
    fodf[ind_mask, icomp] = fodf_mat[icomp, :]
  end
  fodf = reshape(fodf, nx, ny, nz, ncomp-2)

  # Volume fraction of isotropic GM compartment
  fgm = zeros(Float32, nx, ny, nz)
  fgm[ind_mask] = fodf_mat[ncomp, :]

  # Volume fraction of isotropic CSF compartment
  fcsf = zeros(Float32, nx, ny, nz)
  fcsf[ind_mask] = fodf_mat[ncomp-1, :]

  return fodf, fgm, fcsf, noisevar, snr_mean, snr_std
end


"""
Find ODF peaks, given a max number of peaks and an ODF amplitude threshold.
"""
function rumba_peaks(fodf, odf_vertices, idx_neig, thr, f_iso, npeak_max)

  nvert = size(odf_vertices, 1)
  nhalf = Int(nvert/2)
  half_vertices = odf_vertices[1:nhalf, :]	# Vertices on the half sphere

  (nx, ny, nz, nv) = size(fodf)
  peaks = zeros(Float32, nx, ny, nz, 3*npeak_max)

  for iz in 1:nz
    for iy in 1:ny
      for ix in 1:nx
        f_iso_xyz = f_iso[ix, iy, iz]

        # Use higher threshold in voxels with high f_iso
        #thr_xyz = thr + f_iso_xyz
        thr_xyz = thr / (1 - f_iso_xyz)

        fodf_xyz = fodf[ix, iy, iz, 1:nhalf]	# Amplitudes on the half sphere

        (sum(fodf_xyz) == 0) && continue

        # Find local maxima of ODF within a neighborhood around each vertex
        ispeak = find_local_peaks(fodf_xyz, idx_neig, thr_xyz)

        idx_peaks = findall(ispeak)
        v_peaks   = half_vertices[idx_peaks, :]
        f_peaks   = fodf_xyz[idx_peaks]

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
        f_peaks = f_peaks * ((1 - f_iso_xyz) / sum(f_peaks))

        peaks[ix, iy, iz, 1:3*npeak] = vec((f_peaks .* v_peaks)')
      end
    end
  end

  return peaks
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
  E = copy(dwi.vol)
  E[E .< 0] .= eps(Float32)

  ib0 = (dwi.bval .== minimum(dwi.bval))

  s0 = mean(E[:, :, :, ib0], dims=4)
  E = E[:, :, :, .!ib0] ./ s0
  E[isnan.(E)] .= 0
  E = cat(Float32.(s0 .> 0), E, dims=4)		# E=1 when b=0
  E[E .> 1] .= 1

  # Normalize gradient vectors 
  g = vcat([0 0 0], dwi.bvec[.!ib0, :] ./
                    sqrt.(sum(dwi.bvec[.!ib0, :].^2, dims=2)))
  b = vec(vcat([0], dwi.bval[.!ib0, :]))

  (nx, ny, nz, ndir) = size(E)

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

  fi = 1		# Assuming unit volume fraction
  s0 = 1		# Assuming unit signal

  # One radially symmetric tensor compartment for each fODF vertex
  (φ, θ) = cart2sph(odf_dirs.vertices[:,1], odf_dirs.vertices[:,2],
                                            odf_dirs.vertices[:,3])
  θ = -θ

  for ivert in 1:nvert
    Kernel[:, ivert] .= multi_tensor_model(φ[ivert], θ[ivert], fi,
                                          [λ_para, λ_perp, λ_perp], b, g, s0)
  end

  # One isotropic tensor compartment for CSF
  Kernel[:, nvert+1] .= multi_tensor_model(0, 0, fi,
                                          [λ_csf, λ_csf, λ_csf], b, g, s0)

  # One isotropic tensor compartment for GM
  Kernel[:, nvert+2] .= multi_tensor_model(0, 0, fi,
                                          [λ_gm, λ_gm, λ_gm], b, g, s0)

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

  fodf  = MRI(mask, nvert, Float32)
  fcsf  = MRI(mask, 1, Float32)
  fgm   = MRI(mask, 1, Float32)
  var   = MRI(mask, 1, Float32)

  #
  # Reconstruct fODFs
  #
  (fodf.vol, fgm.vol, fcsf.vol, var.vol, snr_mean, snr_std) =
    rumba_sd_iterate(E, Kernel0, fodf_00, niter, mask.vol,
                     n_order, coil_combine, ipat_factor, use_tv)

  f_iso = fgm.vol + fcsf.vol

  #
  # Add isotropic components to ODF and normalize to sum=1
  #
  fodf.vol .= fodf.vol .+ f_iso

  fodf.vol .= fodf.vol ./ sum(fodf.vol, dims=4)
  fodf.vol[isnan.(fodf.vol)] .= 0;

  #
  # Compute GFA
  #
  gfa = MRI(mask, 1, Float32)

#### TODO: Can provide a pre-computed mean to std -> fill(1/size(fodf.vol,4))
  gfa.vol .= std(fodf.vol, dims=4) ./ sqrt.(mean(fodf.vol.^2, dims=4))
  gfa.vol[isnan.(gfa.vol)] .= 0

  #
  # Extract ODF peaks
  #
  npeak = 5			# Max number of peaks per voxel
  fthresh = Float32(.1)		# Min volume fraction to retain a peak

  peak_mat =
    rumba_peaks(fodf.vol, odf_dirs.vertices, idx_neig, fthresh, f_iso, npeak)

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


