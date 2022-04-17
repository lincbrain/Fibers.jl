#=
  Original Author: Anastasia Yendiki

  Copyright © 2022 The General Hospital Corporation (Boston, MA) "MGH"
 
  Terms and conditions for use, reproduction, distribution and contribution
  are found in the 'FreeSurfer Software License Agreement' contained
  in the file 'LICENSE' found in the FreeSurfer distribution, and here:
 
  https://surfer.nmr.mgh.harvard.edu/fswiki/FreeSurferSoftwareLicense
 
  Reporting: freesurfer@nmr.mgh.harvard.edu
=#

using Statistics

export GQI, gqi_rec, gqi_write

"Container for outputs of a GQI fit"
struct GQI
  odf::MRI
  peak::Vector{MRI}
  qa::Vector{MRI}
end


"""
    GQIwork{T}

Pre-allocated workspace for GQI reconstruction computations

- `T::DataType`         : Data type for computations (default: `Float32`)
- `nvol::Int`           : Number of volumes in DWI series
- `nvert::Int`          : Number of ODF vertices (on the half sphere)
- `isort::Vector{Int}`  : Indices of ODF peak vertices (sorted) [`nvert`]
- `s::Vector{T}`        : DWI signal [`nvol`]
- `o::Vector{T}`        : ODF amplitudes [`nvert`]
- `odf_peak::Vector{T}` : ODF amplitudes at local peaks [`nvert`]
- `faces::Matrix{Int}`  : ODF faces (on the half sphere) [nvert x 3]
- `A::Matrix{T}`        : System matrix [nvert x nvol]
"""
struct GQIwork{T}
  nvol::Int
  nvert::Int
  isort::Vector{Int}
  s::Vector{T}
  o::Vector{T}
  odf_peak::Vector{T}
  faces::Matrix{Int}
  A::Matrix{T}

  function GQIwork(bval::Vector{Float32}, bvec::Matrix{Float32}, odf_dirs::ODF=sphere_642, σ::Float32=Float32(1.25), T::DataType=Float32)

    # Number of volumes in DWI series
    nvol = length(bval)

    # Number of ODF vertices (on the half sphere)
    nvert = div(size(odf_dirs.vertices, 1), 2)

    # Indices of sorted ODF peaks
    isort = Vector{Int}(undef, nvert)

    # DWI signal
    s = Vector{T}(undef, nvol)

    # ODF amplitudes
    o = Vector{T}(undef, nvert)

    # ODF amplitudes at local peaks
    odf_peak = Vector{T}(undef, nvert)

    # ODF faces (on the half sphere)
    faces = copy(odf_dirs.faces)
    faces[faces .> nvert] .-= nvert

    # System matrix
    A = Matrix{T}(undef, nvert, nvol)
    bq_vector = bvec .* (sqrt.(bval * T(0.01506)) * T(σ / π))
    A .= sinc.(odf_dirs.vertices[nvert+1:end, :] * bq_vector')

    new{T}(
      nvol,
      nvert,
      isort,
      s,
      o,
      odf_peak,
      faces,
      A
    )
  end
end


"""
    gqi_rec(dwi::MRI, mask::MRI, odf_dirs::ODF=sphere_642, σ::Float32=Float32(1.25))

Perform generalized q-space imaging (GQI) reconstruction of DWIs, and return a
`GQI` structure.

If you use this method, please cite:
Fang-Cheng Yeh, et al. (2010). Generalized q-sampling imaging. IEEE Transactions on Medical Imaging, 29(9), 1626–1635. https://doi.org/10.1109/TMI.2010.2045126

# Arguments
- `dwi::MRI`: A series of DWIs, stored in an `MRI` structure with valid `.bvec`
   and `.bval` fields
- `mask::MRI`: A brain mask volume, stored in an `MRI` structure
- `odf_dirs::ODF=sphere_642`: The vertices and faces of the ODF tessellation,
  stored in an `ODF` structure
- `σ::Float32=Float32(1.25)`: Diffusion sampling length factor

# Output
In the `GQI` structure:
- `.odf`: ODF amplitudes on the half sphere
- `.peak`: Orientation vectors of the 3 peak ODF amplitudes
- `.qa`: Quantitative anisotropy for each of the 3 peak orientations

"""
function gqi_rec(dwi::MRI, mask::MRI, odf_dirs::ODF=sphere_642, σ::Float32=Float32(1.25))

  if isempty(dwi.bval)
    error("Missing b-value table from input DWI structure")
  end

  if isempty(dwi.bvec)
    error("Missing gradient table from input DWI structure")
  end

  npeak = 3;

  W = GQIwork(dwi.bval, dwi.bvec, odf_dirs, σ)

  odf  = MRI(mask, W.nvert)
  peak = Vector{MRI}(undef, npeak)
  qa   = Vector{MRI}(undef, npeak)

  for ipeak in 1:npeak
    peak[ipeak] = MRI(mask, 3)
    qa[ipeak]   = MRI(mask, 1)
  end

  odfmax = 0

  Threads.@threads for iz in 1:size(dwi.vol, 3)
    for iy in 1:size(dwi.vol, 2)
      for ix in 1:size(dwi.vol, 1)
        mask.vol[ix, iy, iz] == 0 && continue

        W.s .= dwi.vol[ix, iy, iz, :]
        W.s[W.s .< 0] .= 0

        maximum(W.s) == 0 && continue

        mul!(W.o, W.A, W.s)
        odf.vol[ix, iy, iz, :] = W.o

        odfmax = max.(odfmax, mean(W.o))
        odfmin = minimum(W.o)

        nvalid = find_peaks!(W)

        for ipeak in 1:npeak
          ipeak > nvalid && continue

          peak[ipeak].vol[ix, iy, iz, :] = odf_dirs.vertices[W.isort[ipeak], :]
          qa[ipeak].vol[ix, iy, iz]      = odf.vol[ix, iy, iz, W.isort[ipeak]] -
                                           odfmin
        end
      end
    end
  end

  for ipeak in 1:npeak
    qa[ipeak].vol /= odfmax
  end

  return GQI(odf, peak, qa)
end


"""
    find_peaks!(W::GQIwork)

Find the vertices whose amplitudes are local peaks and return them sorted
(assume that ODF amplitudes have been computed)
"""
function find_peaks!(W::GQIwork)
  W.odf_peak .= W.o
  W.odf_peak[W.faces[W.o[W.faces[:,2]] .>= W.o[W.faces[:,1]] .||
                     W.o[W.faces[:,3]] .>= W.o[W.faces[:,1]], 1]] .= 0
  W.odf_peak[W.faces[W.o[W.faces[:,1]] .>= W.o[W.faces[:,2]] .||
                     W.o[W.faces[:,3]] .>= W.o[W.faces[:,2]], 2]] .= 0
  W.odf_peak[W.faces[W.o[W.faces[:,2]] .>= W.o[W.faces[:,3]] .||
                     W.o[W.faces[:,1]] .>= W.o[W.faces[:,3]], 3]] .= 0

  sortperm!(W.isort, W.odf_peak, rev=true)

  return length(findall(W.odf_peak .> 0))
end


"""
    gqi_write(gqi::GQI, basename::String)

Write the volumes from a `GQI` structure that was created by `gqi_rec()`
to files whose names start with the specified base name.
"""
function gqi_write(gqi::GQI, basename::String)

  for var in fieldnames(GQI)
    vartype = fieldtype(GQI, var)

    if vartype == MRI
      fname = basename * "_" * string(var) * ".nii.gz"
      mri_write(getfield(gqi, var), fname)
    elseif vartype == Vector{MRI}
      for ivol = 1:length(getfield(gqi, var))
        fname = basename * "_" * string(var) * string(ivol) * ".nii.gz"
        mri_write(getfield(gqi, var)[ivol], fname)
      end
    end
  end
end


