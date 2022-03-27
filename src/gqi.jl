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
  peak1::MRI
  peak2::MRI
  peak3::MRI
  qa1::MRI
  qa2::MRI
  qa3::MRI
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
- `.peak1`, `.peak2`, `.peak3`: Orientation vectors of the 3 peak ODF amplitudes
- `.qa1`, `.qa2`, `.qa3`: Quantitative anisotropy for the 3 peak orientations

"""
function gqi_rec(dwi::MRI, mask::MRI, odf_dirs::ODF=sphere_642, σ::Float32=Float32(1.25))

  if isempty(dwi.bval)
    error("Missing b-value table from input DWI structure")
  end

  if isempty(dwi.bvec)
    error("Missing gradient table from input DWI structure")
  end

  # GQI reconstruction matrix
  bq_vector = dwi.bvec .* (sqrt.(dwi.bval * Float32(0.01506)) * (σ / π))
  A = sinc.(odf_dirs.vertices * bq_vector')

  peak1 = MRI(mask,3)
  peak2 = MRI(mask,3)
  peak3 = MRI(mask,3)
  qa1 = MRI(mask,1)
  qa2 = MRI(mask,1)
  qa3 = MRI(mask,1)

  odfmax = 0

  Threads.@threads for iz in 1:size(dwi.vol, 3)
    for iy in 1:size(dwi.vol, 2)
      for ix in 1:size(dwi.vol, 1)
        mask.vol[ix, iy, iz] == 0 && continue

        s = dwi.vol[ix, iy, iz, :]
        s[s .< 0] .= 0

        maximum(s) == 0 && continue

        odf = A * s

        p = find_peak(odf, odf_dirs.faces)
        odfmax = max.(odfmax, mean(odf))
        odfmin = minimum(odf)

        length(p) == 0 && continue

        peak1.vol[ix, iy, iz, :] = odf_dirs.vertices[p[1], :]
        qa1.vol[ix, iy, iz]      = odf[p[1]] - odfmin

        length(p) == 1 && continue

        peak2.vol[ix, iy, iz, :] = odf_dirs.vertices[p[2], :]
        qa2.vol[ix, iy, iz]      = odf[p[2]] - odfmin

        length(p) == 2 && continue

        peak3.vol[ix, iy, iz, :] = odf_dirs.vertices[p[3], :]
        qa3.vol[ix, iy, iz]      = odf[p[3]] - odfmin
      end
    end
  end

  qa1.vol /= odfmax
  qa2.vol /= odfmax
  qa3.vol /= odfmax

  return GQI(peak1, peak2, peak3, qa1, qa2, qa3)
end


"""
    find_peak(odf::Vector, odf_faces::Matrix)

Given the ODF amplitudes in a voxel (`odf`) and the faces of the ODF
tessellation (`odf_faces`), find the 3 vertices with peak ODF amplitudes
"""
function find_peak(odf::Vector, odf_faces::Matrix)
  faces = odf_faces - (odf_faces .> length(odf))*length(odf)

  is_peak = copy(odf)
  is_peak[faces[odf[faces[:,2]] .>= odf[faces[:,1]] .||
                odf[faces[:,3]] .>= odf[faces[:,1]], 1]] .= 0
  is_peak[faces[odf[faces[:,1]] .>= odf[faces[:,2]] .||
                odf[faces[:,3]] .>= odf[faces[:,2]], 2]] .= 0
  is_peak[faces[odf[faces[:,2]] .>= odf[faces[:,3]] .||
                odf[faces[:,1]] .>= odf[faces[:,3]], 3]] .= 0

  isort = sortperm(-is_peak)
  p = isort[-is_peak[isort] .< 0]

  return p
end


"""
    gqi_write(gqi::GQI, basename::String)

Write the volumes from a `GQI` structure that was created by `gqi_rec()`
to files whose names start with the specified base name.
"""
function gqi_write(gqi::GQI, basename::String)

  for var in fieldnames(GQI)
    mri_write(getfield(gqi, var), basename * "_" * string(var) * ".nii.gz")
  end
end


