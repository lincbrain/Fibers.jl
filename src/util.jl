#=
  Original Author: Anastasia Yendiki

  Copyright © 2022 The General Hospital Corporation (Boston, MA) "MGH"
 
  Terms and conditions for use, reproduction, distribution and contribution
  are found in the 'FreeSurfer Software License Agreement' contained
  in the file 'LICENSE' found in the FreeSurfer distribution, and here:
 
  https://surfer.nmr.mgh.harvard.edu/fswiki/FreeSurferSoftwareLicense
 
  Reporting: freesurfer@nmr.mgh.harvard.edu
=#

using DelimitedFiles, LinearAlgebra

export cart2pol, pol2cart, cart2sph, sph2cart,
       Xform, xfm_read, xfm_apply, xfm_apply!


"""
    cart2pol(x, y)

    Transform Cartesian coordinates (`x`, `y`) to polar coordinates (`φ`, `ρ`),
    where `φ` is in radians.
"""
function cart2pol(x, y)

  φ = atan.(y, x)
  ρ = hypot.(x, y)

  return φ, ρ
end


"""
    pol2cart(φ, ρ)

    Transform polar coordinates (`φ`, `ρ`) to Cartesian coordinates (`x`, `y`),
    where `φ` is in radians.
"""
function pol2cart(φ, ρ)

  x = ρ .* cos.(φ)
  y = ρ .* sin.(φ)

  return x, y
end


"""
    cart2sph(x, y, z)

    Transform Cartesian coordinates (`x`, `y`, `z`) to spherical coordinates
    (`φ`, `θ`, `ρ`), where `φ` and `θ` are in radians.
"""
function cart2sph(x, y, z)

  hypotxy = hypot.(x, y)
  ρ = hypot.(hypotxy, z)
  θ = atan.(z, hypotxy)
  φ = atan.(y, x)

  return φ, θ, ρ
end


"""
    sph2cart(φ, θ, ρ)

    Transform spherical coordinates (`φ`, `θ`, `ρ`) to Cartesian coordinates
    (`x`, `y`, `z`).

    `φ` and `θ` must be in radians.
"""
function sph2cart(φ, θ, ρ)

  z = ρ .* sin.(θ)
  ρcosθ = ρ .* cos.(θ)
  x = ρcosθ .* cos.(φ)
  y = ρcosθ .* sin.(φ)

  return x, y, z
end


"""
    ang2rot(φ, θ)

    Convert polar `φ` and azimuthal `θ` angles to a rotation matrix.

    `φ` and `θ` must be in radians.
"""
function ang2rot(φ, θ)

  # Rotate by φ around the z-axis
  Rz = [cos(φ)  -sin(φ)  0
        sin(φ)   cos(φ)  0
        0         0      1]

  # Rotate by θ around the y-axis
  Ry = [cos(θ)   0   sin(θ)
        0        1   0
       -sin(θ)   0   cos(θ)]

  R =  Rz * Ry

  return R
end


"Container for an image transform"
struct Xform{T<:Number}
  insize::Vector{Int}	# Input volume dimensions
  outsize::Vector{Int}	# Output volume dimensions
  inres::Vector{T}	# Input voxel size
  outres::Vector{T}	# Output voxel size
  invox2ras::Matrix{T}	# Transform from voxel to RAS coordinates in input vol
  outvox2ras::Matrix{T}	# Transform from voxel to RAS coordinates in output vol
  vox2vox::Matrix{T}	# Affine transform between volumes in voxel coordinates
  ras2ras::Matrix{T}	# Affine transform between volumes in RAS coordinates
  voxrot::Matrix{T}	# Rotational component of vox2vox transform
end


"""
    Xform{T}()

Return an empty `Xform` structure with data type T
"""
Xform{T}() where T<:Number = Xform{T}(
  Vector{Int}(undef, 3),
  Vector{Int}(undef, 3),
  Vector{T}(undef, 3),
  Vector{T}(undef, 3),
  Matrix{T}(undef, 4, 4),
  Matrix{T}(undef, 4, 4),
  Matrix{T}(undef, 4, 4),
  Matrix{T}(undef, 4, 4),
  Matrix{T}(undef, 3, 3)
)


"""
    xfm_read(ltafile::String, T::DataType=Float32)

Read a transform from a .lta file and return an `Xform` structure
"""
function xfm_read(ltafile::String, T::DataType=Float32)

  xfm = Xform{T}()

  io = open(ltafile, "r")

  regtype = regmat = readsrc = in_size = out_size = in_res = out_res =
            in_xras = out_xras =  in_yras = out_yras = in_zras = out_zras =
            in_cras = out_cras = nothing

  while !eof(io)
    ln = split(readline(io))

    if isempty(ln)
      continue
    elseif ln[1] == "type"			# Transform type
      regtype = parse(Int, ln[3])
    elseif ln[1] == "1" && ln[2] == "4" && ln[3] == "4"		# Transform
      # Read next 4 lines and concatenate into a matrix
      regmat = reduce(vcat, [parse.(T, split(readline(io)))' for k=1:4])
    elseif ln[1] == "src"			# Input volume info
      readsrc = true
      continue
    elseif ln[1] == "dst"			# Output volume info
      readsrc = false
      continue
    elseif ln[1] == "volume"			# Volume dimensions
      readsrc ? in_size  = parse.(T, ln[3:5]) :
                out_size = parse.(T, ln[3:5])
    elseif ln[1] == "voxelsize"			# Voxel size
      readsrc ? in_res  = parse.(T, ln[3:5]) :
                out_res = parse.(T, ln[3:5])
    elseif ln[1] == "xras"			# x-axis in RAS coordinates
      readsrc ? in_xras  = parse.(T, ln[3:5]) :
                out_xras = parse.(T, ln[3:5])
    elseif ln[1] == "yras"			# y-axis in RAS coordinates
      readsrc ? in_yras  = parse.(T, ln[3:5]) :
                out_yras = parse.(T, ln[3:5])
    elseif ln[1] == "zras"			# z-axis in RAS coordinates
      readsrc ? in_zras  = parse.(T, ln[3:5]) :
                out_zras = parse.(T, ln[3:5])
    elseif ln[1] == "cras"			# Origin in RAS coordinates
      readsrc ? in_cras  = parse.(T, ln[3:5]) :
                out_cras = parse.(T, ln[3:5])
    end
  end

  close(io)

  isnothing(regtype)  && error("Missing transform type in " * ltafile)
  isnothing(regmat)   && error("Missing transform matrix in " * ltafile)
  isnothing(in_size)  && error("Missing source dimensions in " * ltafile)
  isnothing(out_size) && error("Missing destination dimensions in " * ltafile)
  isnothing(in_res)   && error("Missing source resolution in " * ltafile)
  isnothing(out_res)  && error("Missing destination resolution in " * ltafile)
  isnothing(in_xras)  && error("Missing source x_ras in " * ltafile)
  isnothing(out_xras) && error("Missing destination x_ras in " * ltafile)
  isnothing(in_yras)  && error("Missing source y_ras in " * ltafile)
  isnothing(out_yras) && error("Missing destination y_ras in " * ltafile)
  isnothing(in_zras)  && error("Missing source z_ras in " * ltafile)
  isnothing(out_zras) && error("Missing destination z_ras in " * ltafile)
  isnothing(in_cras)  && error("Missing source c_ras in " * ltafile)
  isnothing(out_cras) && error("Missing destination c_ras in " * ltafile)

  # Set input and output volume dimensions
  xfm.insize  .= in_size
  xfm.outsize .= out_size

  # Set input and output resolutions
  xfm.inres  .= in_res
  xfm.outres .= out_res

  # Compute input and output vox2ras matrices
  in_vox2ras = hcat(in_xras * in_res[1],
                    in_yras * in_res[2],
                    in_zras * in_res[3])

  xfm.invox2ras[1:3, 1:3] .= in_vox2ras
  xfm.invox2ras[1:3, 4]   .= in_cras .- (in_vox2ras * in_size) ./ 2
  xfm.invox2ras[4, 1:3]   .= T(0)
  xfm.invox2ras[4, 4]      = T(1)

  out_vox2ras = hcat(out_xras * out_res[1],
                     out_yras * out_res[2],
                     out_zras * out_res[3])

  xfm.outvox2ras[1:3, 1:3] .= out_vox2ras
  xfm.outvox2ras[1:3, 4]   .= out_cras .- (out_vox2ras * out_size) ./ 2
  xfm.outvox2ras[4, 1:3]   .= T(0)
  xfm.outvox2ras[4, 4]      = T(1)

  # Set transform matrix
  if regtype == 0		# LINEAR_VOX_TO_VOX
    xfm.vox2vox .= regmat
    xfm.ras2ras .= xfm.outvox2ras * regmat * inv(xfm.invox2ras)
  elseif regtype == 1		# LINEAR_RAS_TO_RAS
    xfm.vox2vox .= inv(xfm.outvox2ras) * regmat * xfm.invox2ras
    xfm.ras2ras .= regmat
  else
    error("Invalid transform type " * string(regtype) * " in " * ltafile)
  end

  # Compute rotational component
  matsvd = svd(xfm.vox2vox[1:3, 1:3])
  mul!(xfm.voxrot, matsvd.U, matsvd.Vt)

  return xfm
end


"""
    xfm_read(matfile::String, inref::MRI, outref::MRI, T::DataType=Float32)

Read a transform from a .mat file and return an `Xform` structure

Reference volumes in the input and output space of the transform must also be
specified, as `MRI` structures
"""
function xfm_read(matfile::String, inref::MRI, outref::MRI, T::DataType=Float32)

  xfm = Xform{T}()

  # Set input and output volume dimensions
  xfm.insize  .= inref.volsize
  xfm.outsize .= outref.volsize

  # Set input and output resolutions
  xfm.inres  .= inref.volres
  xfm.outres .= outref.volres

  # Set input and output vox2ras matrices
  xfm.invox2ras  .= inref.vox2ras
  xfm.outvox2ras .= outref.vox2ras

  # Convert from FSL-style matrix to true vox2vox matrix
  Din  = Diagonal(vcat(inref.volres,  1))

  if det(inref.vox2ras) > 0
    Din[1, 1] *= T(-1)
    Din[1, 4] = inref.volres[1] * (inref.volsize[1] - 1)
  end

  Dout = Diagonal(vcat(outref.volres, 1))

  if det(outref.vox2ras) > 0
    Dout[1, 1] *= T(-1)
    Dout[1, 4] = outref.volres[1] * (outref.volsize[1] - 1)
  end

  xfm.vox2vox .= inv(Dout) * readdlm(matfile) * Din
  xfm.ras2ras .= outref.vox2ras * xfm.vox2vox * inv(inref.vox2ras)

  # Compute rotational component
  matsvd = svd(xfm.vox2vox[1:3, 1:3])
  mul!(xfm.voxrot, matsvd.U, matsvd.Vt)

  return xfm
end


"""
    Base.inv(xfm::Xform{T})

Invert a transform and return a new `Xform` structure
"""
function Base.inv(xfm::Xform{T}) where T<:Number

  ixfm = Xform{T}()

  ixfm.inres   .= xfm.outres
  ixfm.outres  .= xfm.inres
  ixfm.vox2vox .= inv(xfm.vox2vox)
  ixfm.ras2ras .= inv(xfm.ras2ras)
  ixfm.voxrot  .= xfm.voxrot'

  return ixfm
end


"""
    xfm_apply(xfm::Xform{T}, point::Vector{T})

Apply a transform specified in an `Xform` structure to a point specified in a
vector of length 3, and return the transformed point as a vector of length 3
"""
function xfm_apply(xfm::Xform{T}, point::Vector{T}) where T<:Number

  newpoint = Vector{T}(undef, 3)

  xfm_apply!(newpoint, xfm, point)

  return newpoint
end


"""
    xfm_apply!(outpoint::Vector{T}, xfm::Xform{T}, inpoint::Vector{T})

Apply a transform specified in an `Xform` structure to a point specified in a
vector of length 3, in place
"""
function xfm_apply!(outpoint::Vector{T}, xfm::Xform{T}, inpoint::Vector{T}) where T<:Number

  fill!(outpoint, 0)

  aff = T(0)
  for j in 1:3
    aff += xfm.vox2vox[4, j] * inpoint[j]
  end
  aff += xfm.vox2vox[4, 4]

  for i in 1:3
    for j in 1:3
      outpoint[i] += (xfm.vox2vox[i, j] * inpoint[j])
    end

    outpoint[i] += xfm.vox2vox[i, 4]
    outpoint[i] /= aff
  end
end


