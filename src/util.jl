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

export cart2pol, pol2cart, cart2sph, sph2cart, Xform, xform, xform!


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
  inres::Vector{T}	# Input voxel size
  outres::Vector{T}	# Output voxel size
  mat::Matrix{T}	# Affine transformation matrix
  rot::Matrix{T}	# Rotational component
end


"""
    Xform{T}()

Return an empty `Xform` structure with data type T
"""
Xform{T}() where T<:Number = Xform{T}(
  Vector{T}(undef, 3),
  Vector{T}(undef, 3),
  Matrix{T}(undef, 4, 4),
  Matrix{T}(undef, 3, 3)
)


"""
    Xform{T}(matfile::String, inres::Vector, outres::Vector)

Read a transform from a .mat file and return an `Xform` structure

The voxel sizes of the input and output space of the transform must also be
specified, as vectors of length 3
"""
function Xform{T}(matfile::String, inres::Vector, outres::Vector) where T<:Number

  xfm = Xform{T}()

  xfm.inres  .= inres
  xfm.outres .= outres
  xfm.mat    .= readdlm(matfile)

  # Compute rotational component
  matsvd = svd(xfm.mat[1:3, 1:3])
  mul!(xfm.rot, matsvd.U, matsvd.Vt)

  return xfm
end


"""
    Base.inv(xfm::Xform{T})

Invert a transform and return a new `Xform` structure
"""
function Base.inv(xfm::Xform{T}) where T<:Number

  ixfm = Xform{T}()

  ixfm.inres  .= xfm.outres
  ixfm.outres .= xfm.inres
  ixfm.mat    .= inv(xfm.mat)
  ixfm.rot    .= xfm.rot'

  return ixfm
end


"""
    xform(xfm::Xform{T}, point::Vector{T})

Apply a transform specified in an `Xform` structure to a point specified in a
vector of length 3, and return the transformed point as a vector of length 3
"""
function xform(xfm::Xform{T}, point::Vector{T}) where T<:Number

  newpoint = Vector{T}(undef, 3)

  xform!(newpoint, xfm, point)

  return newpoint
end


"""
    xform!(outpoint::Vector{T}, xfm::Xform{T}, inpoint::Vector{T})

Apply a transform specified in an `Xform` structure to a point specified in a
vector of length 3, in place
"""
function xform!(outpoint::Vector{T}, xfm::Xform{T}, inpoint::Vector{T}) where T<:Number

  fill!(outpoint, 0)

  aff = 0

  for j in 1:3
    coord = xfm.inres[j] * inpoint[j]

    for i in 1:3
      outpoint[i] += (xfm.mat[i, j] * coord)
    end

    aff += xfm.mat[4, j] * coord
  end

  aff += xfm.mat[4, 4]
  
  for i in 1:3
    outpoint[i] += xfm.mat[i, 4]
    outpoint[i] /= (xfm.outres[i] * aff)
  end
end


