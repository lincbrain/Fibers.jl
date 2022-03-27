#=
  Original Author: Anastasia Yendiki

  Copyright © 2022 The General Hospital Corporation (Boston, MA) "MGH"
 
  Terms and conditions for use, reproduction, distribution and contribution
  are found in the 'FreeSurfer Software License Agreement' contained
  in the file 'LICENSE' found in the FreeSurfer distribution, and here:
 
  https://surfer.nmr.mgh.harvard.edu/fswiki/FreeSurferSoftwareLicense
 
  Reporting: freesurfer@nmr.mgh.harvard.edu
=#

export cart2pol, pol2cart, cart2sph, sph2cart


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


