#=
  Original Author: Anastasia Yendiki

  Copyright © 2022 The General Hospital Corporation (Boston, MA) "MGH"
 
  Terms and conditions for use, reproduction, distribution and contribution
  are found in the 'FreeSurfer Software License Agreement' contained
  in the file 'LICENSE' found in the FreeSurfer distribution, and here:
 
  https://surfer.nmr.mgh.harvard.edu/fswiki/FreeSurferSoftwareLicense
 
  Reporting: freesurfer@nmr.mgh.harvard.edu
=#
 
using LinearAlgebra

export Tract, str_add!, trk_read, trk_write


"Container for header and streamline data stored in .trk format"
mutable struct Tract{T<:Number}
  # Header fields (.trk format version 2)
  id_string::Vector{UInt8}
  dim::Vector{Int16}
  voxel_size::Vector{Float32}
  origin::Vector{Float32}
  n_scalars::Int16
  scalar_name::Matrix{UInt8}
  n_properties::Int16
  property_name::Matrix{UInt8}
  vox_to_ras::Matrix{Float32}
  reserved::Vector{UInt8}
  voxel_order::Vector{UInt8}
  voxel_order_original::Vector{UInt8}
  image_orientation_patient::Vector{Float32}
  pad1::Vector{UInt8}
  invert_x::UInt8
  invert_y::UInt8
  invert_z::UInt8
  swap_xy::UInt8
  swap_yz::UInt8
  swap_zx::UInt8
  n_count::Int32
  version::Int32
  hdr_size::Int32

  # Streamline data fields
  npts::Vector{Int32}
  properties::Matrix{T}
  xyz::Vector{Matrix{T}}
  scalars::Vector{Matrix{T}}
end


"""
    Tract{T}()

Return an empty `Tract` structure with data type T
"""
Tract{T}() where T<:Number = Tract{T}(
  Vector{UInt8}(undef, 0),
  Vector{Int16}(undef, 0),
  Vector{Float32}(undef, 0),
  Vector{Float32}(undef, 0),
  Int16(0),
  Matrix{UInt8}(undef, 0, 0),
  Int16(0),
  Matrix{UInt8}(undef, 0, 0),
  Matrix{Float32}(undef, 0, 0),
  Vector{UInt8}(undef, 0),
  Vector{UInt8}(undef, 0),
  Vector{UInt8}(undef, 0),
  Vector{Float32}(undef, 0),
  Vector{UInt8}(undef, 0),
  UInt8(0),
  UInt8(0),
  UInt8(0),
  UInt8(0),
  UInt8(0),
  UInt8(0),
  Int32(0),
  Int32(0),
  Int32(0),

  Vector{Int32}(undef, 0),
  Matrix{T}(undef, 0, 0),
  Vector{Matrix{T}}(undef, 0),
  Vector{Matrix{T}}(undef, 0)
)


"""
    Tract{T}(ref::MRI) where T<:Number

Return a `Tract` structure with data type T and header fields populated based
on the reference `MRI` structure `ref`
"""
function Tract{T}(ref::MRI) where T<:Number

  tr = Tract{T}()

  # In the following I must take into account that mri_read()
  # may have permuted the first 2 dimensions of volsize and volres,
  # but it has not changed the vox2ras matrix

  # Find orientation of image coordinate system
  orient = vox2ras_to_orient(ref.vox2ras)

  # Find patient-to-scanner coordinate transform:
  # Take x and y vectors from vox2RAS matrix, convert to LPS,
  # divide by voxel size
  if ref.ispermuted
    p2s = [-1 0 0; 0 -1 0; 0 0 1] * ref.vox2ras[1:3, 1:2] * 
                                    Diagonal([1, 1]./ref.volres[[2,1]])
  else
    p2s = [-1 0 0; 0 -1 0; 0 0 1] * ref.vox2ras[1:3, 1:2] * 
                                    Diagonal([1, 1]./ref.volres[1:2])
  end

  tr.id_string     = UInt8.(collect("TRACK\0"))

  if ref.ispermuted
    tr.dim         = Int16.(ref.volsize[[2,1,3]])
    tr.voxel_size  = Float32.(ref.volres[[2,1,3]])
  else
    tr.dim         = Int16.(ref.volsize)
    tr.voxel_size  = Float32.(ref.volres)
  end
  tr.origin        = Float32.(zeros(3))

  tr.n_scalars     = Int16(0)
  tr.scalar_name   = UInt8.(zeros(10, 20))
  tr.n_properties  = Int16(0)
  tr.property_name = UInt8.(zeros(10, 20))

  tr.vox_to_ras    = ref.vox2ras
  tr.reserved      = UInt8.(zeros(444))
  tr.voxel_order               = vcat(UInt8.(collect(orient)), UInt8(0))
  tr.voxel_order_original      = tr.voxel_order
  tr.image_orientation_patient = Float32.(p2s[:])
  tr.pad1          = UInt8.(zeros(2))

  tr.invert_x      = UInt8(0)
  tr.invert_y      = UInt8(0)
  tr.invert_z      = UInt8(0)
  tr.swap_xy       = UInt8(0)
  tr.swap_yz       = UInt8(0)
  tr.swap_zx       = UInt8(0)
  tr.n_count       = Int32(0)
  tr.version       = Int32(2)
  tr.hdr_size      = Int32(1000)

  return tr
end


"""
     str_add!(tr::Tract, xyz::Vector{Matrix{T}}, scalars::Union{Vector{Matrix{T}}, Nothing}=nothing, properties::Union{Matrix{T}, Nothing}=nothing) where T<:Number

Append new streamlines of data type T to a Tract structure

# Required inputs
- tr::Tract:
  Tract structure that the streamlines will be added to
- xyz::Vector{Matrix{T}}:
  Voxel coordinates of the points on the new streamlines [nstr][3 x npts]

# Optional inputs (required only if Tract structure contains them)
- scalars::Vector{Matrix{T}:
  Scalars associated with each point on the new streamlines
  [nstr][nscalar x npts]
- properties::Matrix{T}:
  Properties associated with each of the new streamlines [nprop x nstr]
"""
function str_add!(tr::Tract, xyz::Vector{Matrix{T}}, scalars::Union{Vector{Matrix{T}}, Nothing}=nothing, properties::Union{Matrix{T}, Nothing}=nothing) where T<:Number

  # Data type in destination Tract() structure
  Tout = eltype(tr.properties)

  # Add to number of streamlines
  tr.n_count += Int32(length(xyz))

  # Append numbers of points per streamline
  push!(tr.npts, Int32.(size.(xyz, 2))...)

  # Append streamline coordinates
  push!(tr.xyz, map(x -> Tout.(x), xyz)...)

  # Append scalars associated with each point on each streamline (if any)
  if (isnothing(scalars) && tr.n_scalars != 0) ||
    (!isnothing(scalars) && any(size.(scalars, 1) .!= tr.n_scalars))
    error("Must have " * string(tr.n_scalars) *
          " input scalars per point to append to Tract structure")
  end

  if isnothing(scalars)
    push!(tr.scalars, map(x -> Matrix{Tout}(undef, 0, x), size.(xyz, 2))...)
  else
    push!(tr.scalars, map(x -> Tout.(x), scalars)...)
  end

  # Append properties associated with each streamline (if any)
  if (isnothing(properties) && tr.n_properties != 0) ||
    (!isnothing(properties) && size(properties, 1) .!= tr.n_properties)
    error("Must have " * string(tr.n_properties) *
          " input properties per streamline to append to Tract structure")
  end

  if isnothing(properties)
    tr.properties = hcat(tr.properties, Matrix{Tout}(undef, 0, length(xyz)))
  else
    tr.properties = hcat(tr.properties, Tout.(properties))
  end
end


"""
    trk_read(infile::String)

Read tractography streamlines from `infile` and return a `Tract` structure

Input file must be in .trk format, see:
http://trackvis.org/docs/?subsect=fileformat
"""
function trk_read(infile::String)

  T = Float32			# Data type in .trk files is always Float32

  tr = Tract{T}()

  io = open(infile, "r")

  # Read .trk header
  tr.id_string     = read!(io, Vector{UInt8}(undef, 6))

  tr.dim           = read!(io, Vector{Int16}(undef, 3))
  tr.voxel_size    = read!(io, Vector{Float32}(undef, 3))
  tr.origin        = read!(io, Vector{Float32}(undef, 3))

  tr.n_scalars     = read(io, Int16)
  tr.scalar_name   = read!(io, Matrix{UInt8}(undef, 20, 10))
  tr.scalar_name   = permutedims(tr.scalar_name, [2,1])

  tr.n_properties  = read(io, Int16)
  tr.property_name = read!(io, Matrix{UInt8}(undef, 20, 10))
  tr.property_name = permutedims(tr.property_name, [2,1])

  tr.vox_to_ras    = read!(io, Matrix{Float32}(undef, 4, 4))
  tr.vox_to_ras    = permutedims(tr.vox_to_ras, [2,1])
  tr.reserved      = read!(io, Vector{UInt8}(undef, 444))
  tr.voxel_order               = read!(io, Vector{UInt8}(undef, 4))
  tr.voxel_order_original      = read!(io, Vector{UInt8}(undef, 4))
  tr.image_orientation_patient = read!(io, Vector{Float32}(undef, 6))
  tr.pad1          = read!(io, Vector{UInt8}(undef, 2))

  tr.invert_x      = read(io, UInt8)
  tr.invert_y      = read(io, UInt8)
  tr.invert_z      = read(io, UInt8)
  tr.swap_xy       = read(io, UInt8)
  tr.swap_yz       = read(io, UInt8)
  tr.swap_zx       = read(io, UInt8)
  tr.n_count       = read(io, Int32)
  tr.version       = read(io, Int32)
  tr.hdr_size      = read(io, Int32)

  # Read streamline data
  tr.npts       = Vector{Int32}(undef, tr.n_count)
  tr.properties = Matrix{T}(undef, tr.n_properties, tr.n_count)

  for istr in 1:tr.n_count		# Loop over streamlines
    tr.npts[istr] = read(io, Int32)

    push!(tr.xyz, Matrix{T}(undef, 3, tr.npts[istr]))
    push!(tr.scalars, Matrix{T}(undef, tr.n_scalars, tr.npts[istr]))

    for ipt in 1:tr.npts[istr]		# Loop over points on a streamline
      # Divide by voxel size and make voxel coordinates 0-based
      tr.xyz[istr][:, ipt] = read!(io, Vector{T}(undef, 3)) ./
                             tr.voxel_size .- .5

      tr.scalars[istr][:, ipt] = read!(io, Vector{T}(undef, tr.n_scalars))
    end

    tr.properties[:, istr] = read!(io, Vector{T}(undef, tr.n_properties))
  end

  close(io)

  return tr
end

"""
     trk_write(tr::Tract, outfile::String)

Write a `Tract` structure to a file in the .trk format

Return true if an error occurred (i.e., the number of bytes written was not the
expected based on the size of the `Tract` structure)
"""
function trk_write(tr::Tract, outfile::String)

  T = Float32			# Data type in .trk files is always Float32

  io = open(outfile, "w")

  nb = 0

  # Write .trk header
  nb += write(io, UInt8.(tr.id_string))

  nb += write(io, Int16.(tr.dim))
  nb += write(io, Float32.(tr.voxel_size))
  nb += write(io, Float32.(tr.origin))

  nb += write(io, Int16(tr.n_scalars))
  nb += write(io, UInt8.(permutedims(tr.scalar_name, [2,1])))
  nb += write(io, Int16(tr.n_properties))
  nb += write(io, UInt8.(permutedims(tr.property_name, [2,1])))

  nb += write(io, Float32.(permutedims(tr.vox_to_ras, [2,1])))
  nb += write(io, UInt8.(tr.reserved))
  nb += write(io, UInt8.(tr.voxel_order))
  nb += write(io, UInt8.(tr.voxel_order_original))
  nb += write(io, Float32.(tr.image_orientation_patient))
  nb += write(io, UInt8.(tr.pad1))

  nb += write(io, UInt8(tr.invert_x))
  nb += write(io, UInt8(tr.invert_y))
  nb += write(io, UInt8(tr.invert_z))
  nb += write(io, UInt8(tr.swap_xy))
  nb += write(io, UInt8(tr.swap_yz))
  nb += write(io, UInt8(tr.swap_zx))
  nb += write(io, Int32(tr.n_count))
  nb += write(io, Int32(tr.version))
  nb += write(io, Int32(tr.hdr_size))

  # Write streamline data
  for istr in 1:tr.n_count		# Loop over streamlines
    nb += write(io, Int32(tr.npts[istr]))

    for ipt in 1:tr.npts[istr]		# Loop over points on a streamline
      # Make voxel coordinates .5-based and multiply by voxel size
      nb += write(io, T.((tr.xyz[istr][:, ipt] .+ .5) .* tr.voxel_size))

      nb += write(io, T.(tr.scalars[istr][:, ipt]))
    end

    nb += write(io, T.(tr.properties[:, istr]))
  end

  close(io)

  err = (nb != sizeof(UInt8) * 866 +
               sizeof(Int32) * (3 + length(tr.npts)) +
               sizeof(Int16) * 5 +
               sizeof(Float32) * 28 +
               sizeof(T) * (sum(length.(tr.xyz)) +
                            sum(length.(tr.scalars)) +
                            length(tr.properties)))

  return err
end

