#=
  Original Author: Anastasia Yendiki

  Copyright © 2022 The General Hospital Corporation (Boston, MA) "MGH"

  Terms and conditions for use, reproduction, distribution and contribution
  are found in the 'FreeSurfer Software License Agreement' contained
  in the file 'LICENSE' found in the FreeSurfer distribution, and here:

  https://surfer.nmr.mgh.harvard.edu/fswiki/FreeSurferSoftwareLicense

  Reporting: freesurfer@nmr.mgh.harvard.edu
=#

using ColorTypes, DelimitedFiles, Statistics, ImageInTerminal, Plots

export LUT, color_lut, info, disp, show

const julia_red    = RGB(.796, .235, .200)
const julia_blue   = RGB(.251, .388, .847)
const julia_green  = RGB(.220, .596, .149)
const julia_purple = RGB(.584, .345, .698)


"Container for segmentation and tract look-up tables"
struct LUT
  id::Vector{Int}
  name::Vector{String}
  rgb::Vector{RGB}

  function LUT(infile::String)

    if !isfile(infile)
      error(infile * "is not a regular file")
    end

    # Read a look-up table from an input file
    # (assumed to have the format of FreeSurferColorLUT.txt)
    tab = readdlm(infile; comments=true, comment_char='#')

    # Label IDs
    id   = tab[:,1]

    # Label names
    name = tab[:,2]

    # Label display colors
    rgb  = RGB.(tab[:,3]/255, tab[:,4]/255, tab[:,5]/255)

    new(
      id,
      name,
      rgb
    )
  end
end


"The FreeSurfer color look-up table"
const global color_lut = LUT(pkgdir(FreeSurfer) * "/src/FreeSurferColorLUT.txt")


"""
    vol_to_rgb(vol::Array, maxint::Union{Number, Nothing}=nothing)

Convert an image array to an RGB/Gray array for display.

Determine how the image should be displayed:
- If all the values are IDs in the FreeSurfer color look-up table,
  the image is assumed to be is a segmentation map and is converted
  to RGB based on the FreeSurfer color look-up table.
- If the image has size 3 in any dimension, and the sum of squares
  of the values in that dimension is approx. 1 everywhere, the image
  is assumed to be a vector map and is converted to RGB based on
  vector orientation.
- Otherwise, the image is assumed to be a generic intensity map and
  is converted to grayscale, optionally clamping intensities above `maxint`.

Return an array of RGB/Gray values the same size as the input vol.
"""
function vol_to_rgb(vol::Array, maxint::Union{Number, Nothing}=nothing)

  if !any(isnothing.(indexin(unique(vol), color_lut.id)))
    if isempty(color_lut.id)
      error("FreeSurfer color look-up table is undefined")
    end

    # Assume the input is a segmentation map, get RGB of labels from LUT
    return color_lut.rgb[indexin(vol, color_lut.id)]
  end

  dim3 = findall(size(vol) .== 3)

  for idim in dim3
    if all(isapprox.(sum(vol.^2; dims=idim), 1) .|| all(vol.==0, dims=idim))
      # Assume the input is a vector map, get RGB based on vector orientation
      return RGB.(abs.(selectdim(vol,idim,1)),
                  abs.(selectdim(vol,idim,2)),
                  abs.(selectdim(vol,idim,3)))
    end
  end

  # Otherwise, assume the input is a generic intensity map
  if isnothing(maxint)
    return Gray.(vol / maximum(vol))
  else
    return Gray.(min.(vol, maxint) / maxint)
  end
end


"""
    info(mri::MRI)

Show basic info from the header of an `MRI` structure

"""
function info(mri::MRI)

  if !isempty(mri.fspec)
    println("Read from: " * mri.fspec)
  end
  println("Volume dimensions: " * string(collect(size(mri.vol))))
  println("Spatial resolution: " * string(Float64.(mri.volres)))
  if !isempty(mri.bval)
    println("b-values: " * string(Float64.(unique(mri.bval))))
  end
  println("Intensity range: " * string(Float64.([minimum(mri.vol),
                                                 maximum(mri.vol)])))

end


"""
    disp(mri::MRI, mrimod::Union{MRI, Nothing}=nothing)

Quick display of an `MRI` structure (an image slice and a summary of header
info) in the terminal window

# Arguments:
- `mri::MRI`: the main image to display
- `mrimod:MRI`: an optional image to modulate the main image by (e.g., an FA
  map to modulate a vector map)
"""
function disp(mri::MRI, mrimod::Union{MRI, Nothing}=nothing)

  # Find non-empty slices in z dimension
  iz = any(mri.vol .!= 0; dims=([1:2; 4:ndims(mri.vol)]))
  iz = reshape(iz, length(iz))
  iz = findall(iz)

  # Keep only the middle non-empty slice in z dimension
  iz = iz[Int(round(end/2))]

  # Find non-empty slices in x, y dimensions
  iy = any(mri.vol[:,:,iz,:] .!= 0; dims=([1; 3:ndims(mri.vol)]))
  iy = reshape(iy, length(iy))
  iy = findall(iy)

  ix = any(mri.vol[:,iy,iz,:] .!= 0; dims=(2:ndims(mri.vol)))
  ix = reshape(ix, length(ix))
  ix = findall(ix)

  # Keep only the area containing non-empty slices in x, y dimensions
  ix = ix[1]:ix[end]
  iy = iy[1]:iy[end]

  # Subsample to fit display in x, y dimensions
  if mri.ispermuted
    nsub = Int(ceil(length(iy) ./ displaysize(stdout)[2]))
  else
    nsub = Int(ceil(length(ix) ./ displaysize(stdout)[2]))
  end

  ix = ix[1:nsub:end]
  iy = iy[1:nsub:end]

  # Convert image to RGB/Gray array
  rgb = vol_to_rgb(mri.vol[ix, iy, iz, :])

  # Keep first frame only
  rgb = rgb[:, :, 1]

  # Optional intensity modulation
  if !isnothing(mrimod)
    if size(mrimod.vol)[1:3] != size(mri.vol)[1:3]
      error("Dimension mismatch between main image " * 
            string(size(mri.vol)[1:3]) *
            " and modulation image " *
            string(size(mrimod.vol)[1:3]))
    end

    rgbmod = mrimod.vol[ix, iy, iz, 1] / maximum(mrimod.vol)
  end

  if hasfield(eltype(rgb), :r)		# RGB
    # Add α channel to make zero voxels transparent
    rgb = RGBA.(getfield.(rgb, :r), getfield.(rgb, :g), getfield.(rgb, :b),
                Float64.(rgb .!= RGB(0,0,0)))

    # Modulate intensity with (optional) second image
    if !isnothing(mrimod)
      rgb = RGBA.(getfield.(rgb, :r) .* rgbmod,
                  getfield.(rgb, :g) .* rgbmod,
                  getfield.(rgb, :b) .* rgbmod,
                  getfield.(rgb, :alpha))
    end
  else					# Gray
    # Add α channel to make zero voxels transparent
    rgb = GrayA.(getfield.(rgb, :val), Float64.(rgb .!= Gray(0)))

    # Modulate intensity with (optional) second image
    if !isnothing(mrimod)
      rgb = GrayA.(getfield.(rgb, :val) .* rgbmod,
                   getfield.(rgb, :alpha))
    end
  end

  # Display image
  if mri.ispermuted
    ImageInTerminal.imshow(rgb)
  else
    ImageInTerminal.imshow(permutedims(rgb, [2; 1]))
  end

  # Print image info
  println()
  info(mri)
end


"""
    view_axes(vox2ras::Matrix, plane::Char)

Given the `vox2ras` matrix of an image volume, return the axes along which the
volume has to be displayed to view the specified `plane` ('a': axial; 's': sagittal;
'c': coronal).
"""
function view_axes(vox2ras::Matrix, plane::Char)

  # Find orientation of image coordinate system
  orient = vox2ras_to_orient(vox2ras)

  # Find which axes of the volume correspond to the specified viewing plane
  if plane == 'a'			# Axial plane: A->P, R->L
    ax1 = findfirst(orient .== 'A' .|| orient .== 'P')
    (orient[ax1] == 'A') && (ax1 = -ax1)

    ax2 = findfirst(orient .== 'R' .|| orient .== 'L')
    (orient[ax2] == 'R') && (ax2 = -ax2)
  elseif plane == 's'			# Sagittal plane: S->I, P->A
    ax1 = findfirst(orient .== 'S' .|| orient .== 'I')
    (orient[ax1] == 'S') && (ax1 = -ax1)

    ax2 = findfirst(orient .== 'A' .|| orient .== 'P')
    (orient[ax2] == 'P') && (ax2 = -ax2)
  elseif plane == 'c'			# Coronal plane: S->I, R->L
    ax1 = findfirst(orient .== 'S' .|| orient .== 'I')
    (orient[ax1] == 'S') && (ax1 = -ax1)

    ax2 = findfirst(orient .== 'R' .|| orient .== 'L')
    (orient[ax2] == 'R') && (ax2 = -ax2)
  else
    error("Valid viewing planes are: " * string(('a', 's', 'c')))
  end

  return [ax1 ax2]
end


"""
    Base.show(mri::MRI; plane::Char='a', z::Union{Int64, Nothing}=nothing, t::Union{Int64, Nothing}=nothing, title::Union{String, Nothing}=nothing)

Show the `z`-th slice from the `t`-th frame of an `MRI`
structure, along the specified `plane` ('a': axial; 's': sagittal; 'c':
coronal).
"""
function Base.show(mri::MRI; plane::Char='a', z::Union{Int64, Nothing}=nothing, t::Union{Int64, Nothing}=nothing, title::Union{String, Nothing}=nothing, vec::Bool=true)

  # Find which axes of the volume correspond to the specified viewing plane
  ax = view_axes(mri.vox2ras, plane)

  (ax1, ax2) = abs.(ax)
  (flip1, flip2) = (ax .< 0)

  if mri.ispermuted
    (ax1 == 1) && (ax1 = 2)
    (ax1 == 2) && (ax1 = 1)
    (ax2 == 1) && (ax2 = 2)
    (ax2 == 2) && (ax2 = 1)
  end

  # Find the through-plane axis of the volume
  ax3 = setdiff(1:3, [ax1, ax2])[1]

  nx = size(mri.vol, ax2)
  ny = size(mri.vol, ax1)
  nz = size(mri.vol, ax3)

  # Set names and colors of axis labels
  if plane == 'a'			# Axial plane: A->P, R->L
    label1 = ["A", "P"]
    color1 = julia_green

    label2 = ["R", "L"]
    color2 = julia_red
  elseif plane == 's'			# Sagittal plane: S->I, P->A
    label1 = ["S", "I"]
    color1 = julia_blue

    label2 = ["P", "A"]
    color2 = julia_green
  else					# Coronal plane: S->I, R->L
    label1 = ["S", "I"]
    color1 = julia_blue

    label2 = ["R", "L"]
    color2 = julia_red
  end

  # Extract slice of interest 
  isnothing(z) && (z = div(nz, 2))

  imslice = selectdim(mri.vol, ax3, z)

  # Extract volume of interest 
  if isnothing(t)
    t = 1
    if size(imslice, 3) == 3
      imslice = imslice[:, :, 1:3]
    else
      imslice = imslice[:, :, 1]
    end
  else
    imslice = imslice[:, :, t]
  end

  # Max intensity for display (only has effect on grayscale intensity maps)
  maxint = Float32(1)

  if mri.nframes < mri.depth
    maxint = quantile(mri.vol[mri.vol .> 0], .999)
  else				# For larger volumes, only use middle slice
    imtmp = selectdim(mri.vol, ax3,  div(nz, 2))
    maxint = quantile(imtmp[imtmp .> 0], .999)
  end

  # Convert to RGB/Gray array
  rgb = vol_to_rgb(imslice, maxint)

  # Permute in-plane axes if needed
  if ax1 > ax2
    rgb = permutedims(rgb, [2;1])
  end

  # Flip in-plane axes if needed
  rgb = rgb[flip1 ? (end:-1:1) : (1:end), flip2 ? (end:-1:1) : (1:end)]

  # Display volume
  isnothing(title) && (title = basename(mri.fspec))

  p = Plots.plot(rgb, showaxis=false, ticks=[], aspect_ratio=1, title=title)

  # Add annotation for image axes
  Plots.annotate!(nx * .5, ny *.02, (label1[1], color1, 10), :top)
  Plots.annotate!(nx * .5, ny *.98, (label1[2], color1, 10), :bottom)
  Plots.annotate!(nx * .02, ny *.5, (label2[1], color2, 10), :left)
  Plots.annotate!(nx * .98, ny *.5, (label2[2], color2, 10), :right)

  # Add annotation for b-value and gradient vector
  blabel = ""

  if !isempty(mri.bval)
    blabel = "b=" * string(Int(round(mri.bval[t])))
  end

  if !isempty(mri.bvec)
    blabel = blabel * "\ng=[" *
                      string(round(mri.bvec[t,1] * 100)/100) * "," *
                      string(round(mri.bvec[t,2] * 100)/100) * "," *
                      string(round(mri.bvec[t,3] * 100)/100) * "]"
  end

  if !isempty(blabel)
    Plots.annotate!(nx * .02, ny * .02, (blabel, RGB(1,1,1), 9, :left, :top))
  end

  return p
end


