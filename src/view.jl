#=
  Original Author: Anastasia Yendiki

  Copyright © 2022 The General Hospital Corporation (Boston, MA) "MGH"

  Terms and conditions for use, reproduction, distribution and contribution
  are found in the 'FreeSurfer Software License Agreement' contained
  in the file 'LICENSE' found in the FreeSurfer distribution, and here:

  https://surfer.nmr.mgh.harvard.edu/fswiki/FreeSurferSoftwareLicense

  Reporting: freesurfer@nmr.mgh.harvard.edu
=#

using ColorTypes, DelimitedFiles, ImageInTerminal, ImageView

export LUT, color_lut, show, view

const julia_red    = RGB(.796, .235, .200)
const julia_blue   = RGB(.251, .388, .847)
const julia_green  = RGB(.220, .596, .149)
const julia_purple = RGB(.584, .345, .698)


"Container for segmentation and tract look-up tables"
mutable struct LUT
  id::Vector{Int}
  name::Vector{String}
  rgb::Vector{RGB}
end


"""
    LUT()

Return an empty `LUT` structure
"""
LUT() = LUT(
  Vector{Int}(undef, 0),
  Vector{String}(undef, 0),
  Vector{RGB}(undef, 0)
)


"""
    LUT(infile::String)

Read a look-up table from `infile` and return a `LUT` structure

The input file is assumed to have the format of FreeSurferColorLUT.txt
"""
function LUT(infile::String)

  lut = LUT()

  if !isfile(infile)
    error(infile * "is not a regular file")
  end

  tab = readdlm(infile; comments=true, comment_char='#')

  lut.id   = tab[:,1]
  lut.name = tab[:,2]
  lut.rgb  = RGB.(tab[:,3]/255, tab[:,4]/255, tab[:,5]/255)

  return lut
end


"The FreeSurfer color look-up table"
const global color_lut = haskey(ENV, "FREESURFER_HOME") ?
                         LUT(ENV["FREESURFER_HOME"]*"/FreeSurferColorLUT.txt") :
                         LUT()


"""
    vol_to_rgb(vol::Array)

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
  is converted to grayscale.

Return an array of RGB/Gray values the same size as the input vol.
"""
function vol_to_rgb(vol::Array)

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
  return Gray.(vol / maximum(vol))
end


"""
    show(mri::MRI, mrimod::Union{MRI, Nothing}=nothing)

Show an `MRI` structure (an image slice and a summary of header info) in the
terminal window

# Arguments:
- `mri::MRI`: the main image to display
- `mrimod:MRI`: an optional image to modulate the main image by (e.g., an FA
  map to modulate a vector map)
"""
function show(mri::MRI, mrimod::Union{MRI, Nothing}=nothing)

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
    view(mri::MRI, plane::Char='a')

View an `MRI` structure in a slice viewer, along the specified plane ('a':
axial; 's': sagittal; 'c': coronal).
"""
function view(mri::MRI, plane::Char='a')

  if all(plane .!= ['a', 's', 'c'])
    error("Valid viewing planes are: " * string(('a', 's', 'c')))
  end

  # Find orientation of image coordinate system
  orient = vox2ras_to_orient(mri.vox2ras)

  # Find which axes of the volume correspond to the specified viewing plane
  if plane == 'a'			# Axial plane: A->P, R->L
    ax1 = findfirst(orient .== 'A' .|| orient .== 'P')
    flip1 = (orient[ax1] == 'A')
    label1 = ["A", "P"]
    color1 = julia_green

    ax2 = findfirst(orient .== 'R' .|| orient .== 'L')
    flip2 = (orient[ax1] == 'R')
    label2 = ["R", "L"]
    color2 = julia_red
  elseif plane == 's'			# Sagittal plane: S->I, P->A
    ax1 = findfirst(orient .== 'S' .|| orient .== 'I')
    flip1 = (orient[ax1] == 'S')
    label1 = ["S", "I"]
    color1 = julia_blue

    ax2 = findfirst(orient .== 'A' .|| orient .== 'P')
    flip2 = (orient[ax1] == 'P')
    label2 = ["P", "A"]
    color2 = julia_green
  else					# Coronal plane: S->I, R->L
    ax1 = findfirst(orient .== 'S' .|| orient .== 'I')
    flip1 = (orient[ax1] == 'S')
    label1 = ["S", "I"]
    color1 = julia_blue

    ax2 = findfirst(orient .== 'R' .|| orient .== 'L')
    flip2 = (orient[ax1] == 'R')
    label2 = ["R", "L"]
    color2 = julia_red
  end

  if mri.ispermuted
    (ax1 == 1) && (ax1 = 2)
    (ax1 == 2) && (ax1 = 1)
    (ax2 == 1) && (ax2 = 2)
    (ax2 == 2) && (ax2 = 1)
  end

  # Find the through-plane axis of the volume
  ax3 = setdiff(1:3, [ax1, ax2])[1]

  # Convert volume to RGB/Gray array
  rgb = vol_to_rgb(mri.vol)

  # Display volume
  gui = ImageView.imshow(rgb, axes=(ax1,ax2), flipy=flip1, flipx=flip2)

  # Move display to middle slice
  push!(gui["roi"]["slicedata"].signals[1], Int(size(mri.vol, ax3)/2))

  # Add annotation for image axes
  fsize = round(5 * maximum(mri.volsize) / 256)

  annotate!(gui, AnnotationText(size(mri.vol, ax2)*.5, size(mri.vol, ax1)*.05,
            label1[1], color=color1, fontsize=fsize))
  annotate!(gui, AnnotationText(size(mri.vol, ax2)*.5, size(mri.vol, ax1)*.95,
            label1[2], color=color1, fontsize=fsize))
  annotate!(gui, AnnotationText(size(mri.vol, ax2)*.05, size(mri.vol, ax1)*.5,
            label2[1], color=color2, fontsize=fsize))
  annotate!(gui, AnnotationText(size(mri.vol, ax2)*.95, size(mri.vol, ax1)*.5,
            label2[2], color=color2, fontsize=fsize))

  # Add annotation for b-value and gradient vector
  blabel = Vector{String}(undef, 0)

  if !isempty(mri.bval)
    blabel = "b=" .* string.(Int.(round.(mri.bval)))
  end

  if !isempty(mri.bvec)
    blabel = blabel .* "\ng=[" .*
                       string.(round.(mri.bvec[:,1] * 100)/100) .* "," .*
                       string.(round.(mri.bvec[:,2] * 100)/100) .* "," .*
                       string.(round.(mri.bvec[:,3] * 100)/100) .* "]"
  end

  for ib = 1:length(blabel)
    annotate!(gui, AnnotationText(size(mri.vol, ax2)*.1, size(mri.vol, ax1)*.1,
              blabel[ib], t = ib,
              color=RGB(1,1,1), fontsize=fsize, halign="left"))
  end
end

