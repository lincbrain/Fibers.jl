#=
  Image display
=#

using ColorTypes, ImageView

export view


"""
    view(mri::MRI, plane::Char='a')

View an `MRI` structure in a slice viewer, along the specified `plane` ('a':
axial; 's': sagittal; 'c': coronal).
"""
function view(mri::MRI, plane::Char='a')

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

  # Convert volume to RGB/Gray array
  rgb = vol_to_rgb(mri.vol)

  # Set the size of the canvas that the volume will be displayed in
  csize = size(rgb)[[ax1, ax2]]
  csize = (300, Int.(round.(300 * csize[2] / csize[1])))

  # Display volume
  gui = ImageView.imshow(rgb, axes=(ax1,ax2), flipy=flip1, flipx=flip2,
                              canvassize=csize, name=mri.fspec)

  # Move display to middle slice
  push!(gui["roi"]["slicedata"].signals[1], Int(size(mri.vol, ax3)/2))

  # Add annotation for image axes
  fsize = round(5 * maximum(mri.volsize) / 256)

  ImageView.annotate!(gui,
            AnnotationText(size(mri.vol, ax2)*.5, size(mri.vol, ax1)*.05,
            label1[1], color=color1, fontsize=fsize))
  ImageView.annotate!(gui,
            AnnotationText(size(mri.vol, ax2)*.5, size(mri.vol, ax1)*.95,
            label1[2], color=color1, fontsize=fsize))
  ImageView.annotate!(gui,
            AnnotationText(size(mri.vol, ax2)*.05, size(mri.vol, ax1)*.5,
            label2[1], color=color2, fontsize=fsize))
  ImageView.annotate!(gui,
            AnnotationText(size(mri.vol, ax2)*.95, size(mri.vol, ax1)*.5,
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
    ImageView.annotate!(gui,
              AnnotationText(size(mri.vol, ax2)*.1, size(mri.vol, ax1)*.1,
              blabel[ib], t = ib,
              color=RGB(1,1,1), fontsize=fsize, halign="left"))
  end

  return gui
end


