#=
  Original Author: Anastasia Yendiki

  Copyright © 2022 The General Hospital Corporation (Boston, MA) "MGH"
 
  Terms and conditions for use, reproduction, distribution and contribution
  are found in the 'FreeSurfer Software License Agreement' contained
  in the file 'LICENSE' found in the FreeSurfer distribution, and here:
 
  https://surfer.nmr.mgh.harvard.edu/fswiki/FreeSurferSoftwareLicense
 
  Reporting: freesurfer@nmr.mgh.harvard.edu
=#
 
using LinearAlgebra, Distributions, OffsetArrays

export StreamWork, stream, stream_new_line,
       stream_new_point!, stream_micro_new_point!

"""
    StreamWork{T}

Pre-allocated workspace for streamline tractography

- `T::DataType`      : Data type (default: `Float32`)
- `len_min::Int`     : Minimum streamline length (default: 3)
- `len_max::Int`     : Maximum streamline length (default: max(nx,ny,nz))
- `cosang_thresh::T` : Cosine of maximum bending angle (default: cosd(45))
- `step_size::T`     : Step length, in voxels (default: .5 voxel)
- `smooth_coeff::T`  : Vector smoothing coefficient, in [0-1] (default: .2)
- `micro_search_cosang::T`         : Cosine of search angle (default: cosd(10))
- `micro_search_dist::Vector{Int}` : Search distance, in voxels (default: 15)
- `strdims::Vector{Int}`  : In-plane dimensions for 2D LCM [2]
- `dxyz::Matrix{Int}`     : Coordinate increments for voxel jumps in LCM [3 4]
- `edgetype::Matrix{Int}` : Voxel edges connected by i-th LCM element [2 10]
- `mask::BitArray{3}`            : Brain mask (voxel-wise) [nx ny nz]
- `ovecs::Array{T, 5}`           : Orientation vectors [3 nvec nx ny nz]
- `lcms::Array{T, 4}`            : LCM elements [nmat nx ny nz]
- `sublist::Vector{Vector{T}}`   : Subvoxel sampling offsets [nsub][3]
- `pos_now::Vector{Vector{T}}`   : Current (x, y, z) position [ncore][3]
- `vec_now::Vector{Vector{T}}`   : Current orientation vector [ncore][3]
- `pos_next::Vector{Vector{T}}`  : Next (x, y, z) position [ncore][3]
- `vec_next::Vector{Vector{T}}`  : Next orientation vector [ncore][3]
- `ivec_next::Vector{Int}`       : Index of next orientation vector [ncore]
- `cosang::Vector{Vector{T}}`    : Cosine of angle b/w vectors [ncore][nvec]
- `cosangabs::Vector{Vector{T}}` : Absolute value of above [ncore][nvec]
- `dvox::Vector{Vector{Int}}`    : Change in voxel position [ncore][3]
- `lcm::Vector{Vector{T}}`       : LCM vector at single voxel [ncore][10]
- `isdiff::Vector{Bool}`         : Indicator of method difference [ncore]
- `str::Vector{Vector{Matrix{T}}}`     : Streamlines [ncore][nstr][3 npts]
- `flag::Vector{Vector{Vector{Bool}}}` : Flag on streamlines [ncore][nstr][npts]
"""
struct StreamWork{T}
  len_min::Int
  len_max::Int
  cosang_thresh::T
  step_size::T
  smooth_coeff::T
  micro_search_cosang::T
  micro_search_dist::Vector{Int}
  strdims::Vector{Int}
  dxyz::Matrix{Int}
  edgetype::Matrix{Int}
  mask::BitArray{3}
  ovecs::Array{T, 5}
  search_area::Array{Vector{T}, 3}
  lcms::Array{T, 4}
  sublist::Vector{Vector{T}}
  pos_now::Vector{Vector{T}}
  vec_now::Vector{Vector{T}}
  pos_next::Vector{Vector{T}}
  vec_next::Vector{Vector{T}}
  ivec_next::Vector{Int}
  cosang::Vector{Vector{T}}
  cosangabs::Vector{Vector{T}}
  cosang_area::Vector{Array{T, 3}}
  cosangabs_area::Vector{Array{T, 3}}
  dvox::Vector{Vector{Int}}
  lcm::Vector{Vector{T}}
  isdiff::Vector{Bool}
  str::Vector{Vector{Matrix{T}}}
  flag::Vector{Vector{Vector{Bool}}}

  function StreamWork(ovec::Union{MRI,Vector{MRI}}, T::DataType=Float32; f::Union{MRI,Vector{MRI},Nothing}=nothing, f_thresh::Real=.03, fa::Union{MRI,Nothing}=nothing, fa_thresh::Real=.1, mask::Union{MRI,Nothing}=nothing, nsub::Union{Integer,Nothing}=3, len_min::Integer=3, len_max::Integer=(isa(ovec,MRI) ? maximum(ovec.volsize) : maximum(ovec[1].volsize)), ang_thresh::Union{Real,Nothing}=45, step_size::Union{Real,Nothing}=.5, smooth_coeff::Union{Real,Nothing}=.2, search_dist::Integer=15, search_ang::Real=10, lcms::Union{MRI,Nothing}=nothing, lcm_thresh::Real=.099, verbose::Bool=false)

    ovecs = isa(ovec, MRI) ? Vector{MRI}([ovec]) : ovec
    fs    = isa(f, MRI)    ? Vector{MRI}([f])    : f

    nvec = length(ovecs)
    nx, ny, nz = size(ovecs[1].vol)
    nxyz = nx * ny * nz

    # Is this in the microscopy regime (min voxel size under 50 μm)?
    domicro = (minimum(ovecs[1].volres) <= 0.05)

    micro_search_dist = domicro ? fill(Int(search_dist), 3) : Int[]

    # Set defaults that depend on which scale we are operating in
    isnothing(nsub)         && (nsub         = Integer(domicro ? 0 : 3))
    isnothing(ang_thresh)   && (ang_thresh   = T(domicro ? 20 : 45))
    isnothing(step_size)    && (step_size    = T(domicro ? 1 : .5))
    isnothing(smooth_coeff) && (smooth_coeff = T(domicro ? 0 : .2))

    # Generate brain mask array
    if isnothing(mask)
      mask_array = falses(nx, ny, nz)

      for ivec in eachindex(ovecs)
        mask_array .= mask_array .|| any(x -> x!=0, ovecs[ivec].vol, dims=4)
      end
    else
      @views mask_array = (mask.vol[:,:,:,1] .> 0)
    end

    if !isnothing(fa)
      # Warn if voxel-wise threshold seems unreasonable
      fa_min = quantile(fa.vol[vec(mask_array)], 1e-5)
      fa_max = quantile(fa.vol[vec(mask_array)], .9)
      if fa_thresh < fa_min || fa_thresh > fa_max
        println("WARNING: The value of fa_thresh (" * string(fa_thresh) *
                ") is outside the range of most values in the fa volume (" *
                string(fa_min) * ", " * string(fa_max) * ")")
      end

      # Intersect brain mask with fa.vol >=fa_thresh
      @views mask_array .= mask_array .&& (fa.vol[:,:,:,1] .>= T(fa_thresh))
    end

    if !isnothing(f)
      # Warn if peak-wise threshold seems unreasonable
      f_min = quantile(fs[1].vol[vec(mask_array)], 1e-5)
      f_max = quantile(fs[1].vol[vec(mask_array)], .9)
      if f_thresh < f_min || f_thresh > f_max
        println("WARNING: The value of f_thresh (" * string(f_thresh) *
                ") is outside the range of most values in the f volume (" *
                string(f_min) * ", " * string(f_max) * ")")
      end
    end

    # Store orientation vectors for fast computation
    ovec_array  = zeros(T, 3, nvec, nx, ny, nz)

    omask_array = isnothing(f) ? mask_array : similar(mask_array)

    for ivec in eachindex(ovecs)
      if !isnothing(f)
        # Intersect brain mask_array with fs[ivec].vol >= f_thresh
        omask_array .= mask_array .&& fs[ivec].vol .>= T(f_thresh)
      end

      if size(ovecs[ivec].vol, 4) == 3		# Input is orientation vectors
        for idim in axes(ovecs[ivec].vol, 4) 
          ovec_array[idim, ivec, :, :, :] .= view(ovecs[ivec].vol,
                                                  :, :, :, idim) .* omask_array
        end
        # TODO: check if vectors are normalized?
      elseif size(ovecs[ivec].vol, 4) == 1	# Input is 2D orientation angles
        # Through-plane dimension (assume it is the one with max voxel size)
        thrudim = argmax(ovecs[ivec].volres)
        # In-plane dimensions
        strdims = setdiff(1:3, thrudim)

        if domicro
          micro_search_dist[thrudim] = 0
        end

        if -π/2-eps(T) <= minimum(ovecs[ivec].vol) && 
            maximum(ovecs[ivec].vol) <= π/2+eps(T)		# In radians
          ovec_array[strdims[1], ivec, :, :, :] .= cos.(ovecs[ivec].vol) .*
                                                   omask_array
          ovec_array[strdims[2], ivec, :, :, :] .= sin.(ovecs[ivec].vol) .*
                                                   omask_array
        elseif -90 <= minimum(ovecs[ivec].vol) && 
                      maximum(ovecs[ivec].vol) <= 90		# In degrees
          ovec_array[strdims[1], ivec, :, :, :] .= cosd.(ovecs[ivec].vol) .*
                                                   omask_array
          ovec_array[strdims[2], ivec, :, :, :] .= sind.(ovecs[ivec].vol) .*
                                                   omask_array
        else
          error("Input orientations should be 3D vectors or angles ∊ [-90, 90]")
        end
      end
    end

    # Subvoxel sampling
    if nsub > 0
      sublist = [T.(rand(Uniform(-.5+eps(), .5-eps()), 3))
                 for isub in 1:nsub]
    else
      sublist = [zeros(T, 3)]
    end

    # Vectors for point updates
    pos_now      = [zeros(T, 3) for tid in 1:Threads.nthreads()]
    vec_now      = [zeros(T, 3) for tid in 1:Threads.nthreads()]
    pos_next     = [zeros(T, 3) for tid in 1:Threads.nthreads()]
    vec_next     = [zeros(T, 3) for tid in 1:Threads.nthreads()]
    ivec_next    = Vector{Int}(undef, Threads.nthreads())
    cosang       = [zeros(T, nvec) for tid in 1:Threads.nthreads()]
    cosangabs    = [zeros(T, nvec) for tid in 1:Threads.nthreads()]

    # Max bending angle
    cosang_thresh = cosd(T(ang_thresh))
    cosang_45     = cosd(T(45))

    # For probabilistic: Determine ODF vertices around a peak
    # ...

    # Local connection matrices
    if isnothing(lcms)
      lcm_array = Array{T, 4}(undef, 0, 0, 0, 0)
      strdims   = Vector{Int}(undef, 0)
      dxyz      = Matrix{Int}(undef, 0, 0)
      edgetype  = Matrix{Int}(undef, 0, 0)
      dvox      = Vector{Vector{Int}}(undef, 0)
      lcm       = Vector{Vector{T}}(undef, 0)
      isdiff    = Vector{Bool}(undef, 0)
    else
      lcm_array = permutedims(lcms.vol, (4, 1, 2, 3))

      # Warn if threshold seems unreasonable
      lcm_max = maximum(lcms.vol)
      if lcm_thresh > lcm_max
        println("WARNING: The value of lcm_thresh (" * string(lcm_thresh) *
                ") is greater than the maximum value in the lcms volume (" *
                string(lcm_max) * ")")
      end

      # Intersect LCM elements with lcms.vol >= lcm_thresh
      lcm_array .*= (lcm_array .>= lcm_thresh)

      # 2D simplification: a voxel is a pixel in the x-z or y-z or x-y plane
      # Through-plane dimension
      thrudim = findall(vec(all(x -> x==0, ovecs[1].vol, dims=(1,2,3))))
      # In-plane dimensions
      strdims = setdiff(1:3, thrudim)

      # Coordinate increments for exiting through the i-th edge of a voxel
      dxyz = zeros(Int, 3, 4)
      dxyz[strdims[1], :] .= [-1, 0, 1, 0]
      dxyz[strdims[2], :] .= [0, -1, 0, 1]

      # Voxel edges connected by i-th element of a vectorized LCM
      edgetype = [1 1 1 1 2 2 2 3 3 4
                  1 2 3 4 2 3 4 3 4 4]

#=
      # From matrix to vectorized position in LCM
      mat2vec = [1 2 3 4
                 2 5 6 7
                 3 6 8 9
                 4 7 9 10]
=#
    end

    # Vectors for point updates using LCMs
    dvox       = [zeros(Int, 3) for tid in 1:Threads.nthreads()]
    lcm        = [zeros(T, size(lcm_array, 1)) for tid in 1:Threads.nthreads()]
    isdiff     = Vector{Bool}(undef, Threads.nthreads())

    # Fields for microscopy regime
    if domicro
      # Vector field for defining a search area as a cone around an
      # orientation vector
      search_area = Array{Vector{T}}(undef, 2 .* Tuple(micro_search_dist) .+ 1)

      for iz in axes(search_area, 3)
        for iy in axes(search_area, 2)
          for ix in axes(search_area, 1)
            # Radius of search area
            ρx = (ix - micro_search_dist[1] - 1) /
                      (micro_search_dist[1] + T(.5))
            ρy = (iy - micro_search_dist[2] - 1) /
                      (micro_search_dist[2] + T(.5))
            ρz = (iz - micro_search_dist[3] - 1) /
                      (micro_search_dist[3] + T(.5))
            ρ = sqrt(ρx^2 + ρy^2 + ρz^2)

            # Coordinates of each point in search area
            if ρ < 1
              search_area[ix, iy, iz] = [ρx/ρ, ρy/ρ, ρz/ρ] 
            else
              search_area[ix, iy, iz] = zeros(T, 3)
            end
          end
        end
      end

      # Vectors for point updates in microscopy regime
      cosang_area    = [similar(search_area, T) for tid in 1:Threads.nthreads()]
      cosangabs_area = [similar(search_area, T) for tid in 1:Threads.nthreads()]

      # Not needed in microscopy regime
      cosang = Vector{Vector{T}}(undef, 0)
      cosangabs = Vector{Vector{T}}(undef, 0)

      # Max search angle
      micro_search_cosang = cosd(T(search_ang))
    else
      search_area = Array{Vector{T}}(undef, 0, 0, 0)
      cosang_area = Vector{Array{T, 3}}(undef, 0)
      cosangabs_area = Vector{Array{T, 3}}(undef, 0)
      micro_search_cosang = T(Inf)
    end

    # Output vector of streamlines
    str = [Vector{Matrix{T}}(undef, 0) for tid in 1:Threads.nthreads()]

    # Flag used to indicate a difference between methods along streamlines
    flag = [Vector{Vector{Bool}}(undef, 0) for tid in 1:Threads.nthreads()]

    new{T}(
      Int(len_min)::Int,
      Int(len_max)::Int,
      cosang_thresh::T,
      T(step_size)::T,
      T(smooth_coeff)::T,
      T(micro_search_cosang)::T,
      micro_search_dist::Vector{Int},
      strdims::Vector{Int},
      dxyz::Matrix{Int},
      edgetype::Matrix{Int},
      mask_array::BitArray{3},
      ovec_array::Array{T, 5},
      search_area::Array{Vector{T}, 3},
      lcm_array::Array{T, 4},
      sublist::Vector{Vector{T}},
      pos_now::Vector{Vector{T}},
      vec_now::Vector{Vector{T}},
      pos_next::Vector{Vector{T}},
      vec_next::Vector{Vector{T}},
      ivec_next::Vector{Int},
      cosang::Vector{Vector{T}},
      cosangabs::Vector{Vector{T}},
      cosang_area::Vector{Array{T, 3}},
      cosangabs_area::Vector{Array{T, 3}},
      dvox::Vector{Vector{Int}},
      lcm::Vector{Vector{T}},
      isdiff::Vector{Bool},
      str::Vector{Vector{Matrix{T}}},
      flag::Vector{Vector{Vector{Bool}}}
    )
  end
end


"""
    Choose which orientation vector to follow to minimize the bending angle
"""
function stream_pick_by_angle!(ix_next::Int, iy_next::Int, iz_next::Int, W::StreamWork{T}) where T<:Number

  tid = Threads.threadid()

  vec_now   = W.vec_now[tid]
  vec_next  = W.vec_next[tid]
  cosang    = W.cosang[tid]
  cosangabs = W.cosangabs[tid]

  # Find orientation vector that is most similar to the current one
  for ivec in eachindex(cosang)
    v = view(W.ovecs, :, ivec, ix_next, iy_next, iz_next)

    if iszero(v)
      cosang[ivec] = cosangabs[ivec] = T(-Inf)
    else
      cosang[ivec] = dot(vec_now, v) 
      cosangabs[ivec] = abs(cosang[ivec])
    end
  end

  ivec_next = argmax(cosangabs)

  !isfinite(cosang[ivec_next]) && return false

  if cosang[ivec_next] > 0
    vec_next .=   view(W.ovecs, :, ivec_next, ix_next, iy_next, iz_next)
  else
    vec_next .= .-view(W.ovecs, :, ivec_next, ix_next, iy_next, iz_next)
  end

  W.ivec_next[tid] = ivec_next

  return true
end


"""
    Choose which orientation vector to follow based on the LCM
"""
function stream_pick_by_lcm!(ix_next::Int, iy_next::Int, iz_next::Int, W::StreamWork{T}) where T<:Number

  tid = Threads.threadid()

  pos_now   = W.pos_now[tid]
  vec_now   = W.vec_now[tid]
  pos_next  = W.pos_next[tid]
  vec_next  = W.vec_next[tid]
  cosang    = W.cosang[tid]
  cosangabs = W.cosangabs[tid]
  dvox      = W.dvox[tid]
  lcm       = W.lcm[tid]

  ix_now, iy_now, iz_now = round(Int, pos_now[1]), round(Int, pos_now[2]),
                                                   round(Int, pos_now[3])
  dvox[1] = ix_now - ix_next
  dvox[2] = iy_now - iy_next
  dvox[3] = iz_now - iz_next

  if all(dvox .== 0)
    # Not entering a new voxel => Continue along previouly chosen vector
    ivec_next = W.ivec_next[tid]

    v = view(W.ovecs, :, ivec_next, ix_next, iy_next, iz_next)

    if dot(vec_now, v) > 0
      vec_next .= v
    else
      vec_next .= .-v
    end

    return true
  else
    # Entering a new voxel => Find which edge I am entering it from
    entryedgetype = 0
    for j in axes(W.dxyz, 2)
      @views if dvox == W.dxyz[:,j]
        entryedgetype = j
        break
      end
    end

    if entryedgetype == 0
      # Resolve a diagonal jump (find which dimension changes faster)
      if abs(pos_now[W.strdims[1]] - pos_next[W.strdims[1]]) <
         abs(pos_now[W.strdims[2]] - pos_next[W.strdims[2]])
        dvox[W.strdims[2]] = 0
      else
        dvox[W.strdims[1]] = 0
      end

      for j in axes(W.dxyz, 2)
        @views if dvox == W.dxyz[:,j]
          entryedgetype = j
          break
        end
      end
    end

    # Find LCM elements that correspond to connections of the entry edge
    @views lcm .= W.lcms[:, ix_next, iy_next, iz_next]
    for j in axes(W.edgetype, 2)
      @views if !(entryedgetype in W.edgetype[:,j])
        lcm[j] = 0 
      end
    end

    while !iszero(lcm)
      # Random sampling on connections given entry edge
      lcm ./= sum(lcm)
      ilcm = rand(Categorical(lcm))

      # Exit edge for sampled connection
      exitedgetype = (W.edgetype[1, ilcm] == entryedgetype ?
                      W.edgetype[2, ilcm] : W.edgetype[1, ilcm])
      if isempty(exitedgetype)
        exitedgetype = entryedgetype
      end

      # Find orientation vector best aligned to a jump towards this exit edge
      for ivec in eachindex(cosang)
        v = view(W.ovecs, :, ivec, ix_next, iy_next, iz_next)

        if iszero(v)
          cosang[ivec] = cosangabs[ivec] = T(-Inf)
        else
          @views cosang[ivec] = dot(W.dxyz[:, exitedgetype], v)
          cosangabs[ivec] = abs(cosang[ivec])
        end
      end

      ivec_next = argmax(cosangabs)

      !isfinite(cosang[ivec_next]) && return false

      # Keep vector only if it's within 45 degrees of desired jump?
      if true		#abs(cosang[ivec_next]) > W.cosang_45
        if cosang[ivec_next] > 0
          vec_next .=   view(W.ovecs, :, ivec_next, ix_next, iy_next, iz_next)
        else
          vec_next .= .-view(W.ovecs, :, ivec_next, ix_next, iy_next, iz_next)
        end

        W.ivec_next[tid] = ivec_next

        return true
      end

      # Otherwise, sample another connection
      lcm[ilcm] = 0
    end

    # If no vector was found to match any of the connections in the LCM 
    iszero(lcm) && return false
  end
end


"""
    Generate new point for a streamline
"""
function stream_new_point!(W::StreamWork{T}) where T<:Number

  tid = Threads.threadid()

  pos_now  = W.pos_now[tid]
  vec_now  = W.vec_now[tid]
  pos_next = W.pos_next[tid]

  dolcm = !isempty(W.lcms)
  dodiff = dolcm

  pos_next .= pos_now .+ vec_now .* W.step_size	# Assuming vec is NORM=1!

  ix_next, iy_next, iz_next = round(Int, pos_next[1]), round(Int, pos_next[2]),
                                                       round(Int, pos_next[3])

  !(ix_next in axes(W.mask, 1) && iy_next in axes(W.mask, 2)
                               && iz_next in axes(W.mask, 3)) && return false

  !W.mask[ix_next, iy_next, iz_next] && return false

  # Get next direction from ODF (closest to current or based on LCM)
  if !dolcm
    # Choose which vector to follow based on similarity to current direction
    !stream_pick_by_angle!(ix_next, iy_next, iz_next, W) && return false
  else
    # Choose which vector to follow conventionally for comparison
    if dodiff
      !stream_pick_by_angle!(ix_next, iy_next, iz_next, W) && return false
      ivec_next_ang = W.ivec_next[tid]
    end

    # Choose which vector to follow based on LCM
    !stream_pick_by_lcm!(ix_next, iy_next, iz_next, W) && return false
    ivec_next = W.ivec_next[tid]

    W.isdiff[tid] = (dodiff && (ivec_next != ivec_next_ang))
  end

  return true
end


"""
    Generate new point for a micro streamline
"""
function stream_micro_new_point!(W::StreamWork{T}) where T<:Number

  tid = Threads.threadid()

  pos_now   = W.pos_now[tid]
  vec_now   = W.vec_now[tid]
  pos_next  = W.pos_next[tid]
  vec_next  = W.vec_next[tid]
  cosang    = W.cosang_area[tid]
  cosangabs = W.cosangabs_area[tid]

  # Set tentative next position
  pos_next .= pos_now .+ vec_now .* W.step_size	# Assuming vec is NORM=1!

  ix_next, iy_next, iz_next = round(Int, pos_next[1]), round(Int, pos_next[2]),
                                                       round(Int, pos_next[3])

  !(ix_next in axes(W.mask, 1) && iy_next in axes(W.mask, 2)
                               && iz_next in axes(W.mask, 3)) && return false

  !W.mask[ix_next, iy_next, iz_next] && return false

  # Find next position by finding the orientation vector closest to current one,
  # among all positions withing a search area around the tentative position
  ix_search = ix_next-W.micro_search_dist[1] : ix_next+W.micro_search_dist[1]
  iy_search = iy_next-W.micro_search_dist[2] : iy_next+W.micro_search_dist[2]
  iz_search = iz_next-W.micro_search_dist[3] : iz_next+W.micro_search_dist[3]

  search_area_off = OffsetArray(W.search_area, ix_search, iy_search, iz_search)
  cosang_off      = OffsetArray(cosang,        ix_search, iy_search, iz_search)
  cosangabs_off   = OffsetArray(cosangabs,     ix_search, iy_search, iz_search)

  fill!(cosang, T(-Inf))
  fill!(cosangabs, T(-Inf))

  ix_range = intersect(axes(search_area_off, 1), axes(W.mask, 1))
  iy_range = intersect(axes(search_area_off, 2), axes(W.mask, 2))
  iz_range = intersect(axes(search_area_off, 3), axes(W.mask, 3))

  for iz in iz_range
    for iy in iy_range
      for ix in ix_range
        v = search_area_off[ix, iy, iz]

        # Check if voxel is in mask & in search cone around current orientation
        (!W.mask[ix, iy, iz] || iszero(v) || 
         dot(vec_now, v) <= W.micro_search_cosang) && continue

        cosang_off[ix, iy, iz]    = dot(vec_now,
                                        view(W.ovecs, :, 1, ix, iy, iz))
        cosangabs_off[ix, iy, iz] = abs(cosang_off[ix, iy, iz])
      end
    end
  end

  # Find next orienation vector (most similar to current within search area)
  ivec_next = argmax(cosangabs_off)

  !isfinite(cosang_off[ivec_next]) && return false

  # Position where next orienation vector was found
  pos_next[1] = ix_next = ivec_next[1]
  pos_next[2] = iy_next = ivec_next[2]
  pos_next[3] = iz_next = ivec_next[3]

  if cosang_off[ivec_next] > 0
    vec_next .=   view(W.ovecs, :, 1, ix_next, iy_next, iz_next)
  else
    vec_next .= .-view(W.ovecs, :, 1, ix_next, iy_next, iz_next)
  end

  return true
end


"""
    Generate streamline for a seed voxel
"""
function stream_new_line(seed_vox::Vector{Int}, sub_vox::Vector{T}, W::StreamWork{T}) where T<:Number

  tid = Threads.threadid()

  pos_now  = W.pos_now[tid]
  vec_now  = W.vec_now[tid]
  pos_next = W.pos_next[tid]
  vec_next = W.vec_next[tid]

  dolcm = !isempty(W.lcms)
  dodiff = dolcm
  domicro = !isempty(W.search_area)

  npts = 0
  strline = Vector{T}(undef, 0)
  flagline = Vector{Bool}(undef, 0)

  # Initialize in random position within seed voxel
  # Get initial vector from seed voxel (the first vector, if many are available)
  # TODO: sample from the other vectors too?
  W.ivec_next[tid] = 1

  # Go forward and backward from starting position
  for fwd in (1, -1)
    pos_now .= seed_vox .+ sub_vox
    vec_now .= view(W.ovecs, :, W.ivec_next[tid], seed_vox...) .* fwd

    addpt! = (fwd == 1) ? prepend! : append!

    while true
      addnew = domicro ? stream_micro_new_point!(W) : stream_new_point!(W)

      !addnew && break

      # Save current position
      addpt!(strline, pos_now)
      npts += 1

      if dolcm
        if dodiff
          # Save indicator of method difference at current position
          addpt!(flagline, W.isdiff[tid])
        end
      else
        # Check angle threshold (not used with LCMs for now!!)
        dot(vec_now, vec_next) < W.cosang_thresh && break
      end

      # Always have a max length in case it gets stuck
      npts > W.len_max && break

      # Smooth next direction
      if W.smooth_coeff != 0
        vec_next .= W.smooth_coeff .* vec_now .+
                    (1 - W.smooth_coeff) .* vec_next
        vec_next ./= norm(vec_next)
      end

      # Move to next position
      pos_now .= pos_next
      vec_now .= vec_next
    end
  end

  return reshape(strline, 3, :), flagline
end


"""
    stream(ovec::Union{MRI,Vector{MRI}}; odf::Union{MRI,Nothing}=nothing}, f::Union{MRI,Vector{MRI},Nothing}=nothing, f_thresh::Real=.03, fa::Union{MRI,Nothing}=nothing, fa_thresh::Real=.1, mask::Union{MRI,Nothing}=nothing, lcms::Union{MRI,Nothing}=nothing, lcm_thresh::Real=.099, seed::Union{MRI,Nothing}=nothing, nsub::Union{Integer,Nothing}=3, len_min::Integer=3, len_max::Integer=(isa(ovec,MRI) ? maximum(ovec.volsize) : maximum(ovec[1].volsize)), ang_thresh::Union{Real,Nothing}=45, step_size::Union{Real,Nothing}=.5, smooth_coeff::Union{Real,Nothing}=.2, verbose::Bool=false, search_dist::Integer=15, search_ang::Number=10)

Streamline tractography

# Arguments
- `ovec::Union{MRI,Vector{MRI}}` : Orientation vectors [nvec][nx ny nz 3]

# Optional arguments
- `odf::MRI`                  : ODFs [nx ny nz 3 nvert] (default: none)
- `f::Union{MRI,Vector{MRI}}` : Vector amplitudes (or volume fractions),
                                for vector-wise masking [nvec][nx ny nz]
- `f_thresh::Real`            : Minimum vector amplitude (or volume fraction),
                                for vector-wise masking (default: .03)
- `fa::MRI`                   : FA (or other microstructural measure),
                                for voxel-wise masking (default: none)
- `fa_thresh::Real`           : Minimum FA (or other microstructural measure),
                                for voxel-wise masking (default: .1)
- `mask::MRI`                 : Brain mask [nx ny nz]
- `seed::Union{MRI,Nothing}`  : Seed mask [nx ny nz] (default: use brain mask)
- `nsub::Integer`      : Number of subvoxel samples per seed voxel (default: 3)
- `len_min::Integer`   : Minimum streamline length (default: 3)
- `len_max::Integer`   : Maximum streamline length (default: max(nx,ny,nz))
- `ang_thresh::Real`   : Maximum bending angle, in degrees (default: 45)
- `step_size::Real`    : Step length, in voxels (default: .5 voxel)
- `smooth_coeff::Real` : Vector smoothing coefficient, in [0-1] (default: .2)
- `search_dist::Integer`      : Micro search distance, in voxels (default: 15)
- `search_ang::Integer`       : Micro search angle, in degrees (default: 10)
- `lcms::Union{MRI,Nothing}`  : LCMs (default: none) [ny nx nz nmat]
- `lcm_thresh::Real`          : Minimum LCM coefficient (default: .099)
- `verbose::Bool`      : Count differences b/w propagation methods (default: no)

# Outputs
In the `Tract` structure
- `.str`: Voxel coordinates of points along streamlines
- `.scalar`: Indicator function of a method difference (only if `verbose==true`)
"""
function stream(ovec::Union{MRI,Vector{MRI}}; odf::Union{MRI,Nothing}=nothing, f::Union{MRI,Vector{MRI},Nothing}=nothing, f_thresh::Real=.03, fa::Union{MRI,Nothing}=nothing, fa_thresh::Real=.1, mask::Union{MRI,Nothing}=nothing, seed::Union{MRI,Nothing}=nothing, nsub::Union{Integer,Nothing}=3, len_min::Integer=3, len_max::Integer=(isa(ovec,MRI) ? maximum(ovec.volsize) : maximum(ovec[1].volsize)), ang_thresh::Union{Real,Nothing}=45, step_size::Union{Real,Nothing}=.5, smooth_coeff::Union{Real,Nothing}=.2, search_dist::Integer=15, search_ang::Real=10, lcms::Union{MRI,Nothing}=nothing, lcm_thresh::Real=.099, verbose::Bool=false)

  W = StreamWork(ovec; f=f, f_thresh=f_thresh, fa=fa, fa_thresh=fa_thresh,
                 mask=mask, nsub=nsub, len_min=len_min, len_max=len_max,
                 ang_thresh=ang_thresh, step_size=step_size,
                 smooth_coeff=smooth_coeff,
                 search_dist=search_dist, search_ang=search_ang,
                 lcms=lcms, lcm_thresh=lcm_thresh, verbose=verbose)

  dolcm = !isempty(W.lcms)
  dodiff = dolcm

  # Coordinates of seed voxels
  if isnothing(seed)		# Use brain mask as seed mask
    xyz_seed = findall(W.mask .> 0)
  else				# Use provided seed mask
    if size(seed.vol) != size(mask.vol)
      error("Dimension mismatch between seed mask " * string(size(seed.vol))
            * " and brain mask " * string(size(mask.vol)))
    end

    xyz_seed = findall(seed.vol .> 0)
  end

  voxlist = map(x -> Int.([x[1], x[2], x[3]]), xyz_seed)
#@show voxlist = voxlist[1:18]

  nvox_by_thread = div(length(voxlist), Threads.nthreads()) + 1
  istart = collect(1:nvox_by_thread:length(voxlist))
  iend = min.(istart .+ (nvox_by_thread-1), length(voxlist))

  Threads.@threads for icore in eachindex(istart)
    tid = Threads.threadid()

    for ivox in eachindex(voxlist)[istart[icore]:iend[icore]]
      for isub in eachindex(W.sublist)
        # Generate new streamline
        strline, flagline = stream_new_line(voxlist[ivox], W.sublist[isub], W)

        size(strline, 2) < W.len_min && continue

        # Concatenate streamline (vector of vectors) into matrix,
        # push matrix into vector of matrices
        push!(W.str[tid], strline)

        if dodiff
          # Also push indicator of method differences along streamline
          push!(W.flag[tid], flagline)
        end
      end
    end
  end

  # Create a Tract object
  tr = Tract{Float32}(mask)

  # Concatenate streamlines from all threads and add to Tract object
  str_add!(tr, reduce(vcat, W.str), reduce(vcat, W.flag))

  return tr
end


