#=
  Original Author: Anastasia Yendiki

  Copyright © 2022 The General Hospital Corporation (Boston, MA) "MGH"
 
  Terms and conditions for use, reproduction, distribution and contribution
  are found in the 'FreeSurfer Software License Agreement' contained
  in the file 'LICENSE' found in the FreeSurfer distribution, and here:
 
  https://surfer.nmr.mgh.harvard.edu/fswiki/FreeSurferSoftwareLicense
 
  Reporting: freesurfer@nmr.mgh.harvard.edu
=#
 
using LinearAlgebra, Distributions

export stream, stream_new_line, stream_new_point!

"""
    StreamWork{T}

Pre-allocated workspace for streamline tractography

- `T::DataType`      : Data type (default: `Float32`)
- `len_min::Int`     : Minimum streamline length (default: 3)
- `len_max::Int`     : Maximum streamline length (default: max(nx,ny,nz))
- `cosang_thresh::T` : Cosine of maximum bending angle (default: cosd(45))
- `step_size::T`     : Step length, in voxels (default: .5 voxel)
- `smooth_coeff::T`  : Vector smoothing coefficient, in [0-1] (default: .2)
- `strdims::Vector{Int}`  : In-plane dimensions for 2D LCM [2]
- `dxyz::Matrix{Int}`     : Coordinate increments for voxel jumps in LCM [3 4]
- `edgetype::Matrix{Int}` : Voxel edges connected by i-th LCM element [2 10]
- `mask::BitArray{3}`            : Brain mask [nx ny nz]
- `peak::Array{Vector{T}, 4}`    : ODF peak vectors [npeak nx ny nz][3]
- `lcms::Array{T, 4}`            : LCM elements [nmat nx ny nz]
- `sublist::Vector{Vector{T}}`   : Subvoxel sampling offsets [nsub][3]
- `pos_now::Vector{Vector{T}}`   : Current (x, y, z) position [ncore][3]
- `vec_now::Vector{Vector{T}}`   : Current orientation vector [ncore][3]
- `pos_next::Vector{Vector{T}}`  : Next (x, y, z) position [ncore][3]
- `vec_next::Vector{Vector{T}}`  : Next orientation vector [ncore][3]
- `ipeak_next::Vector{Int}`      : Index of next orientation vector [ncore]
- `cosang::Vector{Vector{T}}`    : Cosine of angle b/w vectors [ncore][npeak]
- `cosangabs::Vector{Vector{T}}` : Absolute value of above [ncore][npeak]
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
  strdims::Vector{Int}
  dxyz::Matrix{Int}
  edgetype::Matrix{Int}
  mask::BitArray{3}
  peak::Array{Vector{T}, 4}
  lcms::Array{T, 4}
  sublist::Vector{Vector{T}}
  pos_now::Vector{Vector{T}}
  vec_now::Vector{Vector{T}}
  pos_next::Vector{Vector{T}}
  vec_next::Vector{Vector{T}}
  ipeak_next::Vector{Int}
  cosang::Vector{Vector{T}}
  cosangabs::Vector{Vector{T}}
  dvox::Vector{Vector{Int}}
  lcm::Vector{Vector{T}}
  isdiff::Vector{Bool}
  str::Vector{Vector{Matrix{T}}}
  flag::Vector{Vector{Vector{Bool}}}

  function StreamWork(peak_mri::Vector{MRI}, f_mri::Union{Vector{MRI},Nothing}=nothing, fa_mri::Union{MRI,Nothing}=nothing, mask_mri::Union{MRI,Nothing}=nothing, lcms_mri::Union{MRI,Nothing}=nothing, nsub::Integer=1, len_min::Integer=3, len_max::Integer=maximum(peak_mri.volsize), ang_thresh::Real=45, f_thresh::Real=.03, fa_thresh::Real=.1, lcm_thresh::Real=.099, step_size::Real=.5, smooth_coeff::Real=.2, verbose::Bool=false, T::DataType=Float32)

    npeak = length(peak_mri)
    nx, ny, nz = size(peak_mri[1].vol)
    nxyz = nx * ny * nz

    # Generate brain mask array
    if isnothing(mask_mri)
      mask = falses(nx, ny, nz)

      for ipeak in eachindex(peak_mri)
        mask .= mask .|| any(x -> x!=0, peak_mri[ipeak].vol, dims=4)
      end
    else
      @views mask = (mask_mri.vol[:,:,:,1] .> 0)
    end

    if !isnothing(fa_mri)
      # Warn if threshold seems unreasonable
      fa_min = quantile(fa_mri.vol[vec(mask)], 1e-5)
      fa_max = quantile(fa_mri.vol[vec(mask)], .9)
      if fa_thresh < fa_min || fa_thresh > fa_max
        println("WARNING: The value of fa_thresh (" * string(fa_thresh) *
                ") is outside the range of most values in the fa volume (" *
                string(fa_min) * ", " * string(fa_max) * ")")
      end

      # Intersect brain mask with fa.vol >=fa_thresh
      @views mask .= mask .&& (fa_mri.vol[:,:,:,1] .>= T(fa_thresh))
    end

    # Store peak vectors for fast computation
    peak = fill(zeros(T, 3), npeak, nx, ny, nz)
    for ipeak = 1:npeak
      if isnothing(f_mri)
        ind_mask = findall(vec(mask) .> 0)
      else
        if ipeak == 1
          # Warn if threshold seems unreasonable
          f_min = quantile(f_mri[1].vol[vec(mask)], 1e-5)
          f_max = quantile(f_mri[1].vol[vec(mask)], .9)
          if f_thresh < f_min || f_thresh > f_max
            println("WARNING: The value of f_thresh (" * string(f_thresh) *
                    ") is outside the range of most values in the f volume (" *
                    string(f_min) * ", " * string(f_max) * ")")
          end
        end

        # Intersect brain mask with f[ipeak].vol >= f_thresh
        ind_mask = findall(vec(mask) .> 0 
                           .&& vec(f_mri[ipeak].vol) .>= T(f_thresh))
      end

      peak[ipeak .+ (ind_mask.-1).*npeak] .=
        mapslices(x -> [x],
                  reshape(peak_mri[ipeak].vol, nxyz, :)[ind_mask, :], dims=2)
    end
    # TODO: check if vectors are normalized?

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
    ipeak_next   = Vector{Int}(undef, Threads.nthreads())
    cosang       = [zeros(T, npeak) for tid in 1:Threads.nthreads()]
    cosangabs    = [zeros(T, npeak) for tid in 1:Threads.nthreads()]

    # Max bending angle
    cosang_thresh = cosd(T(ang_thresh))
    cosang_45     = cosd(T(45))

    # For probabilistic: Determine ODF vertices around a peak
    # ...

    # Local connection matrices
    if isnothing(lcms_mri)
      lcms       = Array{T, 4}(undef, 0, 0, 0, 0)
      strdims    = Vector{Int}(undef, 0)
      dxyz       = Matrix{Int}(undef, 0, 0)
      edgetype   = Matrix{Int}(undef, 0, 0)
      dvox       = Vector{Vector{Int}}(undef, 0)
      lcm        = Vector{Vector{T}}(undef, 0)
      isdiff     = Vector{Bool}(undef, 0)
    else
      lcms = permutedims(lcms_mri.vol, (4, 1, 2, 3))

      # Warn if threshold seems unreasonable
      lcm_max = maximum(lcms_mri.vol)
      if lcm_thresh > lcm_max
        println("WARNING: The value of lcm_thresh (" * string(lcm_thresh) *
                ") is greater than the maximum value in the lcms volume (" *
                string(lcm_max) * ")")
      end

      # Intersect LCM elements with lcms.vol >= lcm_thresh
      lcms .*= (lcms .>= lcm_thresh)

      # 2D simplification: a voxel is a pixel in the x-z or y-z or x-y plane
      # Through-plane dimension
      thrudim = findall(vec(all(x -> x==0, peak_mri[1].vol, dims=(1,2,3))))
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

      # Vectors for point updates using LCMs
      dvox       = [zeros(Int, 3) for tid in 1:Threads.nthreads()]
      lcm        = [zeros(T, size(lcms, 1)) for tid in 1:Threads.nthreads()]
      isdiff     = Vector{Bool}(undef, Threads.nthreads())
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
      strdims::Vector{Int},
      dxyz::Matrix{Int},
      edgetype::Matrix{Int},
      mask::BitArray{3},
      peak::Array{Vector{T}, 4},
      lcms::Array{T, 4},
      sublist::Vector{Vector{T}},
      pos_now::Vector{Vector{T}},
      vec_now::Vector{Vector{T}},
      pos_next::Vector{Vector{T}},
      vec_next::Vector{Vector{T}},
      ipeak_next::Vector{Int},
      cosang::Vector{Vector{T}},
      cosangabs::Vector{Vector{T}},
      dvox::Vector{Vector{Int}},
      lcm::Vector{Vector{T}},
      isdiff::Vector{Bool},
      str::Vector{Vector{Matrix{T}}},
      flag::Vector{Vector{Vector{Bool}}}
    )
  end
end


"""
    Choose which ODF peak to follow to minimize the bending angle
"""
function stream_pick_by_angle!(ix_next::Int, iy_next::Int, iz_next::Int, W::StreamWork{T}) where T<:Number

  tid = Threads.threadid()

  vec_now   = W.vec_now[tid]
  vec_next  = W.vec_next[tid]
  cosang    = W.cosang[tid]
  cosangabs = W.cosangabs[tid]

  # Find ODF peak that is most similar to the current one
  for ipeak in eachindex(cosang)
    v = W.peak[ipeak, ix_next, iy_next, iz_next]

    if iszero(v)
      cosang[ipeak] = cosangabs[ipeak] = T(-Inf)
    else
      cosang[ipeak] = vec_now' * v 
      cosangabs[ipeak] = abs(cosang[ipeak])
    end
  end

  ipeak_next = argmax(cosangabs)

  !isfinite(cosang[ipeak_next]) && return false

  if cosang[ipeak_next] > 0
    vec_next .= W.peak[ipeak_next, ix_next, iy_next, iz_next]
  else
    vec_next .= .-W.peak[ipeak_next, ix_next, iy_next, iz_next]
  end

  W.ipeak_next[tid] = ipeak_next

  return true
end


"""
    Choose which ODF peak to follow based on the LCM
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
    # Not entering a new voxel => Continue along previouly chosen peak
    ipeak_next = W.ipeak_next[tid]

    if vec_now' * W.peak[ipeak_next, ix_next, iy_next, iz_next] > 0
      vec_next .= W.peak[ipeak_next, ix_next, iy_next, iz_next]
    else
      vec_next .= .-W.peak[ipeak_next, ix_next, iy_next, iz_next]
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

      # Find ODF peak best aligned to a jump in the direction of this
      # exit edge
      for ipeak in eachindex(cosang)
        v = W.peak[ipeak, ix_next, iy_next, iz_next]

        if iszero(v)
          cosang[ipeak] = cosangabs[ipeak] = T(-Inf)
        else
          @views cosang[ipeak] = W.dxyz[:, exitedgetype]' * v 
          cosangabs[ipeak] = abs(cosang[ipeak])
        end
      end

      ipeak_next = argmax(cosangabs)

      !isfinite(cosang[ipeak_next]) && return false

      # Keep peak only if it's within 45 degrees of desired jump?
      if true		#abs(cosang[ipeak_next]) > W.cosang_45
        if cosang[ipeak_next] > 0
          vec_next .= W.peak[ipeak_next, ix_next, iy_next, iz_next]
        else
          vec_next .= .-W.peak[ipeak_next, ix_next, iy_next, iz_next]
        end

        W.ipeak_next[tid] = ipeak_next

        return true
      end

      # Otherwise, sample another connection
      lcm[ilcm] = 0
    end

    # If no peak was found to match any of the connections in the LCM 
    iszero(lcm) && return false
  end
end


"""
    Generate new point for a streamline
"""
function stream_new_point!(W::StreamWork{T}) where T<:Number

  tid = Threads.threadid()

  pos_now      = W.pos_now[tid]
  vec_now      = W.vec_now[tid]
  pos_next     = W.pos_next[tid]

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
    # Choose which peak to follow based on similarity to current direction
    !stream_pick_by_angle!(ix_next, iy_next, iz_next, W) && return false
  else
    # Choose which peak to follow conventionally for comparison
    if dodiff
      !stream_pick_by_angle!(ix_next, iy_next, iz_next, W) && return false
      ipeak_next_ang = W.ipeak_next[tid]
    end

    # Choose which peak to follow based on LCM
    !stream_pick_by_lcm!(ix_next, iy_next, iz_next, W) && return false
    ipeak_next = W.ipeak_next[tid]

    W.isdiff[tid] = (dodiff && (ipeak_next != ipeak_next_ang))
  end

  return true
end


"""
    Generate streamline for a seed voxel
"""
function stream_new_line(seed_vox::Vector{Int}, sub_vox::Vector{T}, W::StreamWork{T}) where T<:Number

  tid = Threads.threadid()

  pos_now      = W.pos_now[tid]
  vec_now      = W.vec_now[tid]
  pos_next     = W.pos_next[tid]
  vec_next     = W.vec_next[tid]

  dolcm = !isempty(W.lcms)
  dodiff = dolcm

  strline = Vector{Vector{T}}(undef, 0)
  flagline = Vector{Bool}(undef, 0)

  # Initialize in random position within seed voxel
  # Get initial direction from ODF (peak with greatest amplitude)
  # TODO: try the smaller peaks too?
  W.ipeak_next[tid] = 1

  # Go forward and backward from starting position
  for fwd in (1, -1)
    pos_now .= seed_vox .+ sub_vox
    vec_now .= W.peak[W.ipeak_next[tid], seed_vox...] .* fwd

    while true
      addnew = stream_new_point!(W)

      !addnew && break

      # Save current position
      push!(strline, copy(pos_now))

      if dolcm
        if dodiff
          # Save indicator of method difference at current position
          push!(flagline, W.isdiff[tid])
        end
      else
        # Check angle threshold (not used with LCMs for now!!)
        (vec_now' * vec_next) < W.cosang_thresh && break
      end

      # Always have a max length in case it gets stuck
      length(strline) > W.len_max && break

      # Smooth next direction
      vec_next .= W.smooth_coeff .* vec_now .+ (1 - W.smooth_coeff) .* vec_next
      vec_next ./= norm(vec_next)

      # Move to next position
      pos_now .= pos_next
      vec_now .= vec_next
    end

    if fwd == 1
      reverse!(strline, dims=:)
      if dodiff
        reverse!(flagline, dims=:)
      end
    end
  end

  return strline, flagline
end


"""
    stream(peak::Union{MRI,Vector{MRI}}; odf::Union{MRI,Nothing}=nothing}, f::Union{MRI,Vector{MRI},Nothing}=nothing, f_thresh::Real=.03, fa::Union{MRI,Nothing}=nothing, fa_thresh::Real=.1, mask::Union{MRI,Nothing}=nothing, lcms::Union{MRI,Nothing}=nothing, lcm_thresh::Real=.099, seed::Union{MRI,Nothing}=nothing, nsub::Integer=3, len_min::Integer=3, len_max::Integer=(isa(peak,MRI) ? maximum(peak.volsize) : maximum(peak[1].volsize)), ang_thresh::Real=45, step_size::Real=.5, smooth_coeff::Real=.2, verbose::Bool=false)

Streamline tractography

# Arguments
- `peak::Union{MRI,Vector{MRI}}` : ODF peak vectors [npeak][nx ny nz 3]

# Optional arguments
- `odf::MRI`                     : ODFs [nx ny nz 3 nvert] (default: none)
- `f::Union{MRI,Vector{MRI}}`    : ODF peak amplitudes (or volume fractions),
                                   for peak-wise masking [npeak][nx ny nz]
- `f_thresh::Real`               : Minimum peak amplitude (or volume fraction),
                                   for peak-wise masking (default: .03)
- `fa::MRI`                      : FA (or other microstructure measure),
                                   for voxel-wise masking (default: none)
- `fa_thresh::Real`              : Minimum FA (or other microstructure measure),
                                   for voxel-wise masking (default: .1)
- `mask::MRI`                    : Brain mask [nx ny nz]
- `lcms::Union{MRI,Nothing}`     : LCMs (default: none) [ny nx nz nmat]
- `lcm_thresh::Real`             : Minimum LCM coefficient (default: .099)
- `seed::Union{MRI,Nothing}`     : Seed mask (default: none, use brain mask)
- `nsub::Integer`      : Number of subvoxel samples per seed voxel (default: 3)
- `len_min::Integer`   : Minimum streamline length (default: 3)
- `len_max::Integer`   : Maximum streamline length (default: max(nx,ny,nz))
- `ang_thresh::Real`   : Maximum bending angle, in degrees (default: 45)
- `step_size::Real`    : Step length, in voxels (default: .5 voxel)
- `smooth_coeff::Real` : Vector smoothing coefficient, in [0-1] (default: .2)
- `verbose::Bool`      : Count differences b/w propagation methods (default: no)

# Outputs
In the `Tract` structure
- `.str`: Voxel coordinates of points along streamlines
- `.scalar`: Indicator function of a method difference (only if `verbose==true`)
"""
function stream(peak::Union{MRI,Vector{MRI}}; odf::Union{MRI,Nothing}=nothing, f::Union{MRI,Vector{MRI},Nothing}=nothing, f_thresh::Real=.03, fa::Union{MRI,Nothing}=nothing, fa_thresh::Real=.1, mask::Union{MRI,Nothing}=nothing, lcms::Union{MRI,Nothing}=nothing, lcm_thresh::Real=.099, seed::Union{MRI,Nothing}=nothing, nsub::Integer=3, len_min::Integer=3, len_max::Integer=(isa(peak,MRI) ? maximum(peak.volsize) : maximum(peak[1].volsize)), ang_thresh::Real=45, step_size::Real=.5, smooth_coeff::Real=.2, verbose::Bool=false)

  W = StreamWork(isa(peak, MRI) ? Vector{MRI}([peak]) : peak,
                 isa(f, MRI) ? Vector{MRI}([f]) : f,
                 fa, mask, lcms,
                 nsub, len_min, len_max, ang_thresh, f_thresh, fa_thresh,
                 lcm_thresh, step_size, smooth_coeff, verbose)

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

    @views xyz_seed = findall(seed.vol[:,:,:,1] .> 0)
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

        length(strline) < W.len_min && continue

        # Concatenate streamline (vector of vectors) into matrix,
        # push matrix into vector of matrices
        push!(W.str[tid], reduce(hcat, strline))

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


