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
- `mask::BitArray{3}`               : Brain mask [nx ny nz]
- `peak::Array{Vector{T}, 4}`       : ODF peak vectors [npeak nx ny nz][3]
- `lcms::Array{T, 4}`               : LCM elements [nmat nx ny nz]
- `sublist::Vector{Vector{T}}`      : Subvoxel sampling offsets [nsub][3]
- `pos_now::Vector{Vector{T}}`      : Current (x, y, z) position [ncore][3]
- `vec_now::Vector{Vector{T}}`      : Current orientation vector [ncore][3]
- `pos_next::Vector{Vector{T}}`     : Next (x, y, z) position [ncore][3]
- `vec_next::Vector{Vector{T}}`     : Next orientation vector [ncore][3]
- `cosang::Vector{Vector{T}}`       : Cosine of angle b/w vectors [ncore][npeak]
- `cosangabs::Vector{Vector{T}}`    : Absolute value of above [ncore][npeak]
- `str::Vector{Vector{Matrix{T}}}`  : Streamlines [ncore][nstr][3 npts]
- `flag::Vector{Vector{BitVector}}` : Flag along streamline [ncore][nstr][npts]
"""
struct StreamWork{T}
  len_min::Int
  len_max::Int
  cosang_thresh::T
  step_size::T
  smooth_coeff::T
  mask::BitArray{3}
  peak::Array{Vector{T}, 4}
  lcms::Array{T, 4}
  sublist::Vector{Vector{T}}
  pos_now::Vector{Vector{T}}
  vec_now::Vector{Vector{T}}
  pos_next::Vector{Vector{T}}
  vec_next::Vector{Vector{T}}
  cosang::Vector{Vector{T}}
  cosangabs::Vector{Vector{T}}
  str::Vector{Vector{Matrix{T}}}
  flag::Vector{Vector{BitVector}}

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
    pos_now   = [zeros(T, 3) for tid in 1:Threads.nthreads()]
    vec_now   = [zeros(T, 3) for tid in 1:Threads.nthreads()]
    pos_next  = [zeros(T, 3) for tid in 1:Threads.nthreads()]
    vec_next  = [zeros(T, 3) for tid in 1:Threads.nthreads()]
    cosang    = [zeros(T, npeak) for tid in 1:Threads.nthreads()]
    cosangabs = [zeros(T, npeak) for tid in 1:Threads.nthreads()]

    # Max bending angle
    cosang_thresh = cosd(T(ang_thresh))
    cosang_45     = cosd(T(45))

    # For probabilistic: Determine ODF vertices around a peak
    # ...

    # Local connection matrices
    if isnothing(lcms_mri)
      lcms = Array{T, 4}(undef, 0, 0, 0, 0)
    else
    end

    # Output vector of streamlines
    str = [Vector{Matrix{T}}(undef, 0) for tid in 1:Threads.nthreads()]


    # Flag of a difference between methods along streamlines
    flag = [Vector{BitVector}(undef, 0) for tid in 1:Threads.nthreads()]

    new{T}(
      Int(len_min)::Int,
      Int(len_max)::Int,
      cosang_thresh::T,
      T(step_size)::T,
      T(smooth_coeff)::T,
      mask::BitArray{3},
      peak::Array{Vector{T}, 4},
      lcms::Array{T, 4},
      sublist::Vector{Vector{T}},
      pos_now::Vector{Vector{T}},
      vec_now::Vector{Vector{T}},
      pos_next::Vector{Vector{T}},
      vec_next::Vector{Vector{T}},
      cosang::Vector{Vector{T}},
      cosangabs::Vector{Vector{T}},
      str::Vector{Vector{Matrix{T}}},
      flag::Vector{Vector{BitVector}}
    )
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
  vec_next = W.vec_next[tid]
  cosang   = W.cosang[tid]
  cosangabs = W.cosangabs[tid]

dolcm = false
dodiff = false

  pos_next .= pos_now .+ vec_now .* W.step_size	# Assuming vec is NORM=1!

  ix_next, iy_next, iz_next = round(Int, pos_next[1]), round(Int, pos_next[2]),
                                                       round(Int, pos_next[3])

  !(ix_next in axes(W.mask, 1) && iy_next in axes(W.mask, 2)
                               && iz_next in axes(W.mask, 3)) && return false

  !W.mask[ix_next, iy_next, iz_next] && return false

  # Get next direction from ODF (closest to current or based on LCM)
  for ipeak in eachindex(cosang)
    v = W.peak[ipeak, ix_next, iy_next, iz_next]

    if v[1] != 0 || v[2] != 0 || v[3] != 0
      cosang[ipeak] = vec_now' * v 
      cosangabs[ipeak] = abs(cosang[ipeak])
    else
      cosang[ipeak] = cosangabs[ipeak] = T(-Inf)
    end
  end

  if !dolcm
    # Choose which peak to follow based on similarity to current direction
    ipeak_next = argmax(cosangabs)
    !isfinite(cosang[ipeak_next]) && return false

    if cosang[ipeak_next] > 0
      vec_next .= W.peak[ipeak_next, ix_next, iy_next, iz_next]
    else
      vec_next .= .-W.peak[ipeak_next, ix_next, iy_next, iz_next]
    end
  else
    if dodiff
      ipeak_next_ang = argmax(cosangabs)
    end
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

dolcm = false
dodiff = false

  strline = Vector{Vector{T}}(undef, 0)

  # Initialize in random position within seed voxel
  # Get initial direction from ODF (peak with greatest amplitude)
  # TODO: try the smaller peaks too?
  ipeak0 = 1

  # Go forward and backward from starting position
  for fwd in (1, -1)
    pos_now .= seed_vox .+ sub_vox
    vec_now .= W.peak[ipeak0, seed_vox...] .* fwd

    while true
      addnew = stream_new_point!(W)

      !addnew && break

      # Save current position
      push!(strline, copy(pos_now))

      # Check angle threshold (not used with LCMs for now!!)
      if !dolcm
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
    end
  end

  return strline
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
        strline = stream_new_line(voxlist[ivox], W.sublist[isub], W)

        length(strline) < W.len_min && continue

        # Concatenate streamline (vector of vectors) into matrix,
        # push matrix into vector of matrices
        push!(W.str[tid], reduce(hcat, strline))
      end
    end
  end

  # Concatenate vectors of streamline matrices from all threads into one vector
  str = reduce(vcat, W.str)

  # Return a Tract object
  tr = Tract{Float32}(mask)

  str_add!(tr, str)

  return tr
end


