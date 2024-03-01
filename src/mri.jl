#=
  Original Author: Anastasia Yendiki

  Copyright © 2022 The General Hospital Corporation (Boston, MA) "MGH"
 
  Terms and conditions for use, reproduction, distribution and contribution
  are found in the 'FreeSurfer Software License Agreement' contained
  in the file 'LICENSE' found in the FreeSurfer distribution, and here:
 
  https://surfer.nmr.mgh.harvard.edu/fswiki/FreeSurferSoftwareLicense
 
  Reporting: freesurfer@nmr.mgh.harvard.edu
=#
 
#=
  File i/o for nii/mgh files based on MATLAB code by Doug Greve, Bruce Fischl:
    MRIfspec.m
    MRIread.m
    MRIwrite.m
    load_mgh.m
    load_nifti.m
    load_nifti_hdr.m
    save_nifti.m
    fsgettmppath.m
    vox2ras_0to1.m
    vox2ras_tkreg.m
    vox2rasToQform.m
=#

using LinearAlgebra, Printf, DelimitedFiles

export MRI, NIfTIheader, get_tmp_path, mri_filename, mri_read, mri_write,
       mri_read_bfiles, mri_read_bfiles!


"Container for header of a volume stored in NIfTI format"
struct NIfTIheader
  # NIfTI standard header fields
  sizeof_hdr::Int32
  data_type::NTuple{10, UInt8}
  db_name::NTuple{18, UInt8}
  extents::Int32
  session_error::Int16
  regular::UInt8
  dim_info::UInt8
  dim::NTuple{8,Int16}
  intent_p1::Float32
  intent_p2::Float32
  intent_p3::Float32
  intent_code::Int16
  datatype::Int16
  bitpix::Int16
  slice_start::Int16
  pixdim::NTuple{8,Float32}
  vox_offset::Float32
  scl_slope::Float32
  scl_inter::Float32
  slice_end::Int16
  slice_code::Int8
  xyzt_units::Int8
  cal_max::Float32
  cal_min::Float32
  slice_duration::Float32
  toffset::Float32
  glmax::Int32
  glmin::Int32
  descrip::NTuple{80,UInt8}
  aux_file::NTuple{24,UInt8}
  qform_code::Int16
  sform_code::Int16
  quatern_b::Float32
  quatern_c::Float32
  quatern_d::Float32
  quatern_x::Float32
  quatern_y::Float32
  quatern_z::Float32
  srow_x::NTuple{4,Float32}
  srow_y::NTuple{4,Float32}
  srow_z::NTuple{4,Float32}
  intent_name::NTuple{16,UInt8}
  magic::NTuple{4,UInt8}

  # Additional fields
  do_bswap::Bool
  sform::Matrix{Float32}
  qform::Matrix{Float32}
  vox2ras::Matrix{Float32}
end


"Container for header and image data of an MRI volume or volume series"
mutable struct MRI{A<:AbstractArray}
  vol::A
  ispermuted::Bool
  image_type::String
  niftihdr::NIfTIheader

  fspec::String
  pwd::String

  flip_angle::Float32
  tr::Float32
  te::Float32
  ti::Float32

  vox2ras0::Matrix{Float32}
  volsize::Vector{Int32}
  height::Int32
  width::Int32
  depth::Int32
  nframes::Int32

  vox2ras::Matrix{Float32}
  nvoxels::Int32
  xsize::Float32
  ysize::Float32
  zsize::Float32

  x_r::Float32
  x_a::Float32
  x_s::Float32

  y_r::Float32
  y_a::Float32
  y_s::Float32

  z_r::Float32
  z_a::Float32
  z_s::Float32

  c_r::Float32
  c_a::Float32
  c_s::Float32

  vox2ras1::Matrix{Float32}
  Mdc::Matrix{Float32}
  volres::Vector{Float32}
  tkrvox2ras::Matrix{Float32}

  bval::Vector{Float32}
  bvec::Matrix{Float32}
end


"""
    MRI(vol::Array{T,N}) where T<:Number

Return an empty `MRI` structure
"""
MRI(vol::Array{T}) where T<:Number = MRI(
  vol,
  false,
  "",
  NIfTIheader(Int32(0),
              Tuple(zeros(UInt8, 10)),
              Tuple(zeros(UInt8, 18)),
              Int32(0),
              Int16(0),
              UInt8(0),
              UInt8(0),
              Tuple(zeros(Int16, 8)),
              Float32(0),
              Float32(0),
              Float32(0),
              Int16(0),
              Int16(0),
              Int16(0),
              Int16(0),
              Tuple(zeros(Float32, 8)),
              Float32(0),
              Float32(0),
              Float32(0),
              Int16(0),
              Int8(0),
              Int8(0),
              Float32(0),
              Float32(0),
              Float32(0),
              Float32(0),
              Int32(0),
              Int32(0),
              Tuple(zeros(UInt8, 80)),
              Tuple(zeros(UInt8, 24)),
              Int16(0),
              Int16(0),
              Float32(0),
              Float32(0),
              Float32(0),
              Float32(0),
              Float32(0),
              Float32(0),
              Tuple(zeros(Float32, 4)),
              Tuple(zeros(Float32, 4)),
              Tuple(zeros(Float32, 4)),
              Tuple(zeros(UInt8, 16)),
              Tuple(zeros(UInt8, 4)),

              UInt8(0),
              Matrix{Float32}(undef, 0, 0),
              Matrix{Float32}(undef, 0, 0),
              Matrix{Float32}(undef, 0, 0)
             ),

  "",
  "",

  Float32(0),
  Float32(0),
  Float32(0),
  Float32(0),

  Matrix{Float32}(undef, 0, 0),
  Vector{Int32}(undef, 0),
  Int32(0),
  Int32(0),
  Int32(0),
  Int32(0),

  Matrix{Float32}(undef, 0, 0),
  Int32(0),
  Float32(0),
  Float32(0),
  Float32(0),

  Float32(0),
  Float32(0),
  Float32(0),

  Float32(0),
  Float32(0),
  Float32(0),

  Float32(0),
  Float32(0),
  Float32(0),

  Float32(0),
  Float32(0),
  Float32(0),

  Matrix{Float32}(undef, 0, 0),
  Matrix{Float32}(undef, 0, 0),
  Vector{Float32}(undef, 0),
  Matrix{Float32}(undef, 0, 0),

  Vector{Float32}(undef, 0),
  Matrix{Float32}(undef, 0, 0)
)


"""
    MRI(ref::MRI{A}, nframes::Integer=ref.nframes, datatype::DataType=eltype(A)) where A<:AbstractArray

Return an `MRI` structure whose header fields are populated based on a
reference `MRI` structure `ref`, and whose image array is populated with
zeros.

Optionally, the new `MRI` structure can be created with a different number of
frames (`nframes`) than the reference MRI structure.
"""
function MRI(ref::MRI{A}, nframes::Integer=ref.nframes, datatype::DataType=eltype(A)) where A<:AbstractArray

  if nframes == 1
    mri = MRI(zeros(datatype, ref.volsize...))
  else
    mri = MRI(zeros(datatype, ref.volsize..., nframes))
  end
  
  for var in fieldnames(MRI)
    any(var .== (:vol, :fspec, :bval, :bvec)) && continue
    setfield!(mri, var, getfield(ref, var))
  end

  mri.nframes = nframes

  return mri
end


"""
    get_tmp_path(tmpdir::String="")

Return path to a directory where temporary files can be stored.

Search for candidate directories in the following order:
 1. `\$``TMPDIR`:  Check if environment variable is defined and directory exists
 2. `\$``TEMPDIR`: Check if environment variable is defined and directory exists
 3. `/scratch`:    Check if directory exists
 4. `/tmp`:        Check if directory exists
 5. `tmpdir`:      Check if `tmpdir` argument was passed and directory exists

If none of the above exist, use current directory (`./`) and print warning.
"""
function get_tmp_path(tmpdir::String="")

  if haskey(ENV, "TMPDIR")
    tmppath = ENV["TMPDIR"]
    if isdir(tmppath)
      return tmppath
    end
  end

  if haskey(ENV, "TEMPDIR")
    tmppath = ENV["TEMPDIR"]
    if isdir(tmppath)
      return tmppath
    end
  end  

  tmppath = "/scratch"
  if isdir(tmppath)
    return tmppath
  end

  tmppath = "/tmp"
  if isdir(tmppath)
    return tmppath
  end

  tmppath = tmpdir
  if isdir(tmppath)
    return tmppath
  end

  tmppath = "./"
  println("WARNING: get_tmp_path could not find a temporary folder, " *
	  "using current folder")
  return tmppath
end


"""
    vox2ras_0to1(M0::Matrix)

Convert a 0-based vox2ras matrix `M0` to a 1-based vox2ras matrix such that:

Pxyz = M_0 * [c r s 1]' = M_1 * [c+1 r+1 s+1 1]'
"""
function vox2ras_0to1(M0::Matrix)

  if size(M0) != (4,4)
    error("Input must be a 4x4 matrix")
  end

  Q = zeros(4, 4)
  Q[1:3, 4] = ones(3, 1)

  M1 = inv(inv(M0)+Q)

  return M1
end


"""
    vox2ras_tkreg(voldim::Vector, voxres::Vector)

Return a 0-based vox2ras transform of a volume that is compatible with the
registration matrix produced by tkregister. May work with MINC xfm.

# Arguments
- voldim = [ncols;  nrows;  nslices ...]
- volres = [colres; rowres; sliceres ...]
"""
function vox2ras_tkreg(voldim::Vector, voxres::Vector)

  if length(voldim) < 3 | length(voxres) < 3
    error("Input vectors must have at least 3 elements")
  end

  T = zeros(4,4)
  T[4,4] = 1

  T[1,1] = -voxres[1]
  T[1,4] =  voxres[1] * voldim[1]/2

  T[2,3] =  voxres[3]
  T[2,4] = -voxres[3] * voldim[3]/2

  T[3,2] = -voxres[2]
  T[3,4] =  voxres[2] * voldim[2]/2

  return T
end


"""
    vox2ras_to_qform(vox2ras::Matrix)

Convert a vox2ras matrix to NIfTI qform parameters. The vox2ras should be 6 DOF.

Return the following NIfTI header fields:
- hdr.quatern_b
- hdr.quatern_c
- hdr.quatern_d
- hdr.qoffset_x
- hdr.qoffset_y
- hdr.qoffset_z
- hdr.pixdim[1]

From DG's vox2rasToQform.m:
  This code mostly just follows CH's mriToNiftiQform() in mriio.c
"""
function vox2ras_to_qform(vox2ras::Matrix)

  if size(vox2ras) != (4, 4)
    error("vox2ras size=" * string(size(vox2ras)) * ", must be (4, 4)")
  end

  x = vox2ras[1,4]
  y = vox2ras[2,4]
  z = vox2ras[3,4]

  d = sqrt.(sum(vox2ras[:, 1:3].^2; dims=1))
  Mdc = vox2ras[1:3, 1:3] ./ repeat(d, 3)
  if det(Mdc) == 0
    error("vox2ras determinant is 0")
  end

  r11 = Mdc[1,1]
  r21 = Mdc[2,1]
  r31 = Mdc[3,1]
  r12 = Mdc[1,2]
  r22 = Mdc[2,2]
  r32 = Mdc[3,2]
  r13 = Mdc[1,3]
  r23 = Mdc[2,3]
  r33 = Mdc[3,3]

  if det(Mdc) > 0
    qfac = 1.0
  else
    r13 = -r13
    r23 = -r23
    r33 = -r33
    qfac = -1.0
  end

  # From DG's vox2rasToQform.m: "following mat44_to_quatern()"
  a = r11 + r22 + r33 + 1.0
  if a > 0.5
    a = 0.5 * sqrt(a)
    b = 0.25 * (r32-r23) / a
    c = 0.25 * (r13-r31) / a
    d = 0.25 * (r21-r12) / a
  else
    xd = 1.0 + r11 - (r22+r33)
    yd = 1.0 + r22 - (r11+r33)
    zd = 1.0 + r33 - (r11+r22)
    if xd > 1
      b = 0.5 * sqrt(xd)
      c = 0.25 * (r12+r21) / b
      d = 0.25 * (r13+r31) / b
      a = 0.25 * (r32-r23) / b
    elseif yd > 1
      c = 0.5 * sqrt(yd)
      b = 0.25 * (r12+r21) / c
      d = 0.25 * (r23+r32) / c
      a = 0.25 * (r13-r31) / c
    else
      d = 0.5 * sqrt(zd)
      b = 0.25 * (r13+r31) / d
      c = 0.25 * (r23+r32) / d
      a = 0.25 * (r21-r12) / d
    end
    if a < 0
      a = -a
      b = -b
      c = -c
      d = -d
    end
  end

  return [b, c, d, x, y, z, qfac]
end


"""
    vox2ras_to_orient(vox2ras::Matrix)

Convert a vox2ras matrix to a 3-character array indicating image orientation,
e.g., RAS, LIA, etc.
"""
function vox2ras_to_orient(vox2ras::Matrix)

  orient = Vector{Char}(undef, 3)

  for idim in 1:3
    (amax, imax) = findmax(abs.(vox2ras[1:3, idim]))
    if imax == 1
      if vox2ras[imax, idim] > 0
        orient[idim] = 'R'
      else
        orient[idim] = 'L'
      end
    elseif imax == 2
      if vox2ras[imax, idim] > 0
        orient[idim] = 'A'
      else
        orient[idim] = 'P'
      end
    else
      if vox2ras[imax, idim] > 0
        orient[idim] = 'S'
      else
        orient[idim] = 'I'
      end
    end
  end

  return orient
end


"""
    mri_filename(fstring::String, checkdisk::Bool=true)

Return a valid file name, file stem, and file extension, given a string
`fstring` that can be either a file name or a file stem.

Valid extensions are: mgh, mgz, nii, nii.gz. Thus a file name is expected to
have the form stem.{mgh, mgz, nii, nii.gz}.

If `fstring` is a file name, then the stem and extension are determined from
`fstring`.

If `fstring` is a file stem and `checkdisk` is true (default), then the
file name and extension are determined by searching for a file on disk
named `fstring`.{mgh, mgz, nii, nii.gz}, where the possible extensions are
searched for in this order. If no such file is found, then empty strings are
returned.
"""
function mri_filename(fstring::String, checkdisk::Bool=true)

  fname = ""
  fstem = ""
  fext   = ""

  # List of supported file extensions
  extlist = ["mgh", "mgz", "nii", "nii.gz"]

  idot = findlast(isequal('.'), fstring)

  if isnothing(idot) && checkdisk
    # Filename has no extension, check if file exists with a supported extension
    for ext in extlist
      name = fstring * '.' * ext

      if isfile(name)
        fname = name
        fstem = fstring
        fext = ext
      end
    end
  else
    # Filename has an extension, check if it is one of the supported ones
    ext = lowercase(fstring[(idot+1):end]);

    if cmp(ext, "gz") == 0
      idot = findprev(isequal('.'), fstring, idot-1)

      if !isnothing(idot)
        ext = lowercase(fstring[(idot+1):end])
      end
    end

    if any(cmp.(ext, extlist) .== 0)
      fname = fstring
      fstem = fstring[1:idot-1]
      fext = ext
    end
  end

  return fname, fstem, fext
end


"""
    mri_read(infile::String; headeronly::Bool=false, permutedata::Bool=false, reco::Integer=1)

Read an image volume from disk and return an `MRI` structure similar to the
FreeSurfer MRI struct defined in mri.h.

# Arguments
- `infile::String`: Path to the input, which can be:
  1. An MGH file, e.g., f.mgh or f.mgz
  2. A NIfTI file, e.g., f.nii or f.nii.gz
  3. A file stem, in which case the format and full file name are determined
     by finding a file on disk named `infile`.{mgh, mgz, nii, nii.gz}
  4. A Bruker scan directory, which is expected to contain the following files:
     method, acqp, pdata/1/reco, pdata/1/2dseq

# Optional arguments
- `headeronly::Bool=false`: If true, the pixel data are not read in.

- `permutedata::Bool==false`: If true, the first two dimensions of the image
  volume are permuted in the .vol, .volsize, and .volres fields of the output
  structure (this is the default behavior of the MATLAB version). The
  `permutedata` will not affect the vox2ras matrices, which always map indices
  in the (column, row, slice) convention.

- `reco::Integer=1`: (Only relevant to reading Bruker scan directories) Number
  of image reconstruction to load, where 1 denotes the online reconstruction
  and higher numbers denote any additional offline reconstructions performed
  after acquisition.

# Output
In the `MRI` structure:
- Times are in ms and angles are in radians.

- `vox2ras0`: This field contains the vox2ras matrix that converts 0-based
  (column, row, slice) voxel indices to (x, y, z) coordinates. This is the also
  the matrix that mri_write() uses to derive all geometry information
  (e.g., direction cosines, voxel resolution, P0). Thus if any other geometry
  fields of the structure are modified, the change will not be reflected in
  the output volume.

- `vox2ras1`: This field contains the vox2ras matrix that converts 1-based
  (column, row, slice) voxel indices to (x, y, z) coordinates.

- `niftihdr`: If the input file is NIfTI, this field contains a `NIfTIheader`
  structure.
"""
function mri_read(infile::String; headeronly::Bool=false, permutedata::Bool=false, reco::Integer=1)

  if isdir(infile)					#------ Bruker -------#
    mri = load_bruker(infile; headeronly=headeronly, reco=reco)
  else
    (fname, fstem, fext) = mri_filename(infile)
    if isempty(fname) 
      error("Cannot determine format of " * infile)
    end

    if any(fext .== ["mgh", "mgz"])			#-------- MGH --------#
      (vol, M, mr_parms, volsz) = 
       load_mgh(fname; headeronly=headeronly)

      mri = MRI(vol)

      mri.fspec = fname
      mri.pwd = pwd()

      if !isempty(mr_parms)
        (mri.tr, mri.flip_angle, mri.te, mri.ti) = mr_parms
      end

      if isempty(M)
        error("Loading " * fname * " as MGH")
      else
        mri.vox2ras0 = M
      end

      mri.volsize = volsz[1:3]
      mri.nframes = length(volsz) < 4 ? 1 : volsz[4]
    elseif any(fext .== ["nii", "nii.gz"])		#------- NIfTI -------#
      hdr, vol = load_nifti(fname; headeronly=headeronly)

      if !headeronly && isempty(vol)
        error("Loading " * fname * " as NIfTI")
      end

      # Compatibility with MRIread.m:
      # When data have > 4 dims, put all data into dim 4.
      volsz = Int.(hdr.dim[2:end])
      volsz = volsz[findall(volsz.>0)]

      if length(volsz) < 5
        mri = MRI(vol)
      else
        if headeronly
          mri = MRI(Array{eltype(vol), 4}(undef, 0, 0, 0, 0))
        else
          mri = MRI(reshape(vol, volsz[1], volsz[2], volsz[3], :))
        end
      end

      mri.fspec = fname
      mri.pwd = pwd()

      mri.niftihdr = hdr

      mri.tr = hdr.pixdim[5]	# Already in msec

      mri.flip_angle = mri.te = mri.ti = 0

      mri.vox2ras0 = hdr.vox2ras

      mri.volsize = collect(volsz[1:3])
      mri.nframes = length(volsz) < 4 ? 1 : volsz[4]
    else
      error("File extension " * fext * " not supported")
    end

    # Optional DWI tables -----------------------------------------------#

    bfile = fstem * ".bvals"

    if !isfile(bfile)
      bfile = fstem * ".bval"

      if !isfile(bfile)
        bfile = ""
      end
    end

    gfile = fstem * ".bvecs"

    if !isfile(gfile)
      gfile = fstem * ".bvec"

      if !isfile(gfile)
        gfile = ""
      end
    end

    if !isempty(bfile) && !isempty(gfile)
      (b, g) = mri_read_bfiles(bfile, gfile)

      if length(b) == mri.nframes
        mri.bval = b
        mri.bvec = g

        # Normalize gradient vectors
        mri.bvec .= mri.bvec ./ sqrt.(sum(mri.bvec.^2, dims=2))
        mri.bvec[isnan.(mri.bvec)] .= Float32(0)
      end
    end
  end

  # Dimensions not redundant when using header only
  (mri.width, mri.height, mri.depth) = collect(mri.volsize)

  # Set additional header fields related to volume geometry
  mri_set_geometry!(mri)

  # Permute volume from col-row-slice to row-col-slice, if desired
  # (This is the default behavior in the MATLAB version, but not here)
  if permutedata
    mri.vol = permutedims(mri.vol, [2; 1; 3:ndims(mri.vol)])
    mri.volsize = mri.volsize[[2,1,3]]
    mri.volres = mri.volres[[2,1,3]]
    mri.ispermuted = true
  end

  return mri
end


"""
    mri_set_geometry!(mri::MRI)

Set header fields related to volume geometry in an `MRI` structure.

These are redundant fields that can be derived from the `vox2ras0`,
`width`, `height`, `depth` fields. They are in the MRI struct defined
in mri.h, as well as the MATLAB version of this structure, so they have
been added here for completeness.

Note: mri_write() derives all geometry information (i.e., direction cosines,
voxel resolution, and P0 from vox2ras0. If any of the redundant geometry
elements below are modified, the change will not be reflected in the output
volume.
"""
function mri_set_geometry!(mri::MRI)

  mri.vox2ras = mri.vox2ras0

  mri.nvoxels = mri.width * mri.height * mri.depth	# Number of voxels
  mri.xsize = sqrt(sum(mri.vox2ras[:,1].^2))	# Col
  mri.ysize = sqrt(sum(mri.vox2ras[:,2].^2))	# Row
  mri.zsize = sqrt(sum(mri.vox2ras[:,3].^2))	# Slice

  mri.x_r = mri.vox2ras[1,1]/mri.xsize	# Col
  mri.x_a = mri.vox2ras[2,1]/mri.xsize
  mri.x_s = mri.vox2ras[3,1]/mri.xsize

  mri.y_r = mri.vox2ras[1,2]/mri.ysize	# Row
  mri.y_a = mri.vox2ras[2,2]/mri.ysize
  mri.y_s = mri.vox2ras[3,2]/mri.ysize

  mri.z_r = mri.vox2ras[1,3]/mri.zsize	# Slice
  mri.z_a = mri.vox2ras[2,3]/mri.zsize
  mri.z_s = mri.vox2ras[3,3]/mri.zsize

  ic = [mri.width/2; mri.height/2; mri.depth/2; 1]
  c = mri.vox2ras * ic
  mri.c_r = c[1]
  mri.c_a = c[2]
  mri.c_s = c[3]

  #--------------------------------------------------------------------#
  # These are in the MATLAB version of this structure for convenience

  # 1-based vox2ras: Good for doing transforms on julia indices
  mri.vox2ras1 = vox2ras_0to1(mri.vox2ras)

  # Matrix of direction cosines
  mri.Mdc = mri.vox2ras[1:3, 1:3] *
            Diagonal(1 ./ [mri.xsize, mri.ysize, mri.zsize])

  # Vector of voxel resolutions
  mri.volres = [mri.xsize, mri.ysize, mri.zsize]

  mri.tkrvox2ras = vox2ras_tkreg(mri.volsize, mri.volres)
end


"""
    load_bruker(indir::String; headeronly::Bool=false, reco::Integer=1)

Read Bruker image data from disk and return an `MRI` structure similar to the
FreeSurfer MRI struct defined in mri.h.

# Arguments
- `indir::String`: Path to a Bruker scan directory (files called method, acqp,
pdata/1/reco, and pdata/1/2dseq are expected to be found in it)

# Optional arguments
- `headeronly::Bool=false`: If true, the pixel data are not read in.

- `reco::Integer=1`: Number of image reconstruction to load, where 1 denotes
  the online reconstruction and higher numbers denote any additional offline
  reconstructions performed after acquisition.
"""
function load_bruker(indir::String; headeronly::Bool=false, reco::Integer=1)

  dname = abspath(indir)
  methfile = dname * "/method" 
  acqpfile = dname * "/acqp" 
  recofile = dname * "/pdata/" * string(reco) * "/reco"
  visufile = dname * "/pdata/" * string(reco) * "/visu_pars"
  imgfile  = dname * "/pdata/" * string(reco) * "/2dseq"

  if any(.!isfile.([methfile, acqpfile, recofile, imgfile]))
    error("Input directory must contain the files: " *
          "method, acqp, pdata/" * string(reco) * "/reco, pdata/" *
                                   string(reco) * "/2dseq")
  end

  mri = MRI(Array{Float32, 4}(undef, (0, 0, 0, 0)))

  mri.fspec = imgfile
  mri.pwd = pwd()

  slicethick = 1
  nslice = 1
  nb0 = 0

  # Read information for the image header from the Bruker method file
  io = open(methfile, "r")

  while !eof(io)
    ln = readline(io)

    if startswith(ln, "##\$PVM_SpatResol=")		# Voxel size
      ln = readline(io)
      words = split(ln)
      mri.volres = parse.(Float32, words)
    elseif startswith(ln, "##\$PVM_Matrix=")		# Matrix size
      ln = readline(io)
      words = split(ln)
      mri.volsize = parse.(Float32, words)
    elseif startswith(ln, "##\$PVM_SliceThick=")	# Slice thickness (2D)
      ln = split(ln, "=")[2]
      slicethick = parse(Float32, ln)
    elseif startswith(ln, "##\$PVM_SPackArrNSlices=")	# Number of slices (2D)
      ln = readline(io)
      words = split(ln)
      nslice = sum(parse.(Float32, words))
    elseif startswith(ln, "##\$EchoTime=")		# TE
      ln = split(ln, "=")[2]
      mri.te = parse(Float32, ln)
    elseif startswith(ln, "##\$PVM_RepetitionTime=")	# TR
      ln = split(ln, "=")[2]
      mri.tr = parse(Float32, ln)
    elseif startswith(ln, "##\$PVM_DwAoImages=")	# Number of b=0 volumes
      ln = split(ln, "=")[2]
      nb0 = parse(Int64, ln)
    elseif startswith(ln, "##\$PVM_DwDir=")		# Diffusion gradients
      ln = split(ln, "(")[2]
      ln = split(ln, ")")[1]
      words = split(ln, ",")
      nval = prod(parse.(Int64, words))

      nread = 0
      bvec = Vector{Float32}(undef, 0)

      while nread < nval
        ln = readline(io)
        words = split(ln)

        nread += length(words)
	push!(bvec, parse.(Float32, words)...)
      end

      mri.bvec = permutedims(reshape(bvec, 3, :), [2 1])

      # Normalize gradient vectors
      mri.bvec .= mri.bvec ./ sqrt.(sum(mri.bvec.^2, dims=2))
      mri.bvec[isnan.(mri.bvec)] .= Float32(0)
    elseif startswith(ln, "##\$PVM_DwEffBval=")		# b-values
      ln = split(ln, "(")[2]
      ln = split(ln, ")")[1]
      nval = parse(Int64, ln)

      nread = 0
      bval = Vector{Float32}(undef, 0)

      while nread < nval
        ln = readline(io)
        words = split(ln)
        
        nread += length(words)
	push!(bval, parse.(Float32, words)...)
      end

      mri.bval = bval
    end
  end

  close(io)

  # Add b=0 volumes to the gradient table
  # (The method file includes them in the list of b-values but not vectors)
  for ib0 in 1:nb0
    mri.bvec = vcat([0 0 0], mri.bvec)
  end

  # For 2D scans, append resolution and matrix size in the slice dimension
  is2d = (length(mri.volres) == 2 && length(mri.volsize) == 2) ? true : false

  if is2d
    push!(mri.volres, slicethick)
    push!(mri.volsize, nslice)
  end

  # Read receiver gain from Bruker acqp file
  io = open(acqpfile, "r")

  gain = Float32(1)

  while !eof(io)
    ln = readline(io)

    if startswith(ln, "##\$RG=")			# Receiver gain
      ln = split(ln, "=")[2]
      gain = parse(Float32, ln)
    end
  end

  gain /= 64

  close(io)

  # Read information about image binary data from Bruker reco file
  io = open(recofile, "r")

  image_type = ""
  data_type = Int32
  int_offset = Vector{Float32}(undef, 0)
  int_slope = Vector{Float32}(undef, 0)
  byte_order = ""

  while !eof(io)
    ln = readline(io)

    if startswith(ln, "##\$RECO_image_type=")		# Complex, real, etc.
      image_type = split(ln, "=")[2]
    elseif startswith(ln, "##\$RECO_wordtype=")		# Bytes per voxel
      ln = split(ln, "=")[2]

      if ln == "_32BIT_FLOAT"
        data_type = Float32
      elseif ln == "_32BIT_SGN_INT"
        data_type = Int32
      elseif ln == "_16BIT_SGN_INT"
        data_type = Int16
      elseif ln == "_8BIT_UNSGN_INT"
        data_type = UInt8
      end
    elseif startswith(ln, "##\$RECO_map_offset=")	# Intensity offset
      ln = split(ln, "(")[2]
      ln = split(ln, ")")[1]
      nval = parse(Int64, ln)

      nread = 0

      while nread < nval
        ln = readline(io)

        if startswith(ln, "@" * string(nval))		# New since PV360
          ln = split(ln, "(")[2]
          ln = split(ln, ")")[1]
          vals = fill(parse(Float32, ln), nval)
        else
          vals = parse.(Float32, split(ln))
        end

        nread += length(vals)
        push!(int_offset, vals...)
      end
    elseif startswith(ln, "##\$RECO_map_slope")		# Intensity slope
      ln = split(ln, "(")[2]
      ln = split(ln, ")")[1]
      nval = parse(Int64, ln)

      nread = 0

      while nread < nval
        ln = readline(io)

        if startswith(ln, "@" * string(nval))		# New since PV360
          ln = split(ln, "(")[2]
          ln = split(ln, ")")[1]
          vals = fill(parse(Float32, ln), nval)
        else
          vals = parse.(Float32, split(ln))
        end

        nread += length(vals)
        push!(int_slope, vals...)
      end
    elseif startswith(ln, "##\$RECO_byte_order=")	# Byte order
      byte_order = split(ln, "=")[2]
    end
  end

  close(io)

  if image_type == "COMPLEX_IMAGE"
    # Real and imaginary frames share the same slope/offset
    append!(int_slope, int_slope)
    append!(int_offset, int_offset)
  end

  mri.image_type = image_type

  mri.nframes = is2d ? length(int_slope) ÷ nslice : length(int_slope)

  mri.vox2ras0 = Matrix(Diagonal([mri.volres; 1]))

  if headeronly
    return mri
  end

  # Read information about image from Bruker visu_pars file
  data_units = ""
  visu_size = Vector{Int32}(undef, 0)
  visu_order = Vector{Union{Nothing, Int64}}(undef, 0)

  if isfile(visufile)
    io = open(visufile, "r")

    while !eof(io)
      ln = readline(io)

      if startswith(ln, "##\$VisuCoreDataUnits=")	# Image units (if any)
        ln = readline(io)
        data_units = replace(ln, "<"=>"", ">"=>"")
      elseif startswith(ln, "##\$VisuCoreSize=")	# Image size
        ln = readline(io)
        words = split(ln)
        visu_size = parse.(Int32, words)
      elseif startswith(ln, "##\$VisuAcqGradEncoding")	# Encode dimensions
        ln = readline(io)
        words = split(ln)
        visu_order = indexin(["read_enc", "phase_enc", "slice_enc"], words)
      end
    end

    close(io)
  end

  # Read image data
  io = open(imgfile, "r")

  if isempty(visu_order) || visu_order == [1;2;3]	# Volume not permuted
    vol = read!(io, Array{data_type}(undef, (mri.volsize..., mri.nframes)))
  else							# Volume permuted
    if isempty(visu_size)
      visu_size = mri.volsize[[1, 2, 3][vol_order]]
    end
    vol = read!(io, Array{data_type}(undef, (visu_size..., mri.nframes)))
    vol = permutedims(vol, (visu_order..., 4))
  end

  close(io)

  if byte_order == "littleEndian"
    vol .= ltoh.(vol)
  else
    vol .= ntoh.(vol)
  end

  # Apply intensity offset and slope
  if data_type == Float32
    mri.vol = vol
  else
    mri.vol = Array{Float32}(undef, size(vol))

    if is2d	# One slope/offset per slice
      k = 1
      for iframe in 1:mri.nframes
        for islice in 1:mri.volsize[3]
          @views mri.vol[:,:,islice,iframe] .=
              Int32.(vol[:,:,islice,iframe]) ./ int_slope[k] .+ int_offset[k]
          k += 1
        end
      end
    else	# One slope/offset per volume
      for iframe in 1:mri.nframes
        @views mri.vol[:,:,:,iframe] .=
            Int32.(vol[:,:,:,iframe]) ./ int_slope[iframe] .+ int_offset[iframe]
      end
    end
  end

  # If it is a magnitude or complex image and it is unitless, normalize it 
  # by the receiver gain (this is useful, e.g., for combining DWI scans)
  if image_type != "PHASE_IMAGE" && isempty(data_units)
    mri.vol ./= gain
  end

  return mri
end


"""
    load_mgh(fname::String; slices::Union{Vector{Unsigned}, Nothing}=nothing, frames::Union{Vector{Unsigned}, Nothing}=nothing, headeronly::Bool=false)

Load a .mgh or .mgz file from disk.

Return:
- The image data as an array

- The 4x4 vox2ras matrix that transforms 0-based voxel indices to coordinates
  such that: [x y z]' = M * [i j k 1]'

- MRI parameters as a vector: [tr, flip_angle, te, ti]

- The image volume dimensions as a vector (useful when `headeronly` is true,
  in which case the image array will be empty)

# Arguments
- `fname::String`: Path to the .mgh/.mgz file

- `slices::Vector`: 1-based slice numbers to load (default: read all slices)

- `frames::Vector`: 1-based frame numbers to load (default: read all frames)

- `headeronly::Bool=false`: If true, the pixel data are not read in.
"""
function load_mgh(fname::String; slices::Union{Vector{Unsigned}, Nothing}=nothing, frames::Union{Vector{Unsigned}, Nothing}=nothing, headeronly::Bool=false)

  vol = Array{Float32, 4}(undef, 0, 0, 0, 0)
  M = Matrix{Float32}(undef, 0, 0)
  mr_parms = Vector{Float32}(undef, 0)
  volsz = Matrix{Int32}(undef, 0, 0)

  # Unzip if it is compressed
  ext = lowercase(fname[(end-1):end])

  if cmp(ext, "gz") == 0
    # Create unique temporary file name
    tmpfile = tempname(get_tmp_path()) * ".load_mgh.mgh"
    gzipped = true

    if Sys.isapple()
      cmd = `gunzip -c $fname`
    else
      cmd = `zcat $fname`
    end
    run(pipeline(cmd, stdout=tmpfile))
  else
    tmpfile = fname
    gzipped = false
  end

  io = open(tmpfile, "r")

  v       = ntoh(read(io, Int32))
  ndim1   = ntoh(read(io, Int32))
  ndim2   = ntoh(read(io, Int32))
  ndim3   = ntoh(read(io, Int32))
  nframes = ntoh(read(io, Int32))
  type    = ntoh(read(io, Int32))
  dof     = ntoh(read(io, Int32))

  if !isnothing(slices) && any(slices .> ndim3)
    error("Some slices=" * string(slices) * " exceed nslices=" * string(dim3))
  end

  if !isnothing(frames) && any(frames .> nframes)
    error("Some frames=" * string(frames) * " exceed nframes=" * string(nframes))
  end

  UNUSED_SPACE_SIZE = 256
  USED_SPACE_SIZE = 3*4+4*3*4		# Space for RAS transform

  unused_space_size = UNUSED_SPACE_SIZE-2 
  ras_good_flag = ntoh(read(io, Int16))

  if ras_good_flag > 0
    delta  = ntoh.(read!(io, Vector{Float32}(undef, 3)))
    Mdc    = ntoh.(read!(io, Vector{Float32}(undef, 9)))
    Mdc    = reshape(Mdc, (3,3))
    Pxyz_c = ntoh.(read!(io, Vector{Float32}(undef, 3)))

    D = Diagonal(delta)

    Pcrs_c = Float32.([ndim1; ndim2; ndim3])/2	# Should this be kept?

    Pxyz_0 = Pxyz_c - Mdc*D*Pcrs_c

    M = [Mdc*D Pxyz_0; 0 0 0 1]
    ras_xform = [Mdc Pxyz_c; 0 0 0 1]
    unused_space_size = unused_space_size - USED_SPACE_SIZE
  end

  skip(io, unused_space_size)
  nv = ndim1 * ndim2 * ndim3 * nframes
  volsz = [ndim1 ndim2 ndim3 nframes]

  MRI_UCHAR  =  0
  MRI_INT    =  1
  MRI_LONG   =  2
  MRI_FLOAT  =  3
  MRI_SHORT  =  4
  MRI_BITMAP =  5
  MRI_USHRT  = 10

  # Determine number of bytes per voxel
  if type == MRI_FLOAT
    nbytespervox = 4
    dtype = Float32
  elseif type == MRI_UCHAR
    nbytespervox = 1
    dtype = UInt8
  elseif type == MRI_SHORT
    nbytespervox = 2
    dtype = Int16
  elseif type == MRI_USHRT
    nbytespervox = 2
    dtype = UInt16
  elseif type == MRI_INT
    nbytespervox = 4
    dtype = Int32
  end

  if headeronly
    skip(io, nv*nbytespervox)
    if !eof(io)
      mr_parms = ntoh.(read!(io, Vector{Float32}(undef, 4)))
    end
    close(io)

    if gzipped		# Clean up
      cmd = `rm -f $tmpfile`
      run(cmd)
    end

    return vol, M, mr_parms, volsz
  end

  if isnothing(slices) && isnothing(frames)		# Read the entire volume
    vol = ntoh.(read!(io, Array{dtype}(undef, Tuple(volsz))))
  else						# Read a subset of slices/frames
    isnothing(frames) && (frames = 1:nframes)
    isnothing(slices) && (slices = 1:ndim3)

    nvslice = ndim1 * ndim2
    nvvol = nvslice * ndim3
    filepos0 = position(io)

    vol = zeros(dtype, ndim1, ndim2, length(slices), length(frames))

    for iframe in 1:length(frames)
      frame = frames[iframe]

      for islice in 1:length(slices)
        slice = slices[islice]

        filepos = ((frame-1)*nvvol + (slice-1)*nvslice)*nbytespervox + filepos0
        seek(io, filepos)

        vol[:, :, islice, iframe] = 
          ntoh.(read!(io, Array{dtype}(undef, Tuple(volsz[1:2]))))
      end
    end

    # Seek to just beyond the last slice/frame
    filepos = (nframes*nvvol)*nbytespervox + filepos0;
    seek(io, filepos)
  end

  if !eof(io)
    mr_parms = ntoh.(read!(io, Vector{Float32}(undef, 4)))
  end

  close(io)

  if gzipped		# Clean up
    cmd = `rm -f $tmpfile`
    run(cmd)
  end
  
  return vol, M, mr_parms, volsz
end


"""
    load_nifti_hdr(fname::String) -> NIfTIheader

Load the header of a .nii volume from disk.

Return a `NIfTIheader` structure, where units have been converted to mm and
msec, the sform and qform matrices have been computed and stored in the .sform
and .qform fields, and the .vox2ras field has been set to the sform (if valid),
then the qform (if valid).

Assume that the input file is uncompressed (compression is handled in the
wrapper load_nifti()).

Handle data with more than 32k cols by looking at the .dim field of the header.
If dim[2] = -1, then the .glmin field contains the number of columns. This is
FreeSurfer specific, for handling surfaces. When the total number of spatial
voxels equals 163842, then reshape the volume to 163842x1x1xnframes. This is
for handling the 7th order icosahedron used by FS group analysis.
"""
function load_nifti_hdr(fname::String)

  nifti_header_size = 348

  # Read header as vector of bytes
  io = open(fname, "r")
  buffer = read!(io, Vector{UInt8}(undef, nifti_header_size))
  close(io)

  typelist = NIfTIheader.types[1:end-4]
  plist = pointer(buffer) .+ cumsum(sizeof.(typelist))

  # Check endian-ness of first entry to decide if byte order should be reversed 
  headsize = unsafe_load(convert(Ptr{typelist[1]}, pointer(buffer)))

  if headsize == nifti_header_size
    do_bswap = false
    f = identity
  elseif headsize == bswap(Int32(nifti_header_size))
    do_bswap = true
    f = bswap
  else
    error("Invalid header size " * string(headsize) * " found in NIfTI header")
  end

  # Convert vector of bytes to array of header fields
  header = f.([(unsafe_load(convert(Ptr{typelist[i]}, plist[i-1])))
               for i = 2:length(typelist)])

  pushfirst!(header, f.(headsize))

  # "dim" field:
  # This is to accomodate structures with more than 32k cols
  # FreeSurfer specific (see also mriio.c)
  i_dim = findfirst(fieldnames(NIfTIheader) .== :dim)

  if header[i_dim][2] < 0
    i_glmin = findfirst(fieldnames(NIfTIheader) .== :glmin)
    header[i_dim] = (header[i_dim][1], header[i_glmin], header[i_dim][3:end]...)
    header[i_glmin] = 0
  end

  nspatial = prod(Int32.(header[i_dim][2:4]))
  if nspatial == 163842		# Ico7 surface
    header[i_dim] = (header[i_dim][1], 163842, 1, 1, header[i_dim][5:end]...)
  end

  # "xyz_units" field:
  # Find physical and time units of header fields
  i_xyzt_units = findfirst(fieldnames(NIfTIheader) .== :xyzt_units)
  
  xyzunits = header[i_xyzt_units] & Int8(7)	# Bitwise AND with 00000111
  if xyzunits == 1
    xyzscale = Float32(1000)	# meters
  elseif xyzunits == 2
    xyzscale = Float32(1)	# mm
  elseif xyzunits == 3
    xyzscale = Float32(.001)	# microns
  else
    println("WARNING: xyz units code " * string(xyzunits) * 
            " is unrecognized, assuming mm")
    xyzscale = Float32(1)	# just assume mm
  end

  tunits = header[i_xyzt_units] & Int8(56)	# Bitwise AND with 00111000
  if tunits == 8
    tscale = Float32(1000)	# seconds
  elseif tunits == 16
    tscale = Float32(1)		# msec
  elseif tunits == 32
    tscale = Float32(.001)	# microsec
  else
    tscale = Float32(0)		# no time scale
  end

  # "pixdim", "srow_x", "srow_y", "srow_z" fields:
  # Convert to physical units to mm and time units to msec
  i_pixdim = findfirst(fieldnames(NIfTIheader) .== :pixdim)
  header[i_pixdim] = (header[i_pixdim][1],
                      header[i_pixdim][2:4] .* xyzscale...,
                      header[i_pixdim][5] * tscale,
                      header[i_pixdim][6:8]...)

  i_srow_x = findfirst(fieldnames(NIfTIheader) .== :srow_x)
  header[i_srow_x] = header[i_srow_x] .* xyzscale

  i_srow_y = findfirst(fieldnames(NIfTIheader) .== :srow_y)
  header[i_srow_y] = header[i_srow_y] .* xyzscale

  i_srow_z = findfirst(fieldnames(NIfTIheader) .== :srow_z)
  header[i_srow_z] = header[i_srow_z] .* xyzscale

  # Change value in xyzt_units to reflect scale change
  header[i_xyzt_units] = Int8(2) | Int8(16)	# Bitwise OR of 2=mm, 16=msec

  # Sform matrix
  sform = [collect(header[i_srow_x])'; 
	   collect(header[i_srow_y])'; 
	   collect(header[i_srow_z])';
	   Float32.([0 0 0 1])]

  # Qform matrix
  # (From DG's load_nifti_hdr.m: not quite sure how all this works,
  # mainly just copied CH's code from mriio.c)
  i_quatern_b = findfirst(fieldnames(NIfTIheader) .== :quatern_b)
  i_quatern_c = findfirst(fieldnames(NIfTIheader) .== :quatern_c)
  i_quatern_d = findfirst(fieldnames(NIfTIheader) .== :quatern_d)
  i_quatern_x = findfirst(fieldnames(NIfTIheader) .== :quatern_x)
  i_quatern_y = findfirst(fieldnames(NIfTIheader) .== :quatern_y)
  i_quatern_z = findfirst(fieldnames(NIfTIheader) .== :quatern_z)
  b = header[i_quatern_b]
  c = header[i_quatern_c]
  d = header[i_quatern_d]
  x = header[i_quatern_x]
  y = header[i_quatern_y]
  z = header[i_quatern_z]
  a = Float32(1) - (b*b + c*c + d*d)
  if abs(a) < 1.0e-7
    a = Float32(1) / sqrt(b*b + c*c + d*d)
    b = b*a
    c = c*a
    d = d*a
    a = Float32(0)
  else
    a = sqrt(a)
  end
  r11 = a*a + b*b - c*c - d*d
  r12 = 2*b*c - 2*a*d
  r13 = 2*b*d + 2*a*c
  r21 = 2*b*c + 2*a*d
  r22 = a*a + c*c - b*b - d*d
  r23 = 2*c*d - 2*a*b
  r31 = 2*b*d - 2*a*c
  r32 = 2*c*d + 2*a*b
  r33 = a*a + d*d - c*c - b*b
  if header[i_pixdim][1] < 0.0
    r13 = -r13
    r23 = -r23
    r33 = -r33
  end
  qMdc = [r11 r12 r13; r21 r22 r23; r31 r32 r33]
  D = Diagonal(collect(header[i_pixdim][2:4]))
  P0 = [x y z]'
  qform = [qMdc*D P0; 0 0 0 1]

  # "sform_code", "qform_code" fields:
  # Determine which matrix to use as vox2ras
  i_sform_code = findfirst(fieldnames(NIfTIheader) .== :sform_code)
  i_qform_code = findfirst(fieldnames(NIfTIheader) .== :qform_code)

  if header[i_sform_code] != 0
    # Use sform first
    vox2ras = sform
  elseif header[i_qform_code] != 0
    # Then use qform first
    vox2ras = qform
  else
    println("WARNING: neither sform or qform are valid in " * fname)
    D = Diagonal(collect(header[i_pixdim][2:4]))
    P0 = [0 0 0]'
    vox2ras = [D P0; 0 0 0 1]
  end

  return NIfTIheader(header..., do_bswap, sform, qform, vox2ras)
end


"""
    load_nifti(fname::Stringr; headeronly::Bool=false) -> NIfTIheader, Array

Load a NIfTI (.nii or .nii.gz) volume from disk and return an array containing
the image data and a `NIfTIheader` structure and the image data as an array.

Handle compressed NIfTI (nii.gz) by issuing an external Unix call to
uncompress the file to a temporary file, which is then deleted.

The output `NIfTIheader` structure contains:
- the units for each dimension of the volume [mm or msec], in the .pixdim field
- the sform and qform matrices, in the .sform and .qform fields
- the vox2ras matrix, which is the sform (if valid), otherwise the qform, in
  the .vox2ras field
"""
function load_nifti(fname::String; headeronly::Bool=false)

  # Unzip if it is compressed 
  ext = lowercase(fname[(end-1):end])

  if cmp(ext, "gz") == 0
    # Create unique temporary file name
    tmpfile = tempname(get_tmp_path()) * ".load_nifti.nii"
    gzipped = true

    if Sys.isapple()
      cmd = `gunzip -c $fname`
    else
      cmd = `zcat $fname`
    end
    run(pipeline(cmd, stdout=tmpfile))
  else
    tmpfile = fname
    gzipped = false
  end

  # Read NIfTI header
  hdr = load_nifti_hdr(tmpfile)

  # Get volume dimensions
  dim = hdr.dim[2:findlast(hdr.dim .!= 0)]

  # Get data type
  if hdr.datatype == 2
    dtype = UInt8
  elseif hdr.datatype == 4
    dtype = Int16
  elseif hdr.datatype == 8
    dtype = Int32
  elseif hdr.datatype == 16
    dtype = Float32
  elseif hdr.datatype == 64
    dtype = Float64
  elseif hdr.datatype == 256
    dtype = Int8
  elseif hdr.datatype == 512
    dtype = UInt16
  elseif hdr.datatype == 768
    dtype = UInt32
  else
    close(io)
    if gzipped		# Clean up
      cmd = `rm -f $tmpfile`
      run(cmd)
    end
    error("Data type " * string(hdr.datatype) * " not supported")
  end

  vol = Array{dtype, length(dim)}(undef, zeros(Int, length(dim))...)

  # If only header is desired or if there is no vox2ras, return now
  if headeronly || isempty(hdr.vox2ras)
    if gzipped		# Clean up
      cmd = `rm -f $tmpfile`
      run(cmd)
    end
    return hdr, vol
  end

  # Open to read the pixel data
  io = open(tmpfile, "r")

  # Get past the header
  seek(io, Int64(round(hdr.vox_offset)))

  vol = read!(io, Array{dtype}(undef, dim))

  close(io)
  if gzipped		# Clean up
    cmd = `rm -f $tmpfile`
    run(cmd)
  end

  # Check if end-of-file was reached
  if !eof(io)
    error(tmpfile * ", read a " * string(size(vol)) *
          " volume but did not reach end of file")
  end

  # If needed, reverse order of bytes to correct endian-ness
  if hdr.do_bswap
    vol .= bswap.(vol)
  end

  if hdr.scl_slope != 0 && !(hdr.scl_inter == 0 && hdr.scl_slope == 1)
    # Rescaling is not needed if the slope==1 and intersect==0,
    # skipping this preserves the numeric class of the data
    vol .= dtype.(vol .* hdr.scl_slope .+ hdr.scl_inter)
  end

  return hdr, vol
end


"""
    mri_write(mri::MRI, outfile::String, datatype::DataType=Float32)

Write an MRI volume to disk. Return true is an error occurred (i.e., the
number of bytes written were not as expected based on the size of the volume).

# Arguments
- `mri::MRI`: A structure like that returned by `mri_read()`. The geometry
  (i.e., direction cosines, voxel resolution, and P0) are all recomputed from
  mri.vox2ras0. So, if a method has changed one of the other fields, e.g.,
  mri.x_r, this change will not be reflected in the output volume.

- `outfile::String`: Path to the output file, which can be:
  1. An MGH file, e.g., f.mgh or f.mgz (uncompressed or compressed)
  2. A NIfTI file, e.g., f.nii or f.nii.gz (uncompressed or compressed).

- `datatype::DataType=eltype(mri.vol)`: Only applies to NIfTI and can be UInt8,
  Int16, Int32, Float32, Float64, Int8, UInt16, UInt32. By default, the native
  data type of the volume array is preserved when writing to disk.
""" 
function mri_write(mri::MRI, outfile::String, datatype::DataType=eltype(mri.vol))

  err = true

  if isempty(mri.vol)
    error("Input structure has empty vol field")
  end

  vsz = collect(size(mri.vol))
  nvsz = length(vsz)
  if nvsz < 4
    vsz = [vsz; ones(eltype(vsz), 4-nvsz)]
  end

  if isempty(mri.volsize)
    mri.volsize = vsz[1:3]
  end

  if mri.nframes == 0
    mri.nframes = vsz[4]
  end

  if isempty(mri.vox2ras0)
    mri.vox2ras0 = Diagonal(ones(4))
  end

  if isempty(mri.volres)
    mri.volres = sqrt.((ones(3)' * (mri.vox2ras0[1:3,1:3].^2))')
  end

  (fname, fstem, fext) = mri_filename(outfile, false)	# false = no checkdisk
  if isempty(fname) 
    error("Cannot determine format of " * outfile)
  end

  if any(cmp.(fext, ["mgh", "mgz"]) .== 0)		#-------- MGH --------#
    M = mri.vox2ras0
    mr_parms = [mri.tr, mri.flip_angle, mri.te, mri.ti]

    if mri.ispermuted
      err = save_mgh(permutedims(mri.vol, [2; 1; 3:ndims(mri.vol)]),
                     fname, M, mr_parms)
    else
      err = save_mgh(mri.vol, fname, M, mr_parms)
    end
  elseif any(cmp.(fext, ["nii", "nii.gz"]) .== 0)	#------- NIfTI -------#
    hdr_sizeof_hdr    = Int32(348)
    hdr_data_type     = Tuple(zeros(UInt8, 10))
    hdr_db_name       = Tuple(zeros(UInt8, 18))
    hdr_extents       = Int32(0)
    hdr_session_error = Int16(0)
    hdr_regular       = UInt8(0)
    hdr_dim_info      = UInt8(0)

    dim = ones(Int16, 8)
    dim[1] = mri.nframes > 1 ? 4 : 3
    dim[2:4] = mri.ispermuted ? mri.volsize[[2,1,3]] : mri.volsize[1:3]
    dim[5] = mri.nframes

    # This is to accomodate structures with more than 32k cols
    # FreeSurfer specific. See also mriio.c.
    if dim[2] > 2^15
      hdr_glmin = dim[2]
      dim[2] = -1
    end

    hdr_dim = Tuple(dim)

    hdr_intent_p1   = Float32(0)
    hdr_intent_p2   = Float32(0)
    hdr_intent_p3   = Float32(0)
    hdr_intent_code = Int16(0)

    if datatype == UInt8
      hdr_datatype = Int16(2)
      hdr_bitpix   = Int16(8)
    elseif datatype == Int16
      hdr_datatype = Int16(4)
      hdr_bitpix   = Int16(16)
    elseif datatype == Int32
      hdr_datatype = Int16(8)
      hdr_bitpix   = Int16(32)
    elseif datatype == Float32
      hdr_datatype = Int16(16)
      hdr_bitpix   = Int16(32)
    elseif datatype == Float64
      hdr_datatype = Int16(64)
      hdr_bitpix   = Int16(64)
    elseif datatype == Int8
      hdr_datatype = Int16(256)
      hdr_bitpix   = Int16(8)
    elseif datatype == UInt16
      hdr_datatype = Int16(512)
      hdr_bitpix   = Int16(16)
    elseif datatype == UInt32
      hdr_datatype = Int16(768)
      hdr_bitpix   = Int16(32)
    else
      error("Data type " * string(datatype) * " not supported")
    end

    hdr_slice_start = Int16(0)

    if mri.ispermuted
      hdr_pixdim = Float32.((0, mri.volres[[2,1,3]]..., mri.tr, 0, 0, 0))
    else
      hdr_pixdim = Float32.((0, mri.volres[1:3]..., mri.tr, 0, 0, 0))
    end

    hdr_vox_offset = Float32(352)
    hdr_scl_slope  = mri.niftihdr.scl_slope
    hdr_scl_inter  = mri.niftihdr.scl_inter
    hdr_slice_end  = Int16(0)
    hdr_slice_code = Int8(0)
  
    hdr_xyzt_units = Int8(2) | Int8(16)   # Bitwise OR of 2=mm, 16=msec

    hdr_cal_max        = maximum(mri.vol)
    hdr_cal_min        = minimum(mri.vol)
    hdr_slice_duration = Float32(0)
    hdr_toffset        = Float32(0)
    hdr_glmax          = Int32(0)
    if dim[2] != -1
      hdr_glmin        = Int32(0)
    end
    hdr_descrip        = UInt8.(Tuple(@sprintf("%-80s","FreeSurfer julia")))
    hdr_aux_file       = Tuple(zeros(UInt8, 24))
    hdr_qform_code     = Int16(1)		# 1=NIFTI_XFORM_SCANNER_ANAT
    hdr_sform_code     = Int16(1)		# 1=NIFTI_XFORM_SCANNER_ANAT
  
    # Qform (must have 6 DOF)
    (b, c, d, x, y, z, qfac) = vox2ras_to_qform(mri.vox2ras0)
    hdr_pixdim    = (Float32(qfac), hdr_pixdim[2:end]...)
    hdr_quatern_b = Float32(b)
    hdr_quatern_c = Float32(c)
    hdr_quatern_d = Float32(d)
    hdr_quatern_x = Float32(x)
    hdr_quatern_y = Float32(y)
    hdr_quatern_z = Float32(z)

    # Sform (can be any affine)
    hdr_srow_x = Tuple(Float32.(mri.vox2ras0[1,:]))
    hdr_srow_y = Tuple(Float32.(mri.vox2ras0[2,:]))
    hdr_srow_z = Tuple(Float32.(mri.vox2ras0[3,:]))

    hdr_intent_name = (UInt8.(collect("huh?"))..., zeros(UInt8, 12)...)
    hdr_magic       = Tuple(collect("n+1\0"))

    hdr = NIfTIheader( hdr_sizeof_hdr,
                       hdr_data_type,
                       hdr_db_name,
                       hdr_extents,
                       hdr_session_error,
                       hdr_regular,
                       hdr_dim_info,
                       hdr_dim,
                       hdr_intent_p1,
                       hdr_intent_p2,
                       hdr_intent_p3,
                       hdr_intent_code,
                       hdr_datatype,
                       hdr_bitpix,
                       hdr_slice_start,
                       hdr_pixdim,
                       hdr_vox_offset,
                       hdr_scl_slope,
                       hdr_scl_inter,
                       hdr_slice_end,
                       hdr_slice_code,
                       hdr_xyzt_units,
                       hdr_cal_max,
                       hdr_cal_min,
                       hdr_slice_duration,
                       hdr_toffset,
                       hdr_glmax,
                       hdr_glmin,
                       hdr_descrip,
                       hdr_aux_file,
                       hdr_qform_code,
                       hdr_sform_code,
                       hdr_quatern_b,
                       hdr_quatern_c,
                       hdr_quatern_d,
                       hdr_quatern_x,
                       hdr_quatern_y,
                       hdr_quatern_z,
                       hdr_srow_x,
                       hdr_srow_y,
                       hdr_srow_z,
                       hdr_intent_name,
                       hdr_magic,
                       UInt8(0),
                       Matrix{Float32}(undef, 0, 0),
                       Matrix{Float32}(undef, 0, 0),
                       Matrix{Float32}(undef, 0, 0) )

    if mri.ispermuted
      vol = permutedims(mri.vol, [2; 1; 3:ndims(mri.vol)])
    else
      vol = mri.vol
    end

    err = save_nifti(hdr, vol, fname)
  else
    error("File extension " * fext * " not supported")
  end

  if err
    println("WARNING: Problem saving " * outfile)
  end

  # Optional DWI tables -----------------------------------------------#

  if !isempty(mri.bval)
    bfile = fstem * ".bvals"
    writedlm(bfile, mri.bval, ' ')
  end

  if !isempty(mri.bvec)
    gfile = fstem * ".bvecs"
    writedlm(gfile, mri.bvec, ' ')
  end

  return err
end


"""
    save_mgh(vol::Array, fname::String, M::Matrix=Diagonal(ones(4)), mr_parms::Vector=zeros(4))

Write an MRI volume to a .mgh or .mgz file. Return true is an error occurred
(i.e., the number of bytes written were not as expected based on the size of
the volume).

# Arguments
- `vol::Array`: the image data

- `fname::String`: path to the output file

- `M::Matrix`: the 4x4 vox2ras transform such that [x y z]' = M * [i j k 1]',
  where the voxel indices (i, j, k) are 0-based.

- `mr_parms::Vector`: a vector of MRI parameters, [tr, flip_angle, te, ti]
"""
function save_mgh(vol::Array, fname::String, M::Matrix=Diagonal(ones(4)), mr_parms::Vector=zeros(4))

  if size(M) != (4, 4)
    error("M size=" * string(size(M)) * ", must be (4, 4)")
  end

  if length(mr_parms) != 4
    error("mr_parms length=" * string(length(mr_parms)) * ", must be 4")
  end

  (ndim1, ndim2, ndim3, frames) = size(vol)

  dtype = eltype(vol)

  # The MATLAB version ingores these and always writes the data as float
  # Here we use them properly
  MRI_UCHAR =  0
  MRI_INT =    1
  MRI_LONG =   2
  MRI_FLOAT =  3
  MRI_SHORT =  4
  MRI_BITMAP = 5
  MRI_TENSOR = 6
  MRI_USHRT  = 10

  # Determine number of bytes per voxel
  if dtype == Float32
    type = MRI_FLOAT
  elseif dtype == UInt8
    type = MRI_UCHAR
  elseif dtype == Int32
    type = MRI_INT
  elseif dtype == Int64
    type = MRI_LONG
  elseif dtype == Int16
    type = MRI_SHORT
  elseif dtype == UInt16
    type = MRI_USHRT
  end

  io = open(fname, "w")

  nb = 0

  # Write everything as big-endian
  nb += write(io, hton(Int32(1)))		# magic number
  nb += write(io, hton(Int32(ndim1)))
  nb += write(io, hton(Int32(ndim2)))
  nb += write(io, hton(Int32(ndim3)))
  nb += write(io, hton(Int32(frames)))
  nb += write(io, hton(Int32(type)))

  nb += write(io, hton(Int32(1)))		# dof (not used)

  UNUSED_SPACE_SIZE = 256
  USED_SPACE_SIZE = (3*4+4*3*4)		# Space for RAS transform

  MdcD = M[1:3,1:3]

  delta = sqrt.(sum(MdcD.^2; dims=1))
  Mdc = MdcD ./ repeat(delta, 3)

  Pcrs_c = [ndim1/2, ndim2/2, ndim3/2, 1]
  Pxyz_c = M*Pcrs_c
  Pxyz_c = Pxyz_c[1:3]

  nb += write(io, hton(Int16(1)))		# ras_good_flag = 1
  nb += write(io, hton.(Float32.(delta)))
  nb += write(io, hton.(Float32.(Mdc)))
  nb += write(io, hton.(Float32.(Pxyz_c)))

  unused_space_size = UNUSED_SPACE_SIZE-2
  unused_space_size = unused_space_size - USED_SPACE_SIZE
  nb += write(io, UInt8.(zeros(unused_space_size)))

  nb += write(io, hton.(vol))

  nb += write(io, hton.(Float32.(mr_parms)))

  close(io)

  err = (nb != (sizeof(Int32) * 7 +
                sizeof(Int16) + 
                sizeof(UInt8) * unused_space_size +
                sizeof(Float32) * 19 +
                sizeof(dtype) * length(vol)))

  ext = lowercase(fname[(end-1):end])

  if cmp(ext, "gz") == 0
    cmd = `gzip -f $fname`
    run(cmd)
    cmd = `mv $fname.gz $fname`
    run(cmd)
  end

  return err
end


"""
    save_nifti(hdr::NIfTIheader, vol::Array{T}, fname::String) where T<:Number

Write an MRI volume to a .nii or .nii.gz file. Return true is an error occurred
(i.e., the number of bytes written were not as expected based on the size of
the volume).

# Arguments
- `hdr::NIfTIheader`: a NIfTI header structure

- `vol::Array{T}`: an array that contains the image data

- `fname::String`: path to the output file

Handle data structures with more than 32k cols by setting hdr.dim[2] = -1 and
hdr.glmin = ncols. This is FreeSurfer specific, for handling surfaces. The
exception to this is when the total number of spatial voxels equals 163842,
then the volume is reshaped to 27307x1x6xnframes. This is for handling the 7th
order icosahedron used by FS group analysis.
"""
function save_nifti(hdr::NIfTIheader, vol::Array{T}, fname::String) where T<:Number

  ext = lowercase(fname[(end-1):end])
  if cmp(ext, "gz") == 0
    gzip_needed = true
    fname = fname[1:(end-3)]
  else
    gzip_needed = false
  end

  # Check for ico7
  sz = size(vol)
  if sz[1] == 163842
    dim = (27307, 1, 6, size(vol,4))
    vol = reshape(vol, dim)
  end

  io = open(fname, "w")

  nb = 0

  nb += write(io, hdr.sizeof_hdr)
  nb += write(io, collect(hdr.data_type))
  nb += write(io, collect(hdr.db_name))
  nb += write(io, hdr.extents)
  nb += write(io, hdr.session_error)
  nb += write(io, hdr.regular)
  nb += write(io, hdr.dim_info)
  nb += write(io, collect(hdr.dim))
  nb += write(io, hdr.intent_p1)
  nb += write(io, hdr.intent_p2)
  nb += write(io, hdr.intent_p3)
  nb += write(io, hdr.intent_code)
  nb += write(io, hdr.datatype)
  nb += write(io, hdr.bitpix)
  nb += write(io, hdr.slice_start)
  nb += write(io, collect(hdr.pixdim))
  nb += write(io, hdr.vox_offset)
  nb += write(io, hdr.scl_slope)
  nb += write(io, hdr.scl_inter)
  nb += write(io, hdr.slice_end)
  nb += write(io, hdr.slice_code)
  nb += write(io, hdr.xyzt_units)
  nb += write(io, hdr.cal_max)
  nb += write(io, hdr.cal_min)
  nb += write(io, hdr.slice_duration)
  nb += write(io, hdr.toffset)
  nb += write(io, hdr.glmax)
  nb += write(io, hdr.glmin)
  nb += write(io, collect(hdr.descrip))
  nb += write(io, collect(hdr.aux_file))
  nb += write(io, hdr.qform_code)
  nb += write(io, hdr.sform_code)
  nb += write(io, hdr.quatern_b)
  nb += write(io, hdr.quatern_c)
  nb += write(io, hdr.quatern_d)
  nb += write(io, hdr.quatern_x)
  nb += write(io, hdr.quatern_y)
  nb += write(io, hdr.quatern_z)
  nb += write(io, collect(hdr.srow_x))
  nb += write(io, collect(hdr.srow_y))
  nb += write(io, collect(hdr.srow_z))
  nb += write(io, collect(hdr.intent_name))
  nb += write(io, collect(hdr.magic))

  # Pad to get to 352 bytes (header size is 348)
  nb += write(io, zeros(UInt8,4))

  if hdr.datatype == 2
    dtype = UInt8
  elseif hdr.datatype == 4
    dtype = Int16
  elseif hdr.datatype == 8
    dtype = Int32
  elseif hdr.datatype == 16
    dtype = Float32
  elseif hdr.datatype == 64
    dtype = Float64
  elseif hdr.datatype == 256
    dtype = Int8
  elseif hdr.datatype == 512
    dtype = UInt16
  elseif hdr.datatype == 768
    dtype = UInt32
  else
    println("WARNING: data type " * string(hdr.datatype) *
            " not supported, but writing as float")
    dtype = Float32
  end

  nb += write(io, dtype.(vol))

  close(io)

  err = (nb != (sizeof(Float32) * 36 +
                sizeof(Int32) * 4 +
                sizeof(Int16) * 16 +
                sizeof(Int8) * 2 +
                sizeof(UInt8) * 158 +
                sizeof(dtype) * length(vol)))

  if gzip_needed
    cmd = `gzip -f $fname`
    run(cmd)
  end

  return err
end


"""
    mri_read_bfiles(infile1::String, infile2::String)

Read a DWI b-value table and gradient table from text files `infile1` and
`infile2`. The two input files can be specified in any order. The gradient
table file must contain 3 times as many entries as the b-value table file.

Return the b-value table as a vector of size n and the gradient table as a
matrix of size (n, 3).
"""
function mri_read_bfiles(infile1::String, infile2::String)

  tab = Vector{Matrix{Float32}}(undef, 0)

  for infile in (infile1, infile2)
    if !isfile(infile)
      error("Could not open " * infile)
    end

    push!(tab, readdlm(infile, Float32))

    if !all(isa.(tab[end], Number))
      error("File " * infile * " contains non-numeric entries")
    end
  end

  ival, ivec = (length(tab[1]) < length(tab[2])) ? (1, 2) : (2, 1)

  # Convert b-value table to single column
  if size(tab[ival], 2) != 1
    if size(tab[ival], 1) != 1
      error("Wrong format in table " * (ival == 1 ? infile1 : infile2) *
            " (should be single column or row)")
    else
      tab[ival] = permutedims(tab[ival], [2,1])
    end
  end

  # Convert gradient table to three columns
  if size(tab[ivec], 2) != 3
    if size(tab[ivec], 1) != 3
      error("Wrong format in table " * (ivec == 1 ? infile1 : infile2) *
            " (should be three columns or rows)")
    else
      tab[ivec] = permutedims(tab[ivec], [2,1])
    end
  end

  if size(tab[1], 1) != size(tab[2], 1)
    error("Dimension mismatch between tables in " *
          infile1 * " " * string(size(tab[1])) * " and " *
          infile2 * " " * string(size(tab[2])))
  end

  # Return b-value table as vector and gradient table as matrix
  if ival == 1
    return tab[1][:,1], tab[2]
  else
    return tab[1], tab[2][:,1]
  end
end


"""
    mri_read_bfiles!(dwi::MRI, infile1::String, infile2::String)

Set the .bval and .bvec fields of the MRI structure `dwi`, by reading a DWI
b-value table and gradient table from the text files `infile1` and `infile2`.
The two input files can be specified in any order. The gradient table file
must contain 3 times as many entries as the b-value table file.

Return the b-value table as a vector of size n and the gradient table as a
matrix of size (n, 3).
"""
function mri_read_bfiles!(dwi::MRI, infile1::String, infile2::String)

  (tab1, tab2) = mri_read_bfiles(infile1, infile2)

  if size(tab1, 1) != size(dwi.vol, 4)
    error("Number of frames in volume (" * string(size(dwi.vol, 4)) *
          ") does not match dimensions of table in " *
          infile1 * " " * string(size(tab1)))
  end

  if size(tab1, 2) == 1
    dwi.bval = tab1
    dwi.bvec = tab2
  else
    dwi.bval = tab2
    dwi.bvec = tab1
  end

  # Normalize gradient vectors
  dwi.bvec = dwi.bvec ./ sqrt.(sum(dwi.bvec.^2, dims=2))
  dwi.bvec[isnan.(dwi.bvec)] .= Float32(0)

  return tab1, tab2
end



"""
    mri_read(inbase::String, type::DataType; headeronly::Bool=false, permutedata::Bool=false)

Read a set of image files with base name `inbase`, which were generated by
an analysis, into a struct of a custom type (e.g., `DTI`)
"""
function mri_read(inbase::String, type::DataType; headeronly::Bool=false, permutedata::Bool=false)

  if !isstructtype(type)
    error("Type " * type * " is not a structure")
  end

  absbase = abspath(inbase)
  flist = readdir(dirname(absbase), join=true)

  inputs = []

  for var in fieldnames(type)
    ftype = fieldtype.(type, var)

    if ftype == MRI
      infile = absbase * "_" * string(var) * ".nii.gz"
      push!(inputs,
            mri_read(infile; headeronly=headeronly, permutedata=permutedata))
    elseif ftype == Vector{MRI}
      inpat = Regex("^" * absbase * "_" * string(var) * "[0-9]*.nii.gz\$")
      infiles = flist[.!isnothing.(match.(inpat, flist))]
      push!(inputs,
            mri_read.(infiles; headeronly=headeronly, permutedata=permutedata))
    else
      infile = absbase * "_" * string(var) * ".txt"
      mat = Float32.(readdlm(infile))
      if length(mat) == 1
        push!(inputs, mat[1])
      else
        push!(inputs, mat)
      end
    end
  end

  return type(inputs...)
end


