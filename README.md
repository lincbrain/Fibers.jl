<img src="https://user-images.githubusercontent.com/15318615/158502382-3d47c48b-e991-400b-8f7c-e745f32e9643.png" width=800>

### Import this package

If you want julia (and this package) to use multi-threading for faster computation, you have to explicitly tell it to. You can do that either by starting julia  with ```julia --thread auto```, or by setting the environment variable ```JULIA_NUM_THREADS``` to ```auto``` before starting julia. (This will use the total number of CPU threads available on your computer. Alternatively, you can specify a number of threads in place of ```auto```.)

```julia
julia> import FreeSurfer as fs
```

### Read .mgh, .mgz, .nii, .nii.gz volumes

```julia
julia> aa = fs.mri_read("/usr/local/freesurfer/dev/subjects/fsaverage/mri/aparc+aseg.mgz");

julia> fa = fs.mri_read("/usr/local/freesurfer/dev/trctrain/hcp/MGH35_HCP_FA_template.nii.gz");
```

### Display volume and header summary info in the terminal

```julia
julia> fs.disp(aa)

julia> fs.disp(fa)
```

### Display a slice view of the volume in a pop-up window

By default this will be an axial view, middle slice, first frame. See ```?fs.show``` for how to change what is displayed.

```julia
julia> fs.show(aa)

julia> fs.show(fa)
```

### Write .mgh, .mgz, .nii, .nii.gz volumes

```julia
julia> fs.mri_write(aa, "/tmp/aparc+aseg.nii.gz")

julia> fs.mri_write(fa, "/tmp/MGH35_HCP_FA_template.mgz")
```

### Read Bruker scan directories

```julia
julia> ph = fs.mri_read("/opt/nmrdata/PV-7.0.0/ayendiki/Phantom.cO1/5/");
```

### Read a .trk tractography streamline file

```julia
julia> tr = fs.trk_read("/usr/local/freesurfer/dev/trctrain/hcp/mgh_1001/syn/acomm.bbr.prep.trk");
```

### Write a .trk tractography streamline file

```julia
julia> fs.trk_write(tr, "/tmp/acomm.trk")
```

### Tutorial

[A gentle introduction to diffusion MRI](https://github.com/freesurfer/FreeSurfer.jl/blob/master/docs/tutorial.ipynb)

