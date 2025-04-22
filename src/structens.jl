#=
  Structure tensor reconstruction
=#

using LinearAlgebra, StaticArrays, ImageFiltering

export st_eigen, st_recon


"""
    st_eigen{T}
"""
function st_eigen(Sxx::Array{T, 3}, Sxy::Array{T, 3},
                  Sxz::Array{T, 3}, Syy::Array{T, 3},
                  Syz::Array{T, 3}, Szz::Array{T, 3}) where T<:AbstractFloat

  eigvec = Array{T, 5}(undef, size(Sxx)..., 3, 3)
  eigval = Array{T, 4}(undef, size(Sxx)..., 3)

  Threads.@threads for iz in axes(Sxx, 3)
    for iy in axes(Sxx, 2)
      for ix in axes(Sxx, 1)
        S = @SMatrix [ Sxx[ix, iy, iz] 0        0;
                       Sxy[ix, iy, iz] Syy[ix, iy, iz] 0;
                       Sxz[ix, iy, iz] Syz[ix, iy, iz] Szz[ix, iy, iz] ]
        eig = eigen(Symmetric(S, :L))
        eigvec[ix, iy, iz, :, :] .= eig.vectors
        eigval[ix, iy, iz, :] .= eig.values
      end
    end
  end

  return eigvec, eigval
end


"""
    st_recon{T}(vol::Array{T, 3}, sigma::Number, rho::Number)
"""
function st_recon(vol::Array{T, 3}, sigma::Number, rho::Number) where T<:AbstractFloat

  if (sigma > 0)
    println("Smoothing image")
    image = imfilter(T, vol,
                     KernelFactors.gaussian((sigma, sigma, sigma)), "reflect")
  else
    image = vol
  end

  println("Computing gradients")
  gx = imfilter(T, image,
                KernelFactors.scharr((true, true, true), 1), "reflect")
  gy = imfilter(T, image,
                KernelFactors.scharr((true, true, true), 2), "reflect")
  gz = imfilter(T, image,
                KernelFactors.scharr((true, true, true), 3), "reflect")

  println("Computing structure tensor")
  gxx = gx .* gx
  gxy = gx .* gy
  gxz = gx .* gz
  gyy = gy .* gy
  gyz = gy .* gz
  gzz = gz .* gz

  gx = gy = gz = nothing

  if (rho > 0)
    println("Smoothing structure tensor")
    gxx = imfilter(T, gxx,
                   KernelFactors.gaussian((rho, rho, rho)), "reflect")
    gxy = imfilter(T, gxy,
                   KernelFactors.gaussian((rho, rho, rho)), "reflect")
    gxz = imfilter(T, gxz,
                   KernelFactors.gaussian((rho, rho, rho)), "reflect")
    gyy = imfilter(T, gyy,
                   KernelFactors.gaussian((rho, rho, rho)), "reflect")
    gyz = imfilter(T, gyz,
                   KernelFactors.gaussian((rho, rho, rho)), "reflect")
    gzz = imfilter(T, gzz,
                   KernelFactors.gaussian((rho, rho, rho)), "reflect")
  end

  println("Performing eigen-decomposition")
  eigvec, eigval = st_eigen(gxx, gxy, gxz, gyy, gyz, gzz)

  return eigvec, eigval
end


