module Fibers

include("mri.jl")
include("util.jl")
include("trk.jl")
include("show.jl")
#include("view.jl")
include("dti.jl")
include("odf.jl")
include("dsi.jl")
include("gqi.jl")
include("rusd.jl")
include("stream.jl")

function __init__()

#  println("FREESURFER_HOME: " * (haskey(ENV, "FREESURFER_HOME") ?
#                                 ENV["FREESURFER_HOME"] : "not defined"))
end

end # module
