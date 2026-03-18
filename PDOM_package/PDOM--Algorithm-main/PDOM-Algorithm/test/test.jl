using DrWatson

include(srcdir("test_fun.jl"))

Normal_Nonconvex_LASSO(100,200,2,0.0)

Normal_Nonconvex_RPCA(100,5,0.05)

Standard_Nonconvex_LASSO(100,200,2,0.0)

Standard_Nonconvex_RPCA(100,5,0.05)