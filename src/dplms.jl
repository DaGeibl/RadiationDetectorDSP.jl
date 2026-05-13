# src/dplms_filter.jl

"""
    DPLMSChargeFilter

DPLMS (Deconvolution-based Penalized Least Mean Squares) optimum filter.

Computes the optimum FIR filter kernel from a noise matrix and reference signal
according to the method described in:

V. D'Andrea et al., "Optimum Filter Synthesis with DPLMS Method for Energy
Reconstruction", Eur. Phys. J. C 83, 149 (2023).
https://doi.org/10.1140/epjc/s10052-023-11299-z

Constructors:
* `DPLMSChargeFilter(; fields...)`

Fields:
* `noise_matrix`: noise covariance matrix (n×n)
* `reference`: reference signal vector
* `a1`: penalized coefficient for the noise matrix (must be > 0)
* `a2`: penalized coefficient for the reference matrix (must be > 0)
* `a3`: penalized coefficient for the zero-area matrix (must be > 0)
* `ff`: flat-top selector for the reference signal (0 or 1)
* `length`: total length of the filter kernel
"""
Base.@kwdef struct DPLMSChargeFilter{
    M<:AbstractMatrix{<:AbstractFloat},
    R<:AbstractVector{<:AbstractFloat},
    T<:AbstractFloat
} <: AbstractRadFIRFilter
    noise_matrix::M
    reference::R
    a1::T = 50.0
    a2::T = 0.1
    a3::T = 1.0
    ff::Int = 1
    length::Int = size(noise_matrix, 1)
end

export DPLMSChargeFilter

Adapt.adapt_structure(to, flt::DPLMSChargeFilter) = flt

function fltinstance(flt::DPLMSChargeFilter, fi::SamplingInfo)
    fltinstance(ConvolutionFilter(flt), fi)
end

function ConvolutionFilter(flt::DPLMSChargeFilter)
    coeffs = dplms_filter_coeffs(
        flt.noise_matrix, flt.reference,
        flt.a1, flt.a2, flt.a3, flt.ff
    )
    ConvolutionFilter(FFTConvolution(), coeffs)
end



function dplms_filter_coeffs(
    noise_matrix::AbstractMatrix{U},
    reference::AbstractVector{V},
    a1::W, a2::W, a3::W,
    ff::Int
) where {U<:AbstractFloat, V<:AbstractFloat, W<:AbstractFloat}

    noise_mat = Matrix{Float64}(noise_matrix)
    ref = Vector{Float64}(reference)

    n = size(noise_mat, 1)
    @assert size(noise_mat, 2) == n  "noise_matrix must be square"
    @assert length(ref) > 0          "reference signal must be non-empty"
    @assert a1 > 0  "a1 (noise penalty) must be positive"
    @assert a2 > 0  "a2 (reference penalty) must be positive"
    @assert a3 > 0  "a3 (zero-area penalty) must be positive"
    @assert ff in (0, 1)  "ff must be 0 or 1"

    ssize = length(ref)
    flo = div(ssize, 2) - div(n, 2)
    fhi = flo + n

    # Build reference matrix and reference signal vector
    ref_mat = zeros(Float64, n, n)
    ref_sig = zeros(Float64, n)

    offsets = ff == 0 ? (0:0) : (-1:1)
    for i in offsets
        chunk = ref[(flo + i + 1):(fhi + i)]   # Julia is 1-indexed
        ref_mat .+= chunk .* chunk'
        ref_sig .+= chunk
    end
    ref_mat ./= length(offsets)

    # Solve the penalized system: (a1*N + a2*R + a3*ones) * k = ref_sig
    mat = a1 .* noise_mat .+ a2 .* ref_mat .+ a3 .* ones(Float64, n, n)
    kernel = reverse(mat \ ref_sig)

    # Normalize so that the peak of the filtered reference is 1
    y = DSP.conv(ref, kernel)
    maxy = maximum(abs, y)
    kernel ./= maxy

    return kernel
end
