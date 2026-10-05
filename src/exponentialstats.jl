# This file is a part of RadiationDetectorDSP.jl, licensed under the MIT License (MIT).

"""
    singleexponentialstats(input, start, stop, τ)

Fit `offset + amplitude * exp(-(t - t₀)/τ)` by linear least squares with an
externally supplied, finite positive decay constant `τ`. Accepts sample vectors
or `RDWaveform`s with a uniformly spaced, increasing time axis. Bounds are rounded
to sample indices as in [`signalstats`](@ref), and both endpoints are included;
`t₀` is the first selected sample. At least two samples are required.

Return `mean`, `sigma` (RMS around the mean), `amplitude`, `offset`, `τ`, and
`residual_rms` (RMS around the fitted curve). Both RMS values use denominator `n`,
without a degrees-of-freedom correction. Amplitudes may have either sign. For
waveforms, `τ` must have units compatible with the time axis; for plain sample
vectors it is in sample spacings. Fits use the original samples, not logarithms.
"""
function singleexponentialstats(input::SamplesOrWaveform, start::RealQuantity, stop::RealQuantity, τ::RealQuantity)
    fixedexponentialstats(input, start, stop, (τ,))
end
export singleexponentialstats


"""
    doubleexponentialstats(input, start, stop, τ1, τ2)

Fit `offset + amplitude1 * exp(-(t - t₀)/τ1) + amplitude2 * exp(-(t - t₀)/τ2)`
by linear least squares with externally supplied, finite positive decay constants.
The constants may be supplied in either order; amplitudes correspond to that order.
At least three samples are required. Numerically indistinguishable decay shapes
raise `ArgumentError`; use [`singleexponentialstats`](@ref) for a single component.

Return `mean`, `sigma`, `amplitude1`, `amplitude2`, `offset`, `τ1`, `τ2`, and
`residual_rms`. The interval, units, amplitude reference, and RMS conventions are
the same as for [`singleexponentialstats`](@ref). Incorrect fixed decay constants
can leave deterministic structure in the residual RMS; it is not necessarily a
pure noise estimate.
"""
function doubleexponentialstats(input::SamplesOrWaveform, start::RealQuantity, stop::RealQuantity, τ1::RealQuantity, τ2::RealQuantity)
    fixedexponentialstats(input, start, stop, (τ1, τ2))
end
export doubleexponentialstats


# Shared scalar implementation: centered, scaled predictors and explicit
# orthogonalization avoid subtracting nearly equal terms in normal equations.
function fixedexponentialstats(input::SamplesOrWaveform, start::RealQuantity, stop::RealQuantity, τs::Tuple)
    X, Y = _get_axis_and_signal(input)
    @assert axes(X) == axes(Y)
    step_x = step(X)
    @argcheck step_x > zero(step_x)
    from = round(Int, ustrip(NoUnits, (start - first(X)) / step_x)) + firstindex(X)
    until = round(Int, ustrip(NoUnits, (stop - first(X)) / step_x)) + firstindex(X)
    @assert firstindex(X) <= from <= until <= lastindex(X)
    n = until - from + 1
    double = length(τs) == 2
    @argcheck n >= length(τs) + 1
    for τ in τs
        @argcheck isfinite(τ) && τ > zero(τ)
    end

    τ1, τ2 = first(τs), last(τs)
    span = X[until] - X[from]
    rate1, rate2 = promote(float(ustrip(NoUnits, span / τ1)), float(ustrip(NoUnits, span / τ2)))
    denom1, denom2 = expm1(-rate1), expm1(-rate2)
    @argcheck !iszero(denom1) && !iszero(denom2)
    inv_n = inv(oftype(rate1, n))
    origin = one(rate1) * float(Y[from])
    sum_y = zero(origin)
    sum_p1 = sum_p2 = zero(rate1)

    # expm1 preserves the variation of a decay much longer than the fit window.
    @inbounds for i in from:until
        t = oftype(rate1, ustrip(NoUnits, (X[i] - X[from]) / span))
        p1 = expm1(-t * rate1) / denom1
        p2 = expm1(-t * rate2) / denom2
        sum_p1 += p1
        sum_p2 += p2
        sum_y += float(Y[i]) - origin
    end
    mean_y = sum_y * inv_n
    mean_p1, mean_p2 = sum_p1 * inv_n, sum_p2 * inv_n
    s11 = s12 = s22 = zero(rate1)
    s1y = zero(rate1 * origin)
    syy = zero(origin * origin)

    @inbounds for i in from:until
        t = oftype(rate1, ustrip(NoUnits, (X[i] - X[from]) / span))
        p1 = expm1(-t * rate1) / denom1 - mean_p1
        p2 = expm1(-t * rate2) / denom2 - mean_p2
        y = (float(Y[i]) - origin) - mean_y
        s11 += p1 * p1
        s12 += p1 * p2
        s22 += p2 * p2
        s1y += p1 * y
        syy += y * y
    end
    @argcheck s11 > zero(s11)
    projection = s12 / s11
    b1 = s1y / s11
    b2 = zero(b1)

    if double
        sqq = zero(rate1)
        sqy = zero(s1y)
        @inbounds for i in from:until
            t = oftype(rate1, ustrip(NoUnits, (X[i] - X[from]) / span))
            p1 = expm1(-t * rate1) / denom1 - mean_p1
            p2 = expm1(-t * rate2) / denom2 - mean_p2
            q = p2 - projection * p1
            y = (float(Y[i]) - origin) - mean_y
            sqq += q * q
            sqy += q * (y - b1 * p1)
        end
        @argcheck sqq > eps(one(rate1)) * s22 "decay shapes are numerically indistinguishable"
        b2 = sqy / sqq
    end

    residual_sqr_sum = zero(syy)
    @inbounds for i in from:until
        t = oftype(rate1, ustrip(NoUnits, (X[i] - X[from]) / span))
        p1 = expm1(-t * rate1) / denom1 - mean_p1
        p2 = expm1(-t * rate2) / denom2 - mean_p2
        y = (float(Y[i]) - origin) - mean_y
        residual = (y - b1 * p1) - b2 * (p2 - projection * p1)
        residual_sqr_sum += residual * residual
    end
    mean = origin + mean_y
    sigma = sqrt(syy * inv_n)
    residual_rms = sqrt(residual_sqr_sum * inv_n)
    slope1 = b1 - b2 * projection
    amplitude1, amplitude2 = slope1 / denom1, b2 / denom2
    offset = mean - slope1 * mean_p1 - b2 * mean_p2 - amplitude1 - amplitude2

    if double
        return (; mean, sigma, amplitude1, amplitude2, offset, τ1, τ2, residual_rms)
    else
        return (; mean, sigma, amplitude = amplitude1, offset, τ = τ1, residual_rms)
    end
end
