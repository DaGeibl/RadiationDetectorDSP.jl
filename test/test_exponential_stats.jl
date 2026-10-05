# This file is a part of RadiationDetectorDSP.jl, licensed under the MIT License (MIT).
using Test
using RadiationDetectorDSP
using RadiationDetectorSignals
using Unitful
using LinearAlgebra
using Random
using Statistics

@testset "fixed exponential statistics" begin
    @testset "exact fits, signed amplitudes, and units" begin
        xs = (10.0:0.25:40.0)u"μs"
        t = xs .- first(xs)
        τ1, τ2 = 4u"μs", 13u"μs"
        ys = 2u"keV" .- 12u"keV" .* exp.(-t ./ τ1)
        s = @inferred singleexponentialstats(RDWaveform(xs, ys), first(xs), last(xs), τ1)
        @test s.amplitude ≈ -12u"keV"
        @test s.offset ≈ 2u"keV"
        @test s.τ == τ1
        @test s.residual_rms < 1e-12u"keV"
        @test s.mean ≈ mean(ys)
        @test s.sigma ≈ std(ys; corrected=false)

        ys .+= 7u"keV" .* exp.(-t ./ τ2)
        wf = RDWaveform(xs, ys)
        s = @inferred doubleexponentialstats(wf, first(xs), last(xs), τ1, τ2)
        @test s.amplitude1 ≈ -12u"keV"
        @test s.amplitude2 ≈ 7u"keV"
        @test s.offset ≈ 2u"keV"
        @test (s.τ1, s.τ2) == (τ1, τ2)
        @test s.residual_rms < 1e-12u"keV"
        @test s.mean ≈ mean(ys)
        @test s.sigma ≈ std(ys; corrected=false)
        swapped = doubleexponentialstats(wf, first(xs), last(xs), τ2, τ1)
        @test swapped.amplitude1 ≈ s.amplitude2
        @test swapped.amplitude2 ≈ s.amplitude1
        @test swapped.offset ≈ s.offset

        # Mixed compatible time units and a new amplitude reference at the subrange.
        sub = doubleexponentialstats(wf, 15u"μs", 30u"μs", 0.004u"ms", τ2)
        @test sub.amplitude1 ≈ -12u"keV" * exp(-5/4)
        @test sub.amplitude2 ≈ 7u"keV" * exp(-5/13)
        @test sub.offset ≈ 2u"keV"
        @test sub.residual_rms < 1e-12u"keV"
    end

    @testset "noisy fits agree with independent QR" begin
        rng = MersenneTwister(42)
        xs = 0.0:0.01:19.99
        for double in (false, true)
            τ1, τ2 = 3.0, 10.0
            ys = 5 .+ 100 .* exp.(-xs ./ τ1) .+ randn(rng, length(xs))
            double && (ys .+= 30 .* exp.(-xs ./ τ2))
            wf = RDWaveform(xs, ys)
            # Non-grid bounds round to inclusive indices 101:1901.
            idxs = 101:1901
            t = xs[idxs] .- xs[first(idxs)]
            E = double ? hcat(ones(length(t)), exp.(-t ./ τ1), exp.(-t ./ τ2)) :
                         hcat(ones(length(t)), exp.(-t ./ τ1))
            coeff = qr(E) \ ys[idxs]
            ref_rms = sqrt(sum(abs2, ys[idxs] - E * coeff) / length(idxs))
            s = double ? doubleexponentialstats(wf, 1.002, 18.998, τ1, τ2) :
                         singleexponentialstats(wf, 1.002, 18.998, τ1)
            @test s.offset ≈ coeff[1] rtol=1e-10
            @test (double ? s.amplitude1 : s.amplitude) ≈ coeff[2] rtol=1e-10
            double && (@test s.amplitude2 ≈ coeff[3] rtol=1e-10)
            @test s.residual_rms ≈ ref_rms rtol=1e-12
            @test 0.9 < s.residual_rms < 1.1
            @test s.mean ≈ mean(ys[idxs])
            @test s.sigma ≈ std(ys[idxs]; corrected=false)
        end
    end

    @testset "constant, integer, and Float32 samples" begin
        for ys in (fill(5, 100), fill(5f0, 100))
            s = @inferred singleexponentialstats(ys, 1, 100, 10f0)
            d = @inferred doubleexponentialstats(ys, 1, 100, 10f0, 30f0)
            for fit in (s, d)
                @test fit.mean ≈ 5
                @test fit.offset ≈ 5
                @test fit.sigma ≈ 0 atol=1e-12
                @test fit.residual_rms ≈ 0 atol=1e-12
            end
            @test s.amplitude ≈ 0 atol=1e-12
            @test d.amplitude1 ≈ 0 atol=1e-12
            @test d.amplitude2 ≈ 0 atol=1e-12
        end
        t = Float32.(0:99)
        ys = 1.5f0 .+ 4f0 .* exp.(-t ./ 12f0) .- 2f0 .* exp.(-t ./ 35f0)
        s = @inferred doubleexponentialstats(ys, 1, 100, 12f0, 35f0)
        @test s.amplitude1 ≈ 4 rtol=2e-5
        @test s.amplitude2 ≈ -2 rtol=2e-5
        @test s.offset ≈ 1.5 rtol=2e-5
        @test s.residual_rms < 2e-6
    end

    @testset "long decay and large offset" begin
        xs = 0.0:1.0:100.0
        τ = 1e8
        ys = 1e8 .+ 1e6 .* exp.(-xs ./ τ)
        s = singleexponentialstats(RDWaveform(xs, ys), first(xs), last(xs), τ)
        @test s.amplitude ≈ 1e6 rtol=1e-7
        @test s.residual_rms < 2e-8
        @test s.sigma ≈ std(ys; corrected=false) rtol=1e-7
    end

    @testset "model mismatch and missing second component" begin
        xs = 0.0:0.1:30.0
        ys = 2 .+ 10 .* exp.(-xs ./ 3) .+ 5 .* exp.(-xs ./ 12)
        wf = RDWaveform(xs, ys)
        good = doubleexponentialstats(wf, 0, 30, 3, 12)
        wrong = doubleexponentialstats(wf, 0, 30, 5, 20)
        single = singleexponentialstats(wf, 0, 30, 3)
        @test good.residual_rms < 1e-12
        @test wrong.residual_rms > 0.01
        @test single.residual_rms > 0.01
        ys = 2 .+ 10 .* exp.(-xs ./ 3)
        s = doubleexponentialstats(RDWaveform(xs, ys), 0, 30, 3, 12)
        @test s.amplitude1 ≈ 10
        @test s.amplitude2 ≈ 0 atol=1e-12
        @test s.residual_rms < 1e-12
    end

    @testset "validation and minimum windows" begin
        ys = ones(10)
        for τ in (0, -1, Inf, NaN)
            @test_throws ArgumentError singleexponentialstats(ys, 1, 10, τ)
            @test_throws ArgumentError doubleexponentialstats(ys, 1, 10, 2, τ)
            @test_throws ArgumentError doubleexponentialstats(ys, 1, 10, τ, 2)
        end
        @test_throws ArgumentError doubleexponentialstats(ys, 1, 10, 2, 2)
        @test_throws ArgumentError doubleexponentialstats(ys, 1, 10, 2.0, 2.0 + 1e-12)
        @test_throws ArgumentError singleexponentialstats(ys, 1, 1, 2)
        @test_throws ArgumentError doubleexponentialstats(ys, 1, 2, 2, 5)
        @test_throws AssertionError singleexponentialstats(ys, 0, 10, 2)
        @test_throws AssertionError singleexponentialstats(ys, 5, 4, 2)
        @test_throws ArgumentError singleexponentialstats(RDWaveform(10:-1:1, ys), 10, 1, 2)
        @test singleexponentialstats([3., 2.], 1, 2, 2).residual_rms < 1e-12
        @test doubleexponentialstats([3., 2., 1.], 1, 3, 2, 5).residual_rms < 1e-12
    end
end
