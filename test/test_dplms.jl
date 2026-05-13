# This file is a part of RadiationDetectorDSP.jl, licensed under the MIT License (MIT).

using RadiationDetectorDSP
using Test
using LinearAlgebra
using RadiationDetectorSignals
using Unitful
using RadiationDetectorDSP: bc_rdfilt, dplms_filter_coeffs
using Adapt

@testset "DPLMSChargeFilter" begin
    n         = 20
    ssize     = 60
    noise_mat = Matrix{Float64}(I, n, n)
    t         = range(-3.0, 3.0, length = ssize)
    reference = exp.(-0.5 .* t .^ 2)

    step_signal = vcat(fill(0.0, 200), fill(1.0, 200))
    step_wf     = RDWaveform(1u"ns" .* eachindex(step_signal), step_signal)

    @testset "Constructor with default parameters" begin
        flt_default = DPLMSChargeFilter(
            noise_matrix = noise_mat,
            reference    = reference
        )
        @test flt_default.a1 == 50.0
        @test flt_default.a2 == 0.1
        @test flt_default.a3 == 1.0
        @test flt_default.ff == 1
        @test flt_default.length == n
    end

    @testset "Constructor with custom parameters" begin
        flt_custom = DPLMSChargeFilter(
            noise_matrix = noise_mat,
            reference    = reference,
            a1 = 50.0, a2 = 0.1, a3 = 1.0,
            ff = 1,
            length = n
        )
        @test flt_custom.a1 == 50.0
        @test flt_custom.a2 == 0.1
        @test flt_custom.a3 == 1.0
        @test flt_custom.ff == 1
        @test flt_custom.length == n
        @test size(flt_custom.noise_matrix) == (n, n)
        @test length(flt_custom.reference) == ssize
    end

    @testset "Calling filter on waveform" begin
        flt = DPLMSChargeFilter(
            noise_matrix = noise_mat,
            reference    = reference,
            a1 = 50.0, a2 = 0.1, a3 = 1.0,
            ff = 1
        )
        result = flt(step_wf)
        @test result isa RDWaveform
        @test length(result.signal) > 0
    end

    @testset "Adapt.adapt_structure method" begin
        flt = DPLMSChargeFilter(
            noise_matrix = noise_mat,
            reference    = reference,
            a1 = 50.0, a2 = 0.1, a3 = 1.0,
            ff = 1
        )
        # adapt_structure should return the same filter (no adaptation needed)
        adapted = Adapt.adapt_structure(nothing, flt)
        @test adapted === flt
        @test adapted.a1 == flt.a1
        @test adapted.a2 == flt.a2
        @test adapted.a3 == flt.a3
        @test adapted.ff == flt.ff
    end

    @testset "ConvolutionFilter conversion" begin
        flt = DPLMSChargeFilter(
            noise_matrix = noise_mat,
            reference    = reference,
            a1 = 50.0, a2 = 0.1, a3 = 1.0,
            ff = 1
        )
        conv_flt = ConvolutionFilter(flt)
        @test conv_flt isa ConvolutionFilter
        # The convolution filter should have coefficients
        @test !isempty(conv_flt.coeffs)
        @test length(conv_flt.coeffs) == n
    end

    @testset "fltinstance method" begin
        flt = DPLMSChargeFilter(
            noise_matrix = noise_mat,
            reference    = reference,
            a1 = 50.0, a2 = 0.1, a3 = 1.0,
            ff = 1
        )
        # Create a sampling info from time axis
        time_axis = 1u"ns" .* eachindex(step_signal)
        si = SamplingInfo{typeof(step_signal[1])}(time_axis)
        flt_inst = fltinstance(flt, si)
        @test flt_inst isa Union{ConvolutionFilterInst, FFTConvolutionFilterInst}
    end

    @testset "Different ff values" begin
        @testset "ff = 0" begin
            flt_ff0 = DPLMSChargeFilter(
                noise_matrix = noise_mat,
                reference    = reference,
                a1 = 50.0, a2 = 0.1, a3 = 1.0,
                ff = 0
            )
            @test flt_ff0.ff == 0
            result_ff0 = flt_ff0(step_wf)
            @test result_ff0 isa RDWaveform
        end

        @testset "ff = 1" begin
            flt_ff1 = DPLMSChargeFilter(
                noise_matrix = noise_mat,
                reference    = reference,
                a1 = 50.0, a2 = 0.1, a3 = 1.0,
                ff = 1
            )
            @test flt_ff1.ff == 1
            result_ff1 = flt_ff1(step_wf)
            @test result_ff1 isa RDWaveform
        end
    end

    @testset "Parameter validation in dplms_filter_coeffs" begin
        @testset "Non-square noise matrix" begin
            bad_noise_mat = Matrix{Float64}(I, n, n+1)
            @test_throws AssertionError dplms_filter_coeffs(
                bad_noise_mat, reference, 50.0, 0.1, 1.0, 1
            )
        end

        @testset "Empty reference signal" begin
            empty_ref = Float64[]
            @test_throws AssertionError dplms_filter_coeffs(
                noise_mat, empty_ref, 50.0, 0.1, 1.0, 1
            )
        end

        @testset "Negative a1 penalty" begin
            @test_throws AssertionError dplms_filter_coeffs(
                noise_mat, reference, -50.0, 0.1, 1.0, 1
            )
        end

        @testset "Zero a1 penalty" begin
            @test_throws AssertionError dplms_filter_coeffs(
                noise_mat, reference, 0.0, 0.1, 1.0, 1
            )
        end

        @testset "Negative a2 penalty" begin
            @test_throws AssertionError dplms_filter_coeffs(
                noise_mat, reference, 50.0, -0.1, 1.0, 1
            )
        end

        @testset "Zero a2 penalty" begin
            @test_throws AssertionError dplms_filter_coeffs(
                noise_mat, reference, 50.0, 0.0, 1.0, 1
            )
        end

        @testset "Negative a3 penalty" begin
            @test_throws AssertionError dplms_filter_coeffs(
                noise_mat, reference, 50.0, 0.1, -1.0, 1
            )
        end

        @testset "Zero a3 penalty" begin
            @test_throws AssertionError dplms_filter_coeffs(
                noise_mat, reference, 50.0, 0.1, 0.0, 1
            )
        end

        @testset "Invalid ff value" begin
            @test_throws AssertionError dplms_filter_coeffs(
                noise_mat, reference, 50.0, 0.1, 1.0, 2
            )
        end

        @testset "Invalid ff value (negative)" begin
            @test_throws AssertionError dplms_filter_coeffs(
                noise_mat, reference, 50.0, 0.1, 1.0, -1
            )
        end
    end

    @testset "Different matrix types" begin
        @testset "Float32 noise matrix" begin
            noise_mat_f32 = Matrix{Float32}(I, n, n)
            ref_f32 = Float32.(reference)
            flt_f32 = DPLMSChargeFilter(
                noise_matrix = noise_mat_f32,
                reference    = ref_f32,
                a1 = 50.0f0, a2 = 0.1f0, a3 = 1.0f0,
                ff = 1
            )
            result_f32 = flt_f32(step_wf)
            @test result_f32 isa RDWaveform
        end

        @testset "Different matrix types for noise_matrix" begin
            # Using a different positive definite matrix
            noise_mat_custom = Matrix{Float64}(I, n, n) .* 2.0
            flt_custom_mat = DPLMSChargeFilter(
                noise_matrix = noise_mat_custom,
                reference    = reference,
                a1 = 50.0, a2 = 0.1, a3 = 1.0,
                ff = 1
            )
            result_custom = flt_custom_mat(step_wf)
            @test result_custom isa RDWaveform
        end
    end

    @testset "Various penalty combinations" begin
        penalty_combos = [
            (1.0, 0.1, 0.1),
            (10.0, 1.0, 1.0),
            (100.0, 0.01, 0.01),
            (50.0, 0.5, 2.0)
        ]
        for (a1, a2, a3) in penalty_combos
            flt = DPLMSChargeFilter(
                noise_matrix = noise_mat,
                reference    = reference,
                a1 = a1, a2 = a2, a3 = a3,
                ff = 1
            )
            result = flt(step_wf)
            @test result isa RDWaveform
            @test all(isfinite.(result.signal))
        end
    end
end
