//
//  MLXAudioTests.swift
//  MLXAudioTests
//
//  Created by Ben Harraway on 14/04/2025.
//


import Testing
import MLX
import Foundation
import MLX
import Foundation

@testable import MLXAudioTTS
@testable import MLXAudioCodecs


// Run ALL tests with:  xcodebuild test \
// -scheme MLXAudio-Package \
// -destination 'platform=macOS' \
// -only-testing:MLXAudioTests/MLXAudioCodecsTests
//  2>&1 | grep -E "(Suite.*started|Test test.*started|Loaded|Loading|model loaded|input shape|Encoding|Encoded|Decoding|Decoded|Reconstructed|Saved|passed after|failed after|TEST SUCCEEDED|TEST FAILED|Suite.*passed|Test run)"


// MARK: - SNAC Tests
// Run SNAC tests with:  xcodebuild test \
// -scheme MLXAudio-Package \
// -destination 'platform=macOS' \
// -only-testing:MLXAudioTests/SNACTests
//  2>&1 | grep -E "(Suite.*started|Test test.*started|Loaded|Loading|model loaded|input shape|Encoding|Encoded|Decoding|Decoded|Reconstructed|Saved|passed after|failed after|TEST SUCCEEDED|TEST FAILED|Suite.*passed|Test run)"


struct SNACTests {

    @Test func testSNACEncodeDecodeCycle() async throws {
        // 1. Load audio from file
        let audioURL = Bundle.module.url(forResource: "intention", withExtension: "wav", subdirectory: "media")!
        let (sampleRate, audioData) = try loadAudioArray(from: audioURL)
        print("Loaded audio: \(audioData.shape), sample rate: \(sampleRate)")

        // 2. Load SNAC model from HuggingFace (24kHz model)
        print("\u{001B}[33mLoading SNAC model...\u{001B}[0m")
        let snac = try await SNAC.fromPretrained("mlx-community/snac_24khz")
        print("\u{001B}[32mSNAC model loaded!\u{001B}[0m")

        // 3. Reshape audio for SNAC: [batch, channels, samples]
        let audioInput = audioData.reshaped([1, 1, audioData.shape[0]])
        print("Audio input shape: \(audioInput.shape)")

        // 4. Encode audio to codes
        print("\u{001B}[33mEncoding audio...\u{001B}[0m")
        let codes = snac.encode(audioInput)
        print("Encoded to \(codes.count) codebook levels:")
        for (i, code) in codes.enumerated() {
            print("  Level \(i): \(code.shape)")
        }

        // 5. Decode codes back to audio
        print("\u{001B}[33mDecoding audio...\u{001B}[0m")
        let reconstructed = snac.decode(codes)
        print("Reconstructed audio shape: \(reconstructed.shape)")

        // 6. Save reconstructed audio to the same media folder as input
        let outputURL = audioURL.deletingLastPathComponent().appendingPathComponent("intention_snac_reconstructed.wav")
        let outputAudio = reconstructed.squeezed()  // Remove batch/channel dims
        try saveAudioArray(outputAudio, sampleRate: Double(snac.samplingRate), to: outputURL)
        print("\u{001B}[32mSaved reconstructed audio to\u{001B}[0m: \(outputURL.path)")

        // Basic check: output should have samples
        #expect(reconstructed.shape.last! > 0)
    }
}


// MARK: - Mimi Tests
// Run Mimi tests with:  xcodebuild test \
// -scheme MLXAudio-Package \
// -destination 'platform=macOS' \
// -only-testing:MLXAudioTests/MimiTests \
// 2>&1 | grep -E "(Suite.*started|Test test.*started|Loaded|Loading|model loaded|input shape|Encoding|Encoded|Decoding|Decoded|Reconstructed|Saved|passed after|failed after|TEST SUCCEEDED|TEST FAILED|Suite.*passed|Test run)"


struct MimiTests {

    @Test func testMimiEncodeDecodeCycle() async throws {
        // 1. Load audio from file
        let audioURL = Bundle.module.url(forResource: "intention", withExtension: "wav", subdirectory: "media")!
        let (sampleRate, audioData) = try loadAudioArray(from: audioURL)
        print("Loaded audio: \(audioData.shape), sample rate: \(sampleRate)")

        // 2. Load Mimi model from HuggingFace
        print("\u{001B}[33mLoading Mimi model...\u{001B}[0m")
        let mimi = try await Mimi.fromPretrained(
            repoId: "kyutai/moshiko-pytorch-bf16",
            filename: "tokenizer-e351c8d8-checkpoint125.safetensors"
        ) { progress in
            print("Download progress: \(progress.fractionCompleted * 100)%")
        }
        print("\u{001B}[32mMimi model loaded!\u{001B}[0m")

        // 3. Reshape audio for Mimi: [batch, channels, samples]
        let audioInput = audioData.reshaped([1, 1, audioData.shape[0]])
        print("Audio input shape: \(audioInput.shape)")

        // 4. Encode audio to codes
        print("\u{001B}[33mEncoding audio...\u{001B}[0m")
        let codes = mimi.encode(audioInput)
        print("Encoded to codes shape: \(codes.shape)")

        // 5. Decode codes back to audio
        print("\u{001B}[33mDecoding audio...\u{001B}[0m")
        let reconstructed = mimi.decode(codes)
        GPU.clearCache()
        print("Reconstructed audio shape: \(reconstructed.shape)")

        // 6. Save reconstructed audio
        let outputURL = audioURL.deletingLastPathComponent().appendingPathComponent("intention_mimi_reconstructed.wav")
        let outputAudio = reconstructed.squeezed()  // Remove batch/channel dims
        try saveAudioArray(outputAudio, sampleRate: mimi.sampleRate, to: outputURL)
        print("\u{001B}[32mSaved reconstructed audio to\u{001B}[0m: \(outputURL.path)")

        // Basic check: output should have samples
        #expect(reconstructed.shape.last! > 0)
    }
}


