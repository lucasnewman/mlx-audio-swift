@preconcurrency import MLX
@preconcurrency import MLXLMCommon
import MLXAudioCore

public protocol SpeechGenerationModel: AnyObject {
    var sampleRate: Int { get }

    func generate(
        text: String,
        voice: String?,
        generationParameters: GenerateParameters
    ) async throws -> MLXArray

    func generateStream(
        text: String,
        voice: String?,
        generationParameters: GenerateParameters
    ) -> AsyncThrowingStream<AudioGeneration, Error>
}

extension Qwen3Model: SpeechGenerationModel {
    public func generate(
        text: String,
        voice: String?,
        generationParameters: GenerateParameters
    ) async throws -> MLXArray {
        try await generate(
            text: text,
            voice: voice,
            cache: nil,
            parameters: generationParameters
        )
    }

    public func generateStream(
        text: String,
        voice: String?,
        generationParameters: GenerateParameters
    ) -> AsyncThrowingStream<AudioGeneration, Error> {
        generateStream(
            text: text,
            voice: voice,
            refAudio: nil,
            refText: nil,
            cache: nil,
            parameters: generationParameters
        )
    }
}

extension LlamaTTSModel: SpeechGenerationModel {
    public func generate(
        text: String,
        voice: String?,
        generationParameters: GenerateParameters
    ) async throws -> MLXArray {
        try await generate(
            text: text,
            voice: voice,
            cache: nil,
            parameters: generationParameters
        )
    }

    public func generateStream(
        text: String,
        voice: String?,
        generationParameters: GenerateParameters
    ) -> AsyncThrowingStream<AudioGeneration, Error> {
        generateStream(
            text: text,
            voice: voice,
            cache: nil,
            parameters: generationParameters
        )
    }
}

extension SopranoModel: SpeechGenerationModel {
    public func generate(
        text: String,
        voice: String?,
        generationParameters: GenerateParameters
    ) async throws -> MLXArray {
        try await generate(
            text: text,
            voice: voice,
            splitPattern: "\n",
            parameters: generationParameters
        )
    }

    public func generateStream(
        text: String,
        voice: String?,
        generationParameters: GenerateParameters
    ) -> AsyncThrowingStream<AudioGeneration, Error> {
        generateStream(
            text: text,
            voice: voice,
            parameters: generationParameters
        )
    }
}
