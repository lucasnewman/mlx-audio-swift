@preconcurrency import AVFoundation
import os
import Speech

actor SimpleVAD {
    enum Event: Sendable {
        case started
        case stopped(transcription: String?)
    }

    private enum LifecycleState {
        case idle
        case starting
        case ready
        case failed
    }

    var hangTime: TimeInterval

    private var analyzer: SpeechAnalyzer?
    private var analyzerInputContinuation: AsyncStream<AnalyzerInput>.Continuation?
    private var analysisFormat: AVAudioFormat?
    private var analyzerConverter: AVAudioConverter?
    private var transcriberTask: Task<Void, Never>?
    private var lifecycleState: LifecycleState = .idle
    private var isListening = false
    private var lastSpeechTime: TimeInterval?
    private let detectionRMSThreshold: Float

    private let transcriptState = TranscriptState()

    init(hangTime: TimeInterval = 2.0, detectionRMSThreshold: Float = 0.02) {
        self.hangTime = hangTime
        self.detectionRMSThreshold = detectionRMSThreshold

        Task {
            await setupSpeechPipeline()
        }
    }

    deinit {
        analyzerInputContinuation?.finish()
        transcriberTask?.cancel()
    }

    func process(chunk: AudioChunk) async -> Event? {
        guard lifecycleState == .ready else { return nil }
        guard let buffer = AVAudioPCMBuffer.makeFrom(chunk: chunk) else { return nil }
        return await processBuffer(buffer)
    }

    func reset() async {
        isListening = false
        lastSpeechTime = nil
        await transcriptState.reset()
    }

    private func setupSpeechPipeline() async {
        guard lifecycleState == .idle else { return }
        lifecycleState = .starting

        guard let locale = await SpeechTranscriber.supportedLocale(equivalentTo: Locale.current) else {
            print("Warning: Current locale (\(Locale.current)) is not supported for speech transcription.")
            lifecycleState = .failed
            return
        }

        let transcriber = SpeechTranscriber(locale: locale, preset: .progressiveTranscription)
        do {
            try await prepareAssets(for: transcriber)
        } catch {
            print("Error: Unable to prepare on-device transcription: \(error)")
            lifecycleState = .failed
            return
        }

        guard let format = await SpeechAnalyzer.bestAvailableAudioFormat(compatibleWith: [transcriber]) else {
            print("Error: Speech transcriber unavailable until required assets are installed.")
            lifecycleState = .failed
            return
        }

        let inputStream = AsyncStream<AnalyzerInput> { continuation in
            analyzerInputContinuation = continuation
            continuation.onTermination = { _ in
                Task {
                    await self.clearAnalyzerContinuation()
                }
            }
        }

        let analyzer = SpeechAnalyzer(
            modules: [transcriber],
            options: .init(priority: .userInitiated, modelRetention: .lingering)
        )

        self.analyzer = analyzer
        analysisFormat = format
        analyzerConverter = nil
        lifecycleState = .ready

        Task {
            do {
                try await analyzer.start(inputSequence: inputStream)
            } catch {
                print("Error: Speech analyzer failed: \(error)")
                lifecycleState = .failed
            }
        }

        transcriberTask?.cancel()
        transcriberTask = Task {
            do {
                for try await result in transcriber.results {
                    await transcriptState.recordResult(result)
                }
            } catch {
                print("Error: Transcriber results failed: \(error)")
            }
        }
    }

    private func clearAnalyzerContinuation() {
        analyzerInputContinuation = nil
    }

    private func prepareAssets(for transcriber: SpeechTranscriber) async throws {
        if let installationRequest = try await AssetInventory.assetInstallationRequest(supporting: [transcriber]) {
            try await installationRequest.downloadAndInstall()
        }
    }

    private func processBuffer(_ buffer: AVAudioPCMBuffer) async -> Event? {
        enqueueForSpeechTranscription(buffer)

        let now = CACurrentMediaTime()
        let isSpeechFrame = buffer.rmsLevel() >= detectionRMSThreshold

        if isSpeechFrame {
            if !isListening {
                isListening = true
                await transcriptState.beginUtterance()
                print("Did start listening (RMS VAD).")
                lastSpeechTime = now
                return .started
            }
            lastSpeechTime = now
            return nil
        }

        guard isListening, let lastSpeechTime else { return nil }

        let idleDuration = now - lastSpeechTime
        guard idleDuration > hangTime else { return nil }

        isListening = false
        self.lastSpeechTime = nil

        let transcription = await transcriptState.consumeTranscript()
        print("Did stop listening after \(idleDuration)s below RMS threshold.")
        return .stopped(transcription: transcription)
    }

    private func enqueueForSpeechTranscription(_ buffer: AVAudioPCMBuffer) {
        guard let continuation = analyzerInputContinuation else { return }
        guard let converted = convertBufferIfNeeded(buffer) else { return }
        continuation.yield(AnalyzerInput(buffer: converted))
    }

    private func convertBufferIfNeeded(_ buffer: AVAudioPCMBuffer) -> AVAudioPCMBuffer? {
        guard let analysisFormat else { return buffer }
        if formatsMatch(buffer.format, analysisFormat) { return buffer }

        if analyzerConverter == nil ||
            !formatsMatch(analyzerConverter?.inputFormat, buffer.format) ||
            !formatsMatch(analyzerConverter?.outputFormat, analysisFormat) {
            analyzerConverter = AVAudioConverter(from: buffer.format, to: analysisFormat)
        }

        guard let converter = analyzerConverter else {
            print("Error: Unable to create audio converter for speech transcription.")
            return nil
        }

        let ratio = analysisFormat.sampleRate / buffer.format.sampleRate
        let capacity = max(AVAudioFrameCount(Double(buffer.frameLength) * ratio + 1), 1)
        guard let outBuffer = AVAudioPCMBuffer(pcmFormat: analysisFormat, frameCapacity: capacity) else {
            return nil
        }

        var error: NSError?
        let inputState = OSAllocatedUnfairLock(initialState: false)
        converter.convert(to: outBuffer, error: &error) { _, outStatus in
            let shouldProvideInput = inputState.withLock { didProvideInput in
                if didProvideInput {
                    return false
                }
                didProvideInput = true
                return true
            }
            if !shouldProvideInput {
                outStatus.pointee = .noDataNow
                return nil
            }
            outStatus.pointee = .haveData
            return buffer
        }

        if let error {
            print("Error: Audio conversion failed for speech transcription: \(error)")
            return nil
        }
        return outBuffer.frameLength > 0 ? outBuffer : nil
    }

    private func formatsMatch(_ lhs: AVAudioFormat?, _ rhs: AVAudioFormat?) -> Bool {
        guard let lhs, let rhs else { return false }
        return lhs.sampleRate == rhs.sampleRate &&
            lhs.channelCount == rhs.channelCount &&
            lhs.commonFormat == rhs.commonFormat &&
            lhs.isInterleaved == rhs.isInterleaved
    }
}

private actor TranscriptState {
    private var finalizedTranscript = ""
    private var latestHypothesis: String?

    func reset() {
        finalizedTranscript = ""
        latestHypothesis = nil
    }

    func beginUtterance() {
        finalizedTranscript = ""
        latestHypothesis = nil
    }

    func recordResult(_ result: SpeechTranscriber.Result) {
        let text = String(result.text.characters).trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        let isFinalized = CMTimeCompare(result.resultsFinalizationTime, result.range.end) >= 0

        if isFinalized {
            mergeFinalizedText(text)
            latestHypothesis = nil
        } else {
            latestHypothesis = text
        }
    }

    func consumeTranscript() -> String? {
        let finalized = finalizedTranscript.trimmingCharacters(in: .whitespacesAndNewlines)
        let hypothesis = latestHypothesis?.trimmingCharacters(in: .whitespacesAndNewlines)
        let transcription: String? = if !finalized.isEmpty {
            finalized
        } else if let hypothesis, !hypothesis.isEmpty {
            hypothesis
        } else {
            nil
        }

        finalizedTranscript = ""
        latestHypothesis = nil
        return transcription
    }

    private func mergeFinalizedText(_ text: String) {
        if finalizedTranscript.isEmpty {
            finalizedTranscript = text
            return
        }
        if text == finalizedTranscript || finalizedTranscript.hasSuffix(text) {
            return
        }
        if text.hasPrefix(finalizedTranscript) {
            finalizedTranscript = text
            return
        }
        finalizedTranscript += " " + text
    }
}

// MARK: - AVAudioPCMBuffer Helpers

private extension AVAudioPCMBuffer {
    static func makeFrom(chunk: AudioChunk) -> AVAudioPCMBuffer? {
        guard chunk.channelCount > 0, chunk.frameLength > 0 else { return nil }
        guard chunk.samples.count == chunk.frameLength * chunk.channelCount else { return nil }

        guard let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: chunk.sampleRate,
            channels: AVAudioChannelCount(chunk.channelCount),
            interleaved: chunk.isInterleaved
        ) else {
            return nil
        }

        guard let buffer = AVAudioPCMBuffer(
            pcmFormat: format,
            frameCapacity: AVAudioFrameCount(chunk.frameLength)
        ) else {
            return nil
        }

        buffer.frameLength = AVAudioFrameCount(chunk.frameLength)

        guard let destination = buffer.floatChannelData else { return nil }

        if chunk.isInterleaved {
            _ = chunk.samples.withUnsafeBufferPointer { source in
                memcpy(
                    destination[0],
                    source.baseAddress!,
                    chunk.samples.count * MemoryLayout<Float>.size
                )
            }
        } else {
            for channel in 0 ..< chunk.channelCount {
                let sourceOffset = channel * chunk.frameLength
                _ = chunk.samples.withUnsafeBufferPointer { source in
                    memcpy(
                        destination[channel],
                        source.baseAddress!.advanced(by: sourceOffset),
                        chunk.frameLength * MemoryLayout<Float>.size
                    )
                }
            }
        }

        return buffer
    }

    func rmsLevel() -> Float {
        guard format.commonFormat == .pcmFormatFloat32 else {
            assertionFailure("SimpleVAD only supports .pcmFormatFloat32.")
            return 0
        }

        let frameCount = Int(frameLength)
        let channelCount = Int(format.channelCount)
        guard frameCount > 0, channelCount > 0 else { return 0 }
        guard let data = floatChannelData else { return 0 }

        var sumSquares = 0.0
        var sampleCount = 0

        if format.isInterleaved {
            let totalSamples = frameCount * channelCount
            for idx in 0 ..< totalSamples {
                let sample = Double(data[0][idx])
                sumSquares += sample * sample
            }
            sampleCount = totalSamples
        } else {
            for channel in 0 ..< channelCount {
                for frame in 0 ..< frameCount {
                    let sample = Double(data[channel][frame])
                    sumSquares += sample * sample
                }
            }
            sampleCount = frameCount * channelCount
        }

        guard sampleCount > 0 else { return 0 }
        return Float(sqrt(sumSquares / Double(sampleCount)))
    }
}
