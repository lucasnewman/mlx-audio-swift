import AVFoundation
import MLX
import MLXAudioCore
import MLXAudioTTS
import MLXLMCommon

@MainActor
protocol SpeechControllerDelegate: AnyObject {
    func speechController(_ controller: SpeechController, didFinish transcription: String)
}

@MainActor
@Observable
final class SpeechController {
    @ObservationIgnored
    weak var delegate: SpeechControllerDelegate?

    private(set) var isActive: Bool = false
    private(set) var isDetectingSpeech = false
    private(set) var canSpeak: Bool = false
    private(set) var isSpeaking: Bool = false

    var isMicrophoneMuted: Bool {
        audioEngine.isMicrophoneMuted
    }

    @ObservationIgnored
    private let audioEngine: AudioEngine
    @ObservationIgnored
    private var configuredAudioEngine = false
    @ObservationIgnored
    private let vad: SimpleVAD
    @ObservationIgnored
    private var model: SpeechGenerationModel?
    @ObservationIgnored
    private var captureTask: Task<Void, Never>?

    init(ttsRepoId: String = "mlx-community/pocket-tts") {
        self.audioEngine = AudioEngine(inputBufferSize: 1024)
        self.vad = SimpleVAD()
        audioEngine.delegate = self

        Task { @MainActor in
            do {
                print("Loading TTS model: \(ttsRepoId)")
                self.model = try await TTS.loadModel(modelRepo: ttsRepoId)
                print("Loaded TTS model.")
            } catch {
                print("Error loading model: \(error)")
            }
            self.canSpeak = model != nil
        }
    }

    func start() async throws {
#if os(iOS)
        let session = AVAudioSession.sharedInstance()
        try session.setActive(false)
        try session.setCategory(.playAndRecord, mode: .voiceChat, policy: .default, options: [.defaultToSpeaker])
        try session.setPreferredIOBufferDuration(0.02)
        try session.setActive(true)
#endif

        try await ensureEngineStarted()
        startCaptureLoopIfNeeded()
        isActive = true
    }

    func stop() async throws {
        stopCaptureLoop()
        audioEngine.endSpeaking()
        audioEngine.stop()
        isDetectingSpeech = false
        await vad.reset()
#if os(iOS)
        try AVAudioSession.sharedInstance().setActive(false)
#endif
        isActive = false
    }

    func toggleInputMute(toMuted: Bool?) async {
        let currentMuted = audioEngine.isMicrophoneMuted
        let newMuted = toMuted ?? !currentMuted
        audioEngine.isMicrophoneMuted = newMuted

        if newMuted, isDetectingSpeech {
            await vad.reset()
            isDetectingSpeech = false
        }
    }

    func stopSpeaking() async {
        audioEngine.endSpeaking()
    }

    func speak(text: String) async throws {
        guard let model else {
            print("Error: TTS model not yet loaded.")
            return
        }

        let audioStream = model.generatePCMBufferStream(
            text: text,
            voice: "cosette",
            refAudio: nil,
            refText: nil,
            language: "en",
            generationParameters: model.defaultGenerationParameters
        )
        try await ensureEngineStarted()

        audioEngine.speak(buffersStream: audioStream)
    }

    private func ensureEngineStarted() async throws {
        if !configuredAudioEngine {
            try audioEngine.setup()
            configuredAudioEngine = true
            print("Configured audio engine.")
        }
        try audioEngine.start()
        audioEngine.isMicrophoneMuted = false
        print("Started audio engine.")
    }

    private func startCaptureLoopIfNeeded() {
        guard captureTask == nil else { return }

        let stream = audioEngine.capturedChunks
        let vad = self.vad
        captureTask = Task(priority: .userInitiated) { [weak self] in
            await Self.runCaptureLoop(stream: stream, vad: vad) { event in
                await MainActor.run {
                    self?.handleVADEvent(event)
                }
            }
        }
    }

    private func stopCaptureLoop() {
        captureTask?.cancel()
        captureTask = nil
    }

    private func handleVADEvent(_ event: SimpleVAD.Event) {
        switch event {
        case .started:
            isDetectingSpeech = true
        case let .stopped(transcription):
            if let transcription {
                delegate?.speechController(self, didFinish: transcription)
            }
            isDetectingSpeech = false
        }
    }

    @concurrent
    private static func runCaptureLoop(
        stream: AsyncStream<AudioChunk>,
        vad: SimpleVAD,
        onEvent: @escaping @Sendable (SimpleVAD.Event) async -> Void
    ) async {
        for await chunk in stream {
            if Task.isCancelled { break }
            if let event = await vad.process(chunk: chunk) {
                await onEvent(event)
            }
        }
    }

    private func proxyAudioStream<T>(_ upstream: AsyncThrowingStream<T, any Error>, extract: @escaping (T) -> [Float]) -> AsyncThrowingStream<[Float], any Error> {
        AsyncThrowingStream<[Float], any Error> { continuation in
            let task = Task {
                do {
                    for try await value in upstream {
                        continuation.yield(extract(value))
                    }
                    continuation.finish()
                } catch is CancellationError {
                    continuation.finish(throwing: CancellationError())
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { @Sendable _ in task.cancel() }
        }
    }
}

// MARK: - AudioEngineDelegate

extension SpeechController: @MainActor AudioEngineDelegate {
    func audioCaptureEngine(_ engine: AudioEngine, isSpeakingDidChange speaking: Bool) {
        isSpeaking = speaking
    }
}
