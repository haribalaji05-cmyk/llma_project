import { useEffect, useMemo, useRef, useState } from "react";

const apiBase = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

const suggestions = [
  "\u092a\u093e\u0938\u092a\u094b\u0930\u094d\u091f \u0915\u0947 \u0932\u093f\u090f \u0906\u0935\u0947\u0926\u0928 \u0915\u0948\u0938\u0947 \u0915\u0930\u0947\u0902?",
  "How much is the passport fee?",
  "How do I withdraw PF online?",
  "What documents are needed for Aadhaar enrolment?",
];

function App() {
  const [query, setQuery] = useState("");
  const [response, setResponse] = useState(null);
  const [transcript, setTranscript] = useState("");
  const [transcriptLanguage, setTranscriptLanguage] = useState("");
  const [loading, setLoading] = useState(false);
  const [recording, setRecording] = useState(false);
  const [audioUrl, setAudioUrl] = useState("");
  const [message, setMessage] = useState("");
  const [recorder, setRecorder] = useState(null);
  const [inputFile, setInputFile] = useState(null);
  const chunksRef = useRef([]);

  useEffect(() => {
    return () => {
      if (recorder && recorder.state !== "inactive") {
        recorder.stop();
      }
    };
  }, [recorder]);

  const structuredAnswer = useMemo(() => {
    if (!response) return null;
    return {
      title: response.answer.title,
      steps: response.answer.steps,
      documents: response.answer.documents,
      fees: response.answer.fees,
      processing_time: response.answer.processing_time,
      eligibility: response.answer.eligibility,
      official_links: response.answer.official_links,
    };
  }, [response]);

  const hasItems = (value) => Array.isArray(value) && value.length > 0;
  const hasText = (value) => Boolean(value && value !== "Not available");

  const answerSummary = useMemo(() => {
    if (!response) return [];
    return [
      { label: "Language", value: response.language?.toUpperCase() || "EN" },
      { label: "Intent", value: response.intent || "unknown" },
      { label: "Safe", value: response.safety?.is_safe ? "Yes" : "No" },
      { label: "Sources", value: String(response.source_chunks?.length || 0) },
    ];
  }, [response]);

  const fetchChat = async () => {
    if (!query.trim()) {
      setMessage("Please enter a question or use the mic to send audio.");
      return;
    }
    setLoading(true);
    setMessage("");
    setAudioUrl("");
    try {
      const result = await fetch(`${apiBase}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query,
          language: transcriptLanguage || undefined,
        }),
      });
      if (!result.ok) {
        throw new Error(`Chat failed: ${result.statusText}`);
      }
      const payload = await result.json();
      setResponse(payload);
    } catch (error) {
      setMessage(error.message);
    } finally {
      setLoading(false);
    }
  };

  const fetchTranscript = async (file) => {
    if (!file) return;
    setLoading(true);
    setMessage("");
    try {
      const form = new FormData();
      form.append("file", file);
      const result = await fetch(`${apiBase}/speech-to-text`, {
        method: "POST",
        body: form,
      });
      if (!result.ok) {
        throw new Error(`Speech-to-text failed: ${result.statusText}`);
      }
      const payload = await result.json();
      setTranscript(payload.text || "");
      setTranscriptLanguage(payload.language || "");
      if (payload.text) {
        setQuery(payload.text);
        setMessage("Transcript loaded. You can review it before sending.");
      } else if (payload.error) {
        setMessage(`Speech-to-text failed: ${payload.error}`);
      } else {
        setMessage("No speech was detected in the audio.");
      }
    } catch (error) {
      setMessage(error.message);
    } finally {
      setLoading(false);
    }
  };

  const fetchTts = async (text) => {
    if (!text) {
      setMessage("No text available for TTS.");
      return;
    }
    setLoading(true);
    setMessage("");
    try {
      const result = await fetch(`${apiBase}/text-to-speech`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      if (!result.ok) {
        const fallbackMessage = "Backend TTS unavailable. Using browser speech instead.";
        if ("speechSynthesis" in window) {
          const utterance = new SpeechSynthesisUtterance(text);
          window.speechSynthesis.cancel();
          window.speechSynthesis.speak(utterance);
          setMessage(fallbackMessage);
          return;
        }
        throw new Error(`TTS failed: ${result.statusText}`);
      }
      const blob = await result.blob();
      const url = URL.createObjectURL(blob);
      setAudioUrl(url);
    } catch (error) {
      setMessage(error.message);
    } finally {
      setLoading(false);
    }
  };

  const handleFileChange = async (event) => {
    const file = event.target.files[0];
    if (!file) return;
    setInputFile(file);
    await fetchTranscript(file);
  };

  const handleRecording = async () => {
    if (recording) {
      recorder.stop();
      setRecording(false);
      return;
    }

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setMessage("Microphone access is not supported by this browser.");
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      chunksRef.current = [];
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };
      mediaRecorder.onstop = async () => {
        const blob = new Blob(chunksRef.current, { type: "audio/webm" });
        const file = new File([blob], "recording.webm", { type: "audio/webm" });
        setInputFile(file);
        await fetchTranscript(file);
        stream.getTracks().forEach((track) => track.stop());
      };
      mediaRecorder.start();
      setRecorder(mediaRecorder);
      setRecording(true);
      setMessage("Recording in progress. Click again to stop.");
    } catch (error) {
      setMessage(`Microphone access failed: ${error.message}`);
    }
  };

  const handlePlayAnswerAudio = async () => {
    if (!response) {
      setMessage("Generate a response first before requesting audio.");
      return;
    }
    const text = `${response.answer.title}\n${response.answer.steps.join(" ")}`;
    await fetchTts(text);
  };

  const applySuggestion = (text) => {
    setQuery(text);
    setTranscript("");
    setTranscriptLanguage("");
    setMessage("");
  };

  return (
    <div className="page-shell">
      <div className="page-glow page-glow-left" />
      <div className="page-glow page-glow-right" />

      <main className="app-shell">
        <section className="hero-card">
          <div className="hero-copy">
            <span className="eyebrow">India Government Service Assistant</span>
            <h1>Ask by text, microphone, or audio file and get structured answers fast.</h1>
            <p>
              Grounded responses, official links, multilingual support, and a clean JSON-backed
              interface for passport, Aadhaar, PF, and state e-service workflows.
            </p>
          </div>
          <div className="hero-stats">
            <div className="stat-card">
              <span className="stat-label">Modes</span>
              <strong>Text, Mic, Upload</strong>
            </div>
            <div className="stat-card">
              <span className="stat-label">Output</span>
              <strong>Structured + Source-backed</strong>
            </div>
            <div className="stat-card">
              <span className="stat-label">Best For</span>
              <strong>Passport, Aadhaar, PF</strong>
            </div>
          </div>
        </section>

        <div className="workspace-grid">
          <section className="panel composer-panel">
            <div className="panel-header">
              <div>
                <span className="panel-kicker">Input</span>
                <h2>Compose your request</h2>
              </div>
              <span className={`state-pill ${loading ? "state-busy" : "state-ready"}`}>
                {loading ? "Working" : "Ready"}
              </span>
            </div>

            <textarea
              className="query-box"
              value={query}
              onChange={(event) => {
                const nextValue = event.target.value;
                setQuery(nextValue);
                if (transcript && nextValue !== transcript) {
                  setTranscriptLanguage("");
                }
              }}
              placeholder="Ask about passport, Aadhaar, PF, or a state service portal..."
              rows={6}
            />

            <div className="actions">
              <button onClick={fetchChat} disabled={loading} className="primary-btn">
                {loading ? "Processing..." : "Send Query"}
              </button>
              <button
                onClick={handleRecording}
                className={`secondary-btn ${recording ? "recording" : ""}`}
              >
                {recording ? "Stop Recording" : "Use Mic"}
              </button>
              <label className="upload-btn">
                Upload Audio
                <input type="file" accept="audio/*" onChange={handleFileChange} />
              </label>
            </div>

            <div className="meta-row">
              <div className="meta-chip">
                <span>Transcript language</span>
                <strong>{transcriptLanguage || "Auto"}</strong>
              </div>
              <div className="meta-chip">
                <span>Audio file</span>
                <strong>{inputFile?.name || "None selected"}</strong>
              </div>
            </div>

            <div className="suggestions-strip">
              {suggestions.map((item) => (
                <button
                  key={item}
                  type="button"
                  className="suggestion-chip"
                  onClick={() => applySuggestion(item)}
                >
                  {item}
                </button>
              ))}
            </div>

            {transcript ? (
              <div className="transcript-card">
                <div className="mini-header">
                  <span className="mini-badge">Transcript</span>
                  {transcriptLanguage ? <span className="mini-badge soft">{transcriptLanguage}</span> : null}
                </div>
                <p>{transcript}</p>
              </div>
            ) : null}

            {message ? (
              <div className={`status-card ${message.toLowerCase().includes("failed") ? "status-error" : "status-info"}`}>
                {message}
              </div>
            ) : null}
          </section>

          <section className="panel response-panel">
            {response ? (
              <>
                <div className="panel-header">
                  <div>
                    <span className="panel-kicker">Answer</span>
                    <h2>{response.answer.title}</h2>
                  </div>
                  <button onClick={handlePlayAnswerAudio} disabled={loading} className="ghost-btn">
                    Play Audio
                  </button>
                </div>

                <div className="summary-grid">
                  {answerSummary.map((item) => (
                    <div key={item.label} className="summary-card">
                      <span>{item.label}</span>
                      <strong>{item.value}</strong>
                    </div>
                  ))}
                </div>

                {hasItems(structuredAnswer.steps) ? (
                  <div className="content-card">
                    <div className="mini-header">
                      <h3>Steps</h3>
                    </div>
                    <ol className="ordered-list">
                      {structuredAnswer.steps.map((step, index) => (
                        <li key={index}>{step}</li>
                      ))}
                    </ol>
                  </div>
                ) : null}

                <div className="dual-grid">
                  {hasItems(structuredAnswer.documents) ? (
                    <div className="content-card compact-card">
                      <div className="mini-header">
                        <h3>Documents</h3>
                      </div>
                      <ul className="bullet-list">
                        {structuredAnswer.documents.map((doc, index) => (
                          <li key={index}>{doc}</li>
                        ))}
                      </ul>
                    </div>
                  ) : null}

                  {hasItems(structuredAnswer.eligibility) ? (
                    <div className="content-card compact-card">
                      <div className="mini-header">
                        <h3>Eligibility</h3>
                      </div>
                      <ul className="bullet-list">
                        {structuredAnswer.eligibility.map((item, index) => (
                          <li key={index}>{item}</li>
                        ))}
                      </ul>
                    </div>
                  ) : null}
                </div>

                {hasText(structuredAnswer.fees) || hasText(structuredAnswer.processing_time) ? (
                  <div className="attribute-grid">
                    {hasText(structuredAnswer.fees) ? (
                      <div className="content-card compact-card accent-card">
                        <span className="detail-label">Fees</span>
                        <p>{structuredAnswer.fees}</p>
                      </div>
                    ) : null}
                    {hasText(structuredAnswer.processing_time) ? (
                      <div className="content-card compact-card accent-card">
                        <span className="detail-label">Processing Time</span>
                        <p>{structuredAnswer.processing_time}</p>
                      </div>
                    ) : null}
                  </div>
                ) : null}

                {hasItems(structuredAnswer.official_links) ? (
                  <div className="content-card">
                    <div className="mini-header">
                      <h3>Official Links</h3>
                    </div>
                    <ul className="link-list">
                      {structuredAnswer.official_links.map((link, index) => (
                        <li key={index}>
                          <a href={link} target="_blank" rel="noreferrer">
                            {link}
                          </a>
                        </li>
                      ))}
                    </ul>
                  </div>
                ) : null}

                {hasItems(response.follow_up_questions) ? (
                  <div className="content-card">
                    <div className="mini-header">
                      <h3>Follow-up Questions</h3>
                    </div>
                    <ul className="bullet-list">
                      {response.follow_up_questions.map((question, index) => (
                        <li key={index}>{question}</li>
                      ))}
                    </ul>
                  </div>
                ) : null}

                <div className="content-card compact-card">
                  <div className="mini-header">
                    <h3>Safety</h3>
                    <span className={`safety-pill ${response.safety?.is_safe ? "safe" : "unsafe"}`}>
                      {response.safety?.is_safe ? "Safe" : "Restricted"}
                    </span>
                  </div>
                  {response.safety?.message ? <p>{response.safety.message}</p> : <p>No safety issues detected.</p>}
                </div>

                <details className="sources-card">
                  <summary>View source evidence</summary>
                  <ul className="bullet-list">
                    {response.source_chunks?.map((chunk, index) => (
                      <li key={index}>{chunk}</li>
                    ))}
                  </ul>
                </details>

                {audioUrl ? <audio controls src={audioUrl} className="audio-player" /> : null}
              </>
            ) : (
              <div className="empty-state">
                <span className="panel-kicker">Response Preview</span>
                <h2>No answer yet</h2>
                <p>
                  Start with a typed question, a microphone recording, or an uploaded audio file.
                  Your structured answer will appear here with steps, documents, timing, and official links.
                </p>
              </div>
            )}
          </section>
        </div>
      </main>
    </div>
  );
}

export default App;
