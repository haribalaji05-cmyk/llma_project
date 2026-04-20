import { useEffect, useMemo, useRef, useState } from "react";

const apiBase = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

function App() {
  const [query, setQuery] = useState("");
  const [response, setResponse] = useState(null);
  const [transcript, setTranscript] = useState("");
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

  const fetchChat = async () => {
    if (!query.trim()) {
      setMessage("Please enter a question or use the mic to send audio.");
      return;
    }
    setLoading(true);
    setMessage("");
    try {
      const result = await fetch(`${apiBase}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
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
      if (payload.text) {
        setQuery(payload.text);
        setMessage("Transcript loaded. You can edit before sending.");
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
      setMessage("Recording... click again to stop.");
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

  return (
    <div className="app-shell">
      <header>
        <h1>GovAssist</h1>
        <p>Ask government service questions, use audio input, and get structured answers.</p>
      </header>

      <section className="controls">
        <textarea
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Type your question here..."
          rows={4}
        />

        <div className="actions">
          <button onClick={fetchChat} disabled={loading}>
            {loading ? "Loading..." : "Send Query"}
          </button>
          <button onClick={handleRecording} className={recording ? "recording" : ""}>
            {recording ? "Stop Recording" : "Use Mic"}
          </button>
          <label className="file-upload">
            Upload Audio
            <input type="file" accept="audio/*" onChange={handleFileChange} />
          </label>
        </div>

        {transcript ? (
          <div className="transcript-panel">
            <strong>Transcript:</strong>
            <p>{transcript}</p>
          </div>
        ) : null}

        {message ? <div className="status-message">{message}</div> : null}
      </section>

      <section className="response-panel">
        {response ? (
          <>
            <div className="response-header">
              <div>
                <strong>Language:</strong> {response.language}
              </div>
              <div>
                <strong>Intent:</strong> {response.intent}
              </div>
            </div>
            <div className="response-content">
              <h2>{response.answer.title}</h2>
              <div className="answer-block">
                <h3>Steps</h3>
                <ol>
                  {structuredAnswer.steps?.map((step, index) => (
                    <li key={index}>{step}</li>
                  ))}
                </ol>
              </div>
              <div className="answer-block">
                <h3>Documents</h3>
                <ul>
                  {structuredAnswer.documents?.map((doc, index) => (
                    <li key={index}>{doc}</li>
                  ))}
                </ul>
              </div>
              <div className="answer-attributes">
                <div>
                  <strong>Fees:</strong> {structuredAnswer.fees}
                </div>
                <div>
                  <strong>Processing time:</strong> {structuredAnswer.processing_time}
                </div>
              </div>
              <div className="answer-block">
                <h3>Eligibility</h3>
                <ul>
                  {structuredAnswer.eligibility?.map((item, index) => (
                    <li key={index}>{item}</li>
                  ))}
                </ul>
              </div>
              <div className="answer-block">
                <h3>Official Links</h3>
                <ul>
                  {structuredAnswer.official_links?.map((link, index) => (
                    <li key={index}>{link}</li>
                  ))}
                </ul>
              </div>
              <div className="answer-block">
                <h3>Follow-up Questions</h3>
                <ul>
                  {response.follow_up_questions?.map((question, index) => (
                    <li key={index}>{question}</li>
                  ))}
                </ul>
              </div>
              <button onClick={handlePlayAnswerAudio} disabled={loading}>
                Play Answer Audio
              </button>
              {audioUrl ? (
                <audio controls src={audioUrl} className="audio-player" />
              ) : null}
            </div>
          </>
        ) : (
          <div className="empty-state">No answer yet. Send a query to begin.</div>
        )}
      </section>
    </div>
  );
}

export default App;
