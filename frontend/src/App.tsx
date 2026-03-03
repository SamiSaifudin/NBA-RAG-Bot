import { useState, useRef, useEffect } from "react"
import "./App.css"

interface ApiResponse {
  answer: string
}

interface Message {
  role: "user" | "bot"
  content: string
}

function App() {
  const [question, setQuestion] = useState<string>("")
  const [messages, setMessages] = useState<Message[]>([])
  const [loading, setLoading] = useState<boolean>(false)
  const [error, setError] = useState<string>("")
  const bottomRef = useRef<HTMLDivElement>(null)
  const [history, setHistory] = useState<{role: string, content: string}[]>([])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages, loading])

  const askQuestion = async () => {
    if (!question.trim()) return
    setError("")

    const userMessage: Message = { role: "user", content: question }
    setMessages((prev) => [...prev, userMessage])
    setQuestion("")
    setLoading(true)

    try {
      const response = await fetch("http://localhost:8000/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, history })
      })

      if (!response.ok) throw new Error("Something went wrong")

      const data: ApiResponse = await response.json()
      const botMessage: Message = { role: "bot", content: data.answer }
      setMessages((prev) => [...prev, botMessage])
      setHistory(prev => [
        ...prev,
        { role: "user", content: question },
        { role: "assistant", content: data.answer }
      ])
    } catch (err) {
      setError("Failed to get an answer. Please try again.")
    } finally {
      setLoading(false)
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") askQuestion()
  }

  return (
    <div className="container">
      <div className="header">
        <h1>ClutchQuery</h1>
        <p className="subtitle">2025-26 NBA Season Stats</p>
      </div>
      <div className="chat-window">
        {messages.map((msg, index) => (
          <div key={index} className={`message ${msg.role}`}>
            <span className="message-label">
              {msg.role === "user" ? "You" : "NBA Bot"}
            </span>
            <div className="message-bubble">{msg.content}</div>
          </div>
        ))}
        {loading && (
          <div className="message bot">
            <span className="message-label">NBA Bot</span>
            <div className="message-bubble">Thinking...</div>
          </div>
        )}
        {error && <p className="error">{error}</p>}
        <div ref={bottomRef} />
      </div>
      <div className="input-row">
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask ClutchQuery about the 2025-26 NBA season..."
          disabled={loading}
        />
        <button onClick={askQuestion} disabled={loading}>
          {loading ? "..." : "Ask"}
        </button>
      </div>
    </div>
  )
}

export default App