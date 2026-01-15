import { useState } from 'react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import './App.css';

function App() {
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!question.trim()) return;

    // 보낼 질문 임시 저장 후 입력창 즉시 비우기
    const sendData = question;
    setQuestion(''); 
    
    setLoading(true);
    setAnswer(''); 
    
    try {
      const response = await axios.post('http://127.0.0.1:8000/chat', {
        question: sendData
      });
      setAnswer(response.data.answer);
    } catch (error) {
      setAnswer("### ⚠️ 에러\n서버 연결에 실패했습니다.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page-wrapper">
      <div className="content-container">
        <header>
          <h1>질의응답 AI</h1>
          <hr />
        </header>

        <div className="answer-box">
          {loading ? (
            <div className="loading-container">
              <div className="spinner"></div>
              {/* CSS에 추가한 loading-text 클래스 적용 */}
              <p className="loading-text">답변을 생성 중입니다...</p>
            </div>
          ) : (
            <div className="answer-text">
              {answer && (
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {answer}
                </ReactMarkdown>
              )}
            </div>
          )}
        </div>

        <form onSubmit={handleSubmit} className="input-form">
          <input
            type="text"
            className="chat-input"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="궁금한 내용을 입력하세요..."
          />
          <button type="submit" className="submit-button" disabled={loading}>
            질문하기
          </button>
        </form>
      </div>
    </div>
  );
}

export default App;