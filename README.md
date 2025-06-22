# 🧠 Alfred

Alfred is a modular command-line reasoning assistant powered by LLMs. It supports OpenAI, Anthropic, and Ollama (e.g., DeepSeek), offering a local-friendly environment with colorized output and API key protection.

---

## 🚀 Features

- ✅ Toggle between **OpenAI**, **Anthropic Claude**, and **Ollama DeepSeek**
- 🧪 Interactive command-line interface
- ⏱️ Displays model response timing
- 🌈 Colored terminal messages via `colorama`
- 🔐 Environment variable management using `.env` (auto-ignored)

---
🧠 Prerequisite: Download open_deep_research
This project depends on an external module named open_deep_research for advanced document understanding and deep semantic research.

📥 How to set it up
Please clone or download the repository next to the Alfred folder like this:

Copy
Edit
INTERNSHIPS/
├── Alfred/
├── open_deep_research/
Command to clone it:

bash
Copy
Edit
git clone https://github.com/langchain-ai/open_deep_research.git
⚠️ Ensure both Alfred and open_deep_research are in the same parent directory, so the internal imports and relative paths work properly.

## 🛠️ Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/KnightCorp/Alfred.git
cd Alfred
```

### 2. Set Up the Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Add API Keys

```bash
cd src
touch .env
```

```env
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
```

> ✅ `.env` is excluded from version control via `.gitignore`.

### 5. Install Ollama (for local LLMs like DeepSeek)

Ollama lets you run models like `deepseek-llm` or `llama3` locally on your machine.

#### Mac

```bash
brew install ollama
ollama run deepseek-llm
```

#### Linux

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama run deepseek-llm
```

#### Windows

Download the Windows installer from:  
👉 https://ollama.com/download

Then run:

```bash
ollama run deepseek-llm
```

> 🧠 You can replace `deepseek-llm` with any other supported model like `llama3`, `mistral`, etc.

---

### 6. Run the Application

```bash
python core.py
```

---

## 📁 Project Structure

```
ALFRED/
├── src/
│   └── core/              # Source code (LLM logic, CLI handler)
│   └── .env               # API keys (ignored via .gitignore)
├── .gitignore             # Ignores sensitive files like .env
├── LICENSE                # Licensing information
├── README.md              # Project documentation
├── requirements.txt       # Python dependencies
```

---


## 🤖 Supported Models

| Model         | Backend     | Notes                          |
|---------------|-------------|---------------------------------|
| GPT-3.5/4     | OpenAI      | Needs `OPENAI_API_KEY`         |
| Claude 3.5    | Anthropic   | Needs `ANTHROPIC_API_KEY`      |
| DeepSeek      | Ollama LLM  | Runs locally via Ollama setup  |

---

## 📄 License

This project is **proprietary and closed source**.  
All rights reserved. Unauthorized use or distribution is strictly prohibited.
