## ⚙️ 1. Prerequisites on Your Laptop

Before running any commands, make sure:
1. **Docker Desktop is open and running** on your Windows laptop.
   *(In Docker Desktop Settings > General, ensure **"Use the WSL 2 based engine"** is checked).*
2. Your standard NVIDIA GPU drivers are installed.
3. Open your terminal (**Git Bash** or **PowerShell**) inside your existing project repository folder.

---

## 💻 2. Step-by-Step Launch Commands

### Step 1: Sync with the latest `main` branch
Fetch all the merged teammate updates and Docker configurations:
```bash
git checkout main
git pull origin main
```

---

### Step 2: Set up your environment file (`.env`)
Create your local environment file from the example:
```bash
cp .env.example .env
```
Open `.env` in Notepad or VS Code, find line 69, and paste the Gemini API key:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```
*(Save and close the file. Don't worry, `.env` is gitignored so your key won't be pushed to GitHub).*

---

### Step 3: Pull our pre-built team images from GHCR
Instead of spending 45+ minutes compiling deep learning libraries and PyTorch locally on your machine, download our pre-compiled images directly from the GitHub registry:
```bash
docker compose pull
```
*(Docker will download the layers over your internet in a few minutes).*

---

### Step 4: Launch the full multi-agent cluster!
```bash
docker compose up -d
```

**What happens behind the scenes:**
* 🗄️ `qdrant` starts on port `6333` (Vector DB storage).
* 📄 `doc-processor-api` (Zeina) starts on port `8002` (Docling layout OCR).
* 🔍 `retrieval-api` (Salma) starts on port `8003` with **RTX 3060 GPU acceleration**.
* 🧠 `agent-service` (Youssef) starts on port `8004` (LangGraph + Gemini reasoning).
* 🛡️ `answer-validator-api` (Omar) starts on port `8005` (Strict schema & arithmetic).
* 🌐 `orchestrator-api` (Ahmed) starts on port `8001` (Central API gateway).
* 📊 `eval-service` (Khaled) starts on port `8006` (Benchmarking & Langfuse).
* 🎨 `ui-service` (Ahmed) starts on port `8000` (Gradio Web UI).

---

## 🌟 3. Open the Application

Once the command finishes, open your browser to:

👉 **http://localhost:8000**

You will see the LEDGER Gradio interface live! You can:
1. Upload financial documents.
2. Ask complex multi-hop financial reasoning questions.
3. View evidence chunks, calculated formulas, and validation traces.

---

## 🛠️ Useful Commands

* **Check the status of all services:**
  ```bash
  docker compose ps
  ```
* **View live logs of all services:**
  ```bash
  docker compose logs -f
  ```
* **View logs of a specific service (e.g., your retrieval API or agent):**
  ```bash
  docker compose logs -f retrieval-api
  docker compose logs -f agent-service
  ```
* **Stop the cluster when you are done:**
  ```bash
  docker compose down
  ```
