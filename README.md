
# 📰 Web scrapper Toolkit

This is a **Python desktop application** that allows users to **extract and summarize articles** from any URL using a **user-friendly GUI interface**. It also provides features to export the summary as a **PDF** or **audio file**, making it perfect for students, researchers, and readers on-the-go.

## 🚀 Features

- ✅ Enter any article URL and fetch the full content
- 🧠 Summarize long articles using NLP-based techniques
- 🖨️ Export summary to a PDF file
- 🔊 Convert summary to audio (text-to-speech)
- 💡 Easy-to-use and intuitive GUI (built using `Tkinter` and `CustomTkinter`)

## 🖥️ Technologies Used

- Python 3.x
- Tkinter / CustomTkinter (for GUI)
- `requests` & `newspaper3k` (for web scraping & article parsing)
- `pyttsx3` (for text-to-speech)
- `fpdf` (for PDF generation)

## 📦 Installation

Make sure you have Python 3.x installed.

1. **Clone the repository**

```bash
git clone https://github.com/Aruncodings/web_scrapper_toolkit.git
cd article-summarizer-gui
```

2. **Install required packages**

```bash
pip install -r requirements.txt
```

3. **Run the application**

```bash
python gui.py
```

## 📝 How It Works

- The GUI lets you enter a URL of a news article.
- Upon clicking "Summarize":
  - The article is fetched using `newspaper3k`.
  - The content is summarized and displayed.
- Users can export the summary as a PDF or convert it into an audio file using built-in buttons.

## 📁 Project Structure

```
web_scrapper_toolkit/
│
├── gui.py          # GUI layout and interaction logic
├── main.py         # Main application logic and function bindings
├── requirements.txt# dependencies list
└── README.md       # You're here!
```

## 🧑‍💻 Author

**Arun**  
*Aspiring Web Developer with a passion for Python-based automation and GUI applications.*  
📫 (https://www.linkedin.com/in/arunkumar-mahendiran)

## 📜 License

This project is open source and available under the [MIT License](LICENSE).
