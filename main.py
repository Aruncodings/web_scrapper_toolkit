
import os
import re
import random
import logging
import tempfile
import requests
import nltk
import time
import sqlite3
import bcrypt
import shutil
from datetime import datetime
from bs4 import BeautifulSoup
from newspaper import Article, Config
from fpdf import FPDF
from gtts import gTTS
from urllib.parse import urlparse
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from PIL import Image
import readability
from textblob import TextBlob
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist


try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("summarizer.log"), logging.StreamHandler()])
logger = logging.getLogger(__name__)

TEMP_DIR = tempfile.mkdtemp()
REQUEST_TIMEOUT = 30  # seconds


from googletrans import Translator

translator = Translator()

news_config = Config()
news_config.browser_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
news_config.request_timeout = REQUEST_TIMEOUT


class UserDB:
    def __init__(self):
        self.conn = sqlite3.connect('users.db', check_same_thread=False)
        self.create_tables()

    def create_tables(self):
        cursor = self.conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS users
                         (id INTEGER PRIMARY KEY AUTOINCREMENT,
                          username TEXT UNIQUE NOT NULL,
                          password_hash TEXT NOT NULL,
                          email TEXT,
                          created_at DATETIME)''')
        self.conn.commit()

    def create_user(self, username, password, email):
        hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
        try:
            cursor = self.conn.cursor()
            cursor.execute('''INSERT INTO users 
                            (username, password_hash, email, created_at)
                            VALUES (?, ?, ?, ?)''',
                           (username, hashed, email, datetime.now()))
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def verify_user(self, username, password):
        cursor = self.conn.cursor()
        cursor.execute('SELECT password_hash FROM users WHERE username = ?', (username,))
        result = cursor.fetchone()
        if result and bcrypt.checkpw(password.encode(), result[0]):
            return True
        return False

    def close(self):
        self.conn.close()


class ArticleProcessor:
    def __init__(self):
        self.current_metadata = None
        self.current_summary = None
        self.current_url = None
        self.current_images = []
        self.current_screenshot_path = None
        self._ensure_temp_dir()

    def _ensure_temp_dir(self):
        """Ensure temp directory exists and is writable"""
        try:
            os.makedirs(TEMP_DIR, exist_ok=True)
          
            test_file = os.path.join(TEMP_DIR, 'test.tmp')
            with open(test_file, 'w') as f:
                f.write('test')
            os.unlink(test_file)
        except Exception as e:
            logger.error(f"Temp directory error: {e}")
            raise RuntimeError(f"Could not access temp directory: {TEMP_DIR}")

    @staticmethod
    def is_valid_url(url):
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except ValueError:
            return False

    @staticmethod
    def sanitize_filename(filename):
        filename = re.sub(r'[^\w\s.-]', '', filename)
        return filename.strip().replace(' ', '_')[:50]

    def extract_images_from_url(self, url, max_images=None, min_size_kb=10, max_size_mb=10):
        """Extract images from a webpage with better filtering and no default limit

        Args:
            url: URL of the webpage to extract images from
            max_images: Maximum number of images to return (None for no limit)
            min_size_kb: Minimum image size in KB to consider (default 10KB)
            max_size_mb: Maximum image size in MB to consider (default 10MB)

        Returns:
            List of image URLs sorted by likely importance
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }

            if not self.is_valid_url(url):
                logger.error(f"Invalid URL provided for image extraction: {url}")
                return []

            try:
                response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
                response.raise_for_status()
            except requests.RequestException as e:
                logger.error(f"Failed to fetch URL for image extraction: {e}")
                return []

            soup = BeautifulSoup(response.content, 'html.parser')
            images = soup.find_all('img')
            image_data = []
            base_url = urlparse(url)
            base_domain = f"{base_url.scheme}://{base_url.netloc}"

            for img in images:
                img_url = img.get('src') or img.get('data-src') or img.get('data-lazy-src')
                if not img_url:
                    continue

                # Fix URL formatting
                try:
                    if img_url.startswith('//'):
                        img_url = f"{base_url.scheme}:{img_url}"
                    elif img_url.startswith('/'):
                        img_url = f"{base_domain}{img_url}"
                    elif not img_url.startswith(('http', 'https')):
                        img_url = f"{base_domain}/{img_url.lstrip('/')}"

                    # Validate the constructed URL
                    parsed = urlparse(img_url)
                    if not all([parsed.scheme, parsed.netloc]):
                        continue
                except Exception as e:
                    logger.debug(f"URL formatting error for {img_url}: {e}")
                    continue

                # Skip unwanted images
                if any([img_url.startswith('data:'),
                        'icon' in img_url.lower(),
                        'logo' in img_url.lower(),
                        img_url.endswith('.svg'),
                        'pixel' in img_url.lower(),
                        'track' in img_url.lower()]):
                    continue

                # Get image dimensions from attributes if available
                width = int(img.get('width', 0))
                height = int(img.get('height', 0))

                # Score image by likely importance
                score = 0
                if width >= 300 and height >= 300:  # Reasonable size
                    score += 2
                elif width >= 100 and height >= 100:
                    score += 1

                # Check for alt text
                if img.get('alt'):
                    score += 1

                # Check for class names that might indicate importance
                class_names = img.get('class', [])
                if class_names and any(x in ['hero', 'main', 'feature', 'content']
                                       for x in class_names):
                    score += 2

                image_data.append({
                    'url': img_url,
                    'score': score,
                    'width': width,
                    'height': height
                })

            # Filter and sort images
            filtered_images = []
            for img in image_data:
                try:
                    # Get image size via HEAD request
                    img_head = requests.head(img['url'], headers=headers, timeout=5)

                    if not img_head.headers.get('Content-Type', '').startswith('image/'):
                        continue

                    content_length = int(img_head.headers.get('Content-Length', 0))

                    # Size filtering
                    if content_length < min_size_kb * 1024:  # Smaller than min size
                        continue
                    if content_length > max_size_mb * 1024 * 1024:  # Larger than max size
                        continue

                    filtered_images.append(img)

                except Exception as e:
                    logger.debug(f"Skipping image {img['url']}: {str(e)}")
                    continue

            # Sort by score (highest first), then by size
            filtered_images.sort(key=lambda x: (-x['score'], x['width'] * x['height']), reverse=True)

            # Extract just the URLs
            image_urls = [img['url'] for img in filtered_images]

            # Apply max_images limit if specified
            if max_images is not None:
                image_urls = image_urls[:max_images]

            return image_urls

        except Exception as e:
            logger.error(f"Image extraction failed: {str(e)}")
            return []

    def capture_full_page_screenshot(self, url):
        try:
            # Updated ChromeDriver configuration
            chrome_options = Options()
            chrome_options.add_argument("--headless=new")  # Updated headless argument
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920,1080")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--no-sandbox")

            # Use Chrome service with specific version
            driver = webdriver.Chrome(options=chrome_options)

            driver.get(url)
            time.sleep(5)  # Increase wait time for page to load

            # Get page dimensions and set window size accordingly
            total_height = driver.execute_script("return document.body.scrollHeight")
            driver.set_window_size(1920, total_height)

            # Capture screenshot
            screenshot_path = os.path.join(TEMP_DIR, f"screenshot_{int(time.time())}.png")
            driver.save_screenshot(screenshot_path)
            driver.quit()

            self.current_screenshot_path = screenshot_path
            return screenshot_path

        except Exception as e:
            logger.error(f"Screenshot capture failed: {e}")
            raise

    def summarize_article(self, url, accuracy_level="medium", summary_length="medium",
                          format_style="standard"):
        if not self.is_valid_url(url):
            raise ValueError("Invalid URL provided")

        article = Article(url, config=news_config)
        try:
            article.download()
            article.parse()
        except Exception as e:
            logger.error(f"Failed to download or parse article: {e}")
            raise Exception("Could not access the article. Please check if the URL is correct and accessible.")

        html_content = article.html
        title = article.title if article.title else "Untitled Article"
        publish_date = article.publish_date.strftime('%Y-%m-%d') if article.publish_date else "Unknown"
        author = ', '.join(article.authors) if article.authors else "Unknown"

        full_text = article.text
        if not full_text:
            raise Exception(
                "Could not extract text from the article. The page might be using JavaScript to load content.")

        word_count = len(full_text.split())
        sentences = sent_tokenize(full_text)
        sentence_count = len(sentences)

        avg_wpm = 200
        reading_time = round(word_count / avg_wpm)

        try:
            article.nlp()
            default_summary = article.summary
            if not default_summary:
                default_summary = ' '.join(sentences[:min(5, len(sentences))])
        except Exception as e:
            logger.error(f"NLP processing failed: {e}")
            default_summary = ' '.join(sentences[:min(5, len(sentences))])

        # Calculate target word counts for each summary length
        target_word_counts = {
            'short': min(1000, word_count // 10),  # ~10% of article or 1000 words max
            'medium': min(2000, word_count // 2),  # ~50% of article or 2000 words max
            'long': min(4000, word_count)  # Up to 4000 words or full article
        }

        if summary_length == 'short':
            # For short summary, take about 10% of the article
            sentences = sent_tokenize(full_text)
            summary = ' '.join(sentences[:int(len(sentences) * 0.1)])
        elif summary_length == 'medium':
            # For medium summary, take about 50% of the article
            sentences = sent_tokenize(full_text)
            summary = ' '.join(sentences[:int(len(sentences) * 0.5)])
        elif summary_length == 'long':
            # For long summary, take up to 4000 words or full article
            sentences = sent_tokenize(full_text)
            summary = full_text if word_count <= 4000 else ' '.join(sentences[:len(sentences)])
        else:
            summary = default_summary

        # If summary is still too short, add more content
        current_word_count = len(summary.split())
        if summary_length in target_word_counts and current_word_count < target_word_counts[summary_length]:
            remaining_words_needed = target_word_counts[summary_length] - current_word_count
            additional_sentences = []

            # Get sentences not already in summary
            all_sentences = sent_tokenize(full_text)
            summary_sentences = sent_tokenize(summary)
            remaining_sentences = [s for s in all_sentences if s not in summary_sentences]

            # Add sentences until we reach target word count
            word_counter = 0
            for sentence in remaining_sentences:
                sentence_words = len(sentence.split())
                if word_counter + sentence_words <= remaining_words_needed:
                    additional_sentences.append(sentence)
                    word_counter += sentence_words
                else:
                    break

            if additional_sentences:
                summary += " " + ' '.join(additional_sentences)

        if accuracy_level == 'high':
            summary += f"\n\nPublished on {publish_date} by {author}. Reading time: {reading_time} minutes."
        elif accuracy_level == 'low':
            sentences = sent_tokenize(summary)
            summary = ' '.join(sentences[:max(1, len(sentences) // 2)])

        keywords = self.extract_keywords(full_text)
        sentiment_label, sentiment_value = self.analyze_sentiment(full_text)
        readability_metrics = self.calculate_readability(full_text)
        citations = self.extract_citations(full_text, html_content)
        facts = self.extract_facts(full_text)

        formatted_summary = self.format_summary(summary, format_style)
        related_articles = self.get_related_articles(title, keywords)

        metadata = {
            'title': title,
            'publish_date': publish_date,
            'author': author,
            'word_count': len(summary.split()),
            'sentence_count': len(sent_tokenize(summary)),
            'reading_time': round(len(summary.split()) / avg_wpm),
            'accuracy_level': accuracy_level,
            'summary_length': summary_length,
            'format_style': format_style,
            'keywords': keywords,
            'sentiment': {
                'label': sentiment_label,
                'value': sentiment_value
            },
            'readability': readability_metrics,
            'citations': citations,
            'facts': facts,
            'related_articles': related_articles
        }

        self.current_metadata = metadata
        self.current_summary = formatted_summary
        self.current_url = url
        self.current_images = self.extract_images_from_url(url)

        return metadata, formatted_summary

    def generate_pdf(self, output_path=None):
        """Generate a PDF summary with robust Unicode and text handling"""
        if not self.current_summary or not self.current_metadata:
            raise ValueError("No summary to save. Please summarize an article first.")

        # Try ReportLab first (more robust), fall back to FPDF if not available
        try:
            # Import reportlab
            from reportlab.lib.pagesizes import A4
            from reportlab.lib import colors
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
            from reportlab.pdfbase import pdfmetrics
            from reportlab.pdfbase.ttfonts import TTFont
            import os

            logger.info("Using ReportLab for PDF generation")

            # Determine output path
            if not output_path:
                output_path = os.path.join(TEMP_DIR,
                                           f"{self.sanitize_filename(self.current_metadata.get('title', 'untitled'))}_summary.pdf")

            # Set up the document
            doc = SimpleDocTemplate(
                output_path,
                pagesize=A4,
                rightMargin=15,
                leftMargin=15,
                topMargin=15,
                bottomMargin=15
            )

            # Add fonts - First ensure we have at least one Unicode-compatible font
            self.ensure_unicode_font_available()  # Reuse the font downloader

            try:
                from fpdf import FPDF_FONT_DIR
                dejavu_path = os.path.join(FPDF_FONT_DIR, "DejaVuSans.ttf")
                if os.path.exists(dejavu_path):
                    pdfmetrics.registerFont(TTFont('DejaVu', dejavu_path))
                    font_name = 'DejaVu'
                else:
                    # Check for Roboto as alternate
                    roboto_path = os.path.join(FPDF_FONT_DIR, "Roboto-Regular.ttf")
                    if os.path.exists(roboto_path):
                        pdfmetrics.registerFont(TTFont('Roboto', roboto_path))
                        font_name = 'Roboto'
                    else:
                        # Fall back to Helvetica which is built-in
                        font_name = 'Helvetica'
            except Exception as e:
                logger.warning(f"Could not register custom font: {e}. Using Helvetica.")
                font_name = 'Helvetica'

            # Create styles
            styles = getSampleStyleSheet()
            title_style = ParagraphStyle(
                'Title',
                parent=styles['Title'],
                fontName=font_name,
                fontSize=16,
                alignment=1,  # Center alignment
                spaceAfter=12
            )

            heading_style = ParagraphStyle(
                'Heading',
                parent=styles['Heading2'],
                fontName=font_name,
                fontSize=14,
                spaceAfter=10
            )

            normal_style = ParagraphStyle(
                'Normal',
                parent=styles['Normal'],
                fontName=font_name,
                fontSize=11,
                spaceBefore=6,
                spaceAfter=6
            )

            # Function to safely handle text for XML/HTML
            def safe_text(text):
                if not text:
                    return ""

                # Replace XML special characters
                text = (str(text)
                        .replace('&', '&amp;')
                        .replace('<', '&lt;')
                        .replace('>', '&gt;')
                        .replace('"', '&quot;')
                        .replace("'", '&#39;'))

                return text

            # Build the document content
            story = []

            # Add title
            title_text = safe_text(self.current_metadata.get('title', 'Untitled Document'))
            if len(title_text) > 100:  # Truncate very long titles
                title_text = title_text[:97] + "..."

            story.append(Paragraph(title_text, title_style))

            # Add metadata as a table for better formatting
            metadata = [
                ["Author:", safe_text(self.current_metadata.get('author', 'Unknown'))],
                ["Published:", safe_text(self.current_metadata.get('publish_date', 'Unknown'))],
                ["Reading Time:", f"{safe_text(str(self.current_metadata.get('reading_time', 'Unknown')))} minutes"]
            ]

            # Add keywords if available
            if 'keywords' in self.current_metadata and self.current_metadata['keywords']:
                keywords = ', '.join(self.current_metadata['keywords'][:5])
                metadata.append(["Keywords:", safe_text(keywords)])

            # Create metadata table
            meta_table = Table(metadata, colWidths=[80, 415])
            meta_table.setStyle(TableStyle([
                ('FONT', (0, 0), (-1, -1), font_name, 10),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('TEXTCOLOR', (0, 0), (0, -1), colors.gray),
                ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
            ]))

            story.append(meta_table)
            story.append(Spacer(1, 15))

            # Add summary heading
            story.append(Paragraph("Summary", heading_style))

            # Process summary text - split into paragraphs for better handling
            try:
                summary = safe_text(self.current_summary)
                paragraphs = summary.split('\n\n')

                # Convert each paragraph to a Paragraph object
                for p in paragraphs:
                    if p.strip():  # Skip empty paragraphs
                        story.append(Paragraph(p.replace('\n', '<br/>'), normal_style))
            except Exception as e:
                # Fallback for any parsing issues
                logger.error(f"Error formatting summary: {e}")
                # Try to add the entire summary as one chunk
                try:
                    story.append(Paragraph(safe_text(self.current_summary).replace('\n', '<br/>'), normal_style))
                except Exception as e2:
                    logger.error(f"Error with fallback formatting: {e2}")
                    # Last resort - add plain text warning
                    story.append(
                        Paragraph("Error formatting summary content. Please check the original text.", normal_style))

            # Build the PDF
            doc.build(story)
            logger.info(f"PDF successfully created at {output_path}")
            return output_path

        except ImportError:
            logger.info("ReportLab not available. Falling back to FPDF.")
            # Fall back to FPDF version if ReportLab is not available
            return self._generate_pdf_with_fpdf(output_path)
        except Exception as e:
            logger.error(f"PDF generation with ReportLab failed: {e}")
            logger.info("Falling back to FPDF as an alternative")
            try:
                return self._generate_pdf_with_fpdf(output_path)
            except Exception as e2:
                logger.error(f"Both PDF generation methods failed. Last error: {e2}")
                raise

    def _generate_pdf_with_fpdf(self, output_path=None):
        """Fallback PDF generator using FPDF"""
        try:
            from fpdf import FPDF, FPDF_FONT_DIR
            import os
            import re

            # Create PDF with slightly larger margins to prevent space issues
            pdf = FPDF(orientation='P', unit='mm', format='A4')

            # Set compression to False to avoid issues with certain Unicode characters
            pdf.set_compression(False)

            pdf.add_page()

            # ===== Font Setup with Better Unicode Handling =====
            # First ensure we have at least one Unicode-compatible font
            self.ensure_unicode_font_available()

            # Prioritize fonts with better Unicode support
            font_priority = [
                ('DejaVuSans', {
                    'normal': os.path.join(FPDF_FONT_DIR, 'DejaVuSans.ttf'),
                    'bold': os.path.join(FPDF_FONT_DIR, 'DejaVuSans-Bold.ttf'),
                    'italic': os.path.join(FPDF_FONT_DIR, 'DejaVuSans-Oblique.ttf')
                }),
                ('NotoSans', {
                    'normal': os.path.join(FPDF_FONT_DIR, 'NotoSans-Regular.ttf'),
                    'bold': os.path.join(FPDF_FONT_DIR, 'NotoSans-Bold.ttf')
                }),
                ('Roboto', {
                    'normal': os.path.join(FPDF_FONT_DIR, 'Roboto-Regular.ttf'),
                    'bold': os.path.join(FPDF_FONT_DIR, 'Roboto-Bold.ttf')
                })
            ]

            selected_font = None
            for font_name, font_files in font_priority:
                if os.path.exists(font_files['normal']):
                    try:
                        pdf.add_font(font_name, '', font_files['normal'], uni=True)
                        if 'bold' in font_files and os.path.exists(font_files['bold']):
                            pdf.add_font(font_name, 'B', font_files['bold'], uni=True)
                        if 'italic' in font_files and os.path.exists(font_files['italic']):
                            pdf.add_font(font_name, 'I', font_files['italic'], uni=True)
                        selected_font = font_name
                        break
                    except Exception as e:
                        logger.warning(f"Failed to add font {font_name}: {e}")
                        continue

            if not selected_font:
                # Ultimate fallback - use standard Helvetica (but with limited Unicode)
                selected_font = 'Helvetica'
                logger.warning("Using built-in Helvetica font. Some Unicode characters may not display correctly.")

            # ===== Improved Text Preprocessing =====
            def safe_text(text):
                """Better handling of Unicode text for PDF output"""
                if not text:
                    return ""

                # Replace problematic characters or symbols that might cause issues
                replacements = {
                    "–": "-",  # en dash
                    "—": "-",  # em dash
                    """: '"',  # smart quotes
                    """: '"',
                    "'": "'",
                    "'": "'",
                    "…": "...",
                    "•": "*",
                    "\u2028": " ",  # line separator
                    "\u2029": " "  # paragraph separator
                }

                for old, new in replacements.items():
                    text = text.replace(old, new)

                # Break any extremely long words (longer than 30 chars) that might cause layout issues
                words = text.split()
                max_word_length = 30
                for i, word in enumerate(words):
                    if len(word) > max_word_length:
                        # Insert soft hyphens every 25 characters for long words
                        words[i] = '-'.join([word[j:j + 25] for j in range(0, len(word), 25)])

                processed_text = ' '.join(words)

                # If using a font with limited Unicode support (like Helvetica)
                if selected_font == 'Helvetica':
                    # More aggressive replacement for non-Latin chars
                    return ''.join(c if ord(c) < 128 else '?' for c in processed_text)

                return processed_text

            # ===== PDF Generation with More Robust Error Handling =====
            # Set increased margins to prevent space issues
            pdf.set_margins(15, 15, 15)  # Left, Top, Right margins increased

            # Add title
            pdf.set_font(selected_font, 'B', 14)  # Slightly smaller font size
            title = safe_text(self.current_metadata.get('title', 'Untitled Document'))
            if len(title) > 60:  # Reduced from 70
                title = title[:57] + "..."
            pdf.cell(0, 10, title, 0, 1, 'C')

            # Add metadata with error handling
            pdf.set_font(selected_font, 'I', 10)

            # Use write_html for better text wrapping in metadata
            author = safe_text(self.current_metadata.get('author', 'Unknown'))
            if len(author) > 40:  # Truncate if too long
                author = author[:37] + "..."
            pdf.cell(0, 6, f"Author: {author}", 0, 1)

            pdf.cell(0, 6, f"Published: {safe_text(self.current_metadata.get('publish_date', 'Unknown'))}", 0, 1)
            pdf.cell(0, 6,
                     f"Reading Time: {safe_text(str(self.current_metadata.get('reading_time', 'Unknown')))} minutes", 0,
                     1)

            # Add keywords if available, with length control
            if 'keywords' in self.current_metadata and self.current_metadata['keywords']:
                # Limit each keyword to 15 chars max to prevent overflow
                processed_keywords = []
                for kw in self.current_metadata['keywords'][:5]:
                    if len(kw) > 15:
                        processed_keywords.append(kw[:12] + "...")
                    else:
                        processed_keywords.append(kw)

                keywords = ', '.join(processed_keywords)
                pdf.cell(0, 6, f"Keywords: {safe_text(keywords)}", 0, 1)

            # Add summary with improved wrapping
            pdf.ln(5)
            pdf.set_font(selected_font, 'B', 12)
            pdf.cell(0, 10, "Summary", 0, 1)
            pdf.set_font(selected_font, '', 11)

            # Process summary using multiple approaches to ensure ALL text is saved
            summary = safe_text(self.current_summary)

            # Method 1: Try writing paragraph by paragraph
            try:
                paragraphs = re.split(r'\n\s*\n', summary)

                for paragraph in paragraphs:
                    # Clean up paragraph and ensure it's not empty
                    paragraph = paragraph.strip()
                    if not paragraph:
                        continue

                    try:
                        pdf.multi_cell(0, 6, paragraph)
                        pdf.ln(3)  # Small space between paragraphs
                    except Exception as e:
                        logger.warning(f"Error with paragraph: {e}")
                        # Try ASCII-only version of this paragraph
                        sanitized = ''.join(c if ord(c) < 128 else '?' for c in paragraph)
                        pdf.multi_cell(0, 6, sanitized)
                        pdf.ln(3)

            except Exception as e:
                logger.warning(f"Paragraph method failed: {e}. Using sentence method.")

                # Method 2: Try sentence by sentence
                try:
                    sentences = re.split(r'(?<=[.!?])\s+', summary)
                    for sentence in sentences:
                        sentence = sentence.strip()
                        if not sentence:
                            continue

                        try:
                            pdf.multi_cell(0, 6, sentence)
                        except:
                            # Try ASCII-only version
                            sanitized = ''.join(c if ord(c) < 128 else '?' for c in sentence)
                            pdf.multi_cell(0, 6, sanitized)
                except Exception as e:
                    logger.warning(f"Sentence method failed: {e}. Using small chunks.")

                    # Method 3: Final fallback - write very small chunks with maximum safety
                    chunk_size = 100  # Very small chunks for safety
                    for i in range(0, len(summary), chunk_size):
                        chunk = summary[i:i + chunk_size]
                        try:
                            pdf.multi_cell(0, 6, chunk)
                        except Exception as e:
                            logger.warning(f"Error with chunk: {e}")
                            # Use ASCII-only as last resort
                            sanitized = ''.join(c if ord(c) < 128 else '?' for c in chunk)
                            try:
                                pdf.multi_cell(0, 6, sanitized)
                            except:
                                # If even ASCII fails, just log and continue
                                logger.error(f"Failed to write chunk starting with: {chunk[:20]}...")

            # Regardless of which method worked, add a note about potential formatting
            pdf.ln(5)
            pdf.set_font(selected_font, 'I', 9)
            pdf.multi_cell(0, 5,
                           "Note: Some formatting or special characters may have been simplified for compatibility.")

            # Save the PDF
            if not output_path:
                output_path = os.path.join(TEMP_DIR,
                                           f"{self.sanitize_filename(self.current_metadata.get('title', 'untitled'))}_summary.pdf")

            pdf.output(output_path)
            return output_path

        except Exception as e:
            logger.error(f"FPDF generation failed: {e}")
            raise

    def ensure_unicode_font_available(self):
        """Ensure that at least one Unicode-compatible font is available"""
        try:
            from fpdf import FPDF_FONT_DIR
            import requests
            import os

            # Create font directory if it doesn't exist
            os.makedirs(FPDF_FONT_DIR, exist_ok=True)

            # Prioritize downloading Roboto as a reliable Unicode font
            roboto_url = "https://github.com/google/fonts/raw/main/apache/roboto/Roboto-Regular.ttf"
            roboto_bold_url = "https://github.com/google/fonts/raw/main/apache/roboto/Roboto-Bold.ttf"
            roboto_path = os.path.join(FPDF_FONT_DIR, "Roboto-Regular.ttf")
            roboto_bold_path = os.path.join(FPDF_FONT_DIR, "Roboto-Bold.ttf")

            # Download Roboto Regular if needed
            if not os.path.exists(roboto_path):
                logger.info("Downloading Roboto Regular font for Unicode support")
                response = requests.get(roboto_url)
                if response.status_code == 200:
                    with open(roboto_path, "wb") as f:
                        f.write(response.content)
                else:
                    logger.warning(f"Failed to download Roboto font: {response.status_code}")

            # Download Roboto Bold if needed
            if not os.path.exists(roboto_bold_path):
                logger.info("Downloading Roboto Bold font for Unicode support")
                response = requests.get(roboto_bold_url)
                if response.status_code == 200:
                    with open(roboto_bold_path, "wb") as f:
                        f.write(response.content)

            # Optionally download DejaVu Sans as a backup with excellent Unicode support
            dejavu_url = "https://github.com/dejavu-fonts/dejavu-fonts/raw/master/ttf/DejaVuSans.ttf"
            dejavu_path = os.path.join(FPDF_FONT_DIR, "DejaVuSans.ttf")

            if not os.path.exists(dejavu_path):
                try:
                    logger.info("Downloading DejaVu Sans font for extended Unicode support")
                    response = requests.get(dejavu_url)
                    if response.status_code == 200:
                        with open(dejavu_path, "wb") as f:
                            f.write(response.content)
                except Exception as e:
                    logger.warning(f"Failed to download DejaVu font (using Roboto as fallback): {e}")

        except Exception as e:
            logger.warning(f"Error ensuring Unicode fonts: {e}")

    def generate_audio(self, output_path=None):
        """Generate audio from article summary and save to file"""
        if not self.current_summary:
            raise ValueError("No summary to convert to audio")

        try:
            # Limit text length to avoid errors
            summary_text = self.current_summary
            if len(summary_text) > 5000:
                summary_text = summary_text[:5000] + "... (text truncated for audio conversion)"

            # Remove markdown formatting for better speech
            summary_text = re.sub(r'#+ ', '', summary_text)  # Remove headings
            summary_text = re.sub(r'\*\*(.*?)\*\*', r'\1', summary_text)  # Remove bold
            summary_text = re.sub(r'\*(.*?)\*', r'\1', summary_text)  # Remove italic

            # Generate audio file
            if not output_path:
                output_path = os.path.join(TEMP_DIR, f"summary_audio_{int(time.time())}.mp3")

            tts = gTTS(text=summary_text, lang='en', slow=False)
            tts.save(output_path)
            logger.info(f"Audio saved to {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Audio generation failed: {e}")
            raise

    def extract_keywords(self, text, count=10):
        try:
            words = word_tokenize(text.lower())
            stop_words = set(stopwords.words('english'))
            filtered_words = [word for word in words if word.isalnum() and word not in stop_words and len(word) > 2]
            fdist = FreqDist(filtered_words)
            return [word for word, freq in fdist.most_common(count)]
        except Exception as e:
            logger.error(f"Keyword extraction failed: {e}")
            return ["No keywords available"]

    @staticmethod
    def analyze_sentiment(text):
        try:
            analysis = TextBlob(text)
            polarity = analysis.sentiment.polarity

            if polarity > 0.1:
                return "Positive", polarity
            elif polarity < -0.1:
                return "Negative", polarity
            else:
                return "Neutral", polarity
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return "Unknown", 0.0

    @staticmethod
    def calculate_readability(text):
        try:
            if len(text.split()) < 100:
                return {"grade_level": "Insufficient text", "reading_ease": "N/A"}

            results = readability.getmeasures(text, lang='en')
            grade_level = results['readability grades']['FleschKincaidGradeLevel']
            reading_ease = results['readability grades']['FleschReadingEase']

            if reading_ease >= 90:
                ease_label = "Very Easy"
            elif reading_ease >= 80:
                ease_label = "Easy"
            elif reading_ease >= 70:
                ease_label = "Fairly Easy"
            elif reading_ease >= 60:
                ease_label = "Standard"
            elif reading_ease >= 50:
                ease_label = "Fairly Difficult"
            elif reading_ease >= 30:
                ease_label = "Difficult"
            else:
                ease_label = "Very Difficult"

            grade_label = f"Grade {grade_level:.1f} Level"
            return {"grade_level": grade_label, "reading_ease": ease_label}

        except Exception as e:
            logger.error(f"Readability calculation failed: {e}")
            return {"grade_level": "Unknown", "reading_ease": "Unknown"}

    @staticmethod
    def extract_citations(text, html):
        try:
            soup = BeautifulSoup(html, 'html.parser')
            citations = []

            links = soup.find_all('a')
            for link in links:
                href = link.get('href')
                text = link.get_text().strip()
                if href and ('cite' in href.lower() or 'reference' in text.lower()):
                    citations.append(text)

            citation_pattern = re.compile(r'\[([\d,\s]+)\]|\(([\w\s]+, \d{4})\)')
            matches = citation_pattern.findall(text)

            for match in matches:
                if match[0]:  # Numbered citation
                    citations.append(f"[{match[0]}]")
                elif match[1]:  # Author-year citation
                    citations.append(f"({match[1]})")

            return list(set(citations))[:10]
        except Exception as e:
            logger.error(f"Citation extraction failed: {e}")
            return []

    @staticmethod
    def extract_facts(text, max_facts=7):
        try:
            sentences = sent_tokenize(text)
            facts = []

            fact_indicators = [
                "according to",
                "shows that",
                "research",
                "study",
                "found that",
                "reveals",
                "reported",
                "discovered",
                "demonstrates",
                "proves",
                "confirms",
                "indicates",
                "suggests",
                "evidence",
                "data",
                "statistics",
                "percent",
                "%"
            ]

            for sentence in sentences:
                if any(indicator in sentence.lower() for indicator in fact_indicators):
                    facts.append(sentence)

            if len(facts) < max_facts:
                number_pattern = re.compile(r'\d+\.\d+|\d+')
                for sentence in sentences:
                    if number_pattern.search(sentence) and sentence not in facts:
                        facts.append(sentence)
                        if len(facts) >= max_facts:
                            break

            return facts[:max_facts]
        except Exception as e:
            logger.error(f"Fact extraction failed: {e}")
            return []

    @staticmethod
    def get_related_articles(title, keywords, count=5):
        try:
            related = []
            title_words = title.split()
            if len(title_words) > 3:
                new_title = " ".join(title_words[:3]) + ": What You Need to Know"
                related.append(new_title)

            for i, keyword in enumerate(keywords[:4]):
                if i == 0:
                    related.append(f"The Complete Guide to {keyword.title()}")
                elif i == 1:
                    related.append(f"How {keyword.title()} Is Changing the Industry")
                elif i == 2:
                    related.append(f"Top 10 {keyword.title()} Trends in 2025")
                else:
                    related.append(f"Understanding {keyword.title()}: A Deep Dive")

            return related
        except Exception as e:
            logger.error(f"Related articles generation failed: {e}")
            return []

    @staticmethod
    def translate_text(text, target_language):
        try:
            detected_language = translator.detect(text).lang
            if detected_language == target_language:
                return text, detected_language

            translation = translator.translate(text, dest=target_language)
            return translation.text, detected_language
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return text, "unknown"

    @staticmethod
    def format_summary(summary, format_style):
        try:
            if format_style == "bullet":
                sentences = sent_tokenize(summary)
                formatted = "# Summary\n\n"
                for sentence in sentences:
                    formatted += f"• {sentence}\n"
                return formatted

            elif format_style == "academic":
                sentences = sent_tokenize(summary)
                third = len(sentences) // 3

                formatted = "# Academic Summary\n\n"
                formatted += "## Introduction\n\n"
                formatted += " ".join(sentences[:third]) + "\n\n"

                formatted += "## Analysis\n\n"
                formatted += " ".join(sentences[third:2 * third]) + "\n\n"

                formatted += "## Conclusion\n\n"
                formatted += " ".join(sentences[2 * third:])
                return formatted

            elif format_style == "simplified":
                simplified = summary
                word_replacements = {
                    "utilize": "use",
                    "endeavor": "try",
                    "commence": "begin",
                    "terminate": "end",
                    "facilitate": "help",
                    "accordingly": "so",
                    "subsequently": "later",
                    "additionally": "also",
                    "fundamental": "basic",
                    "demonstrate": "show",
                    "comprehend": "understand"
                }

                for complex_word, simple_word in word_replacements.items():
                    simplified = re.sub(r'\b' + complex_word + r'\b', simple_word, simplified, flags=re.IGNORECASE)

                return "# Simple Summary\n\n" + simplified

            else:  # standard format
                return "# Article Summary\n\n" + summary

        except Exception as e:
            logger.error(f"Summary formatting failed: {e}")
            return summary

    def cleanup(self):
        """Clean up temporary files"""
        try:
            for filename in os.listdir(TEMP_DIR):
                file_path = os.path.join(TEMP_DIR, filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    logger.error(f"Failed to delete {file_path}. Reason: {e}")
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
