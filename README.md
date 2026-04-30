# Smart Fashion Advisor

An AI-powered personal stylist application that combines skin tone detection, outfit recommendations, and an intelligent fashion chatbot.

## Features

### 1. User Profile
- Enter your name and select gender for personalized recommendations

### 2. Photo Input (Two Options)
- **Camera Capture**: Use your webcam to take a photo
- **Photo Upload**: Upload an existing photo from your device

### 3. Skin Tone Detection
- Uses MediaPipe Face Mesh for accurate face detection
- Analyzes skin from cheeks and forehead regions
- Classifies skin tone: Fair, Light, Medium, Olive, Brown, or Dark
- Lighting normalization for accurate results

### 4. Outfit Recommendations
- Personalized color palette based on your skin tone
- Filter by occasion (Casual, Formal, Party, Sports, Ethnic, Travel)
- Filter by season (Summer, Winter, Spring, Fall)
- Categories: Topwear, Bottomwear, Footwear, Dresses/Traditional

### 5. AI Fashion Chatbot
- Ask about colors that suit you
- Get outfit suggestions for any occasion
- Learn how to pair different pieces
- Seasonal fashion advice
- Powered by Google Gemini AI (with smart fallback)

## Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd SmartFashionAdvisor
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **(Optional) Set up Gemini AI for enhanced chatbot:**
   ```bash
   # Get your API key from https://makersuite.google.com/app/apikey
   
   # Windows
   set GEMINI_API_KEY=your_api_key_here
   
   # macOS/Linux
   export GEMINI_API_KEY=your_api_key_here
   ```
   Note: The chatbot works without the API key using intelligent fallback responses.

## Running the Application

1. **Start the server:**
   ```bash
   python app.py
   ```

2. **Open your browser and navigate to:**
   ```
   http://localhost:5000
   ```

## Deploy on Railway

1. Push this project to GitHub.
2. In Railway, create a new project and choose **Deploy from GitHub repo**.
3. Add environment variable in Railway:
   - `GEMINI_API_KEY=your_key` (optional for chatbot enhancement)
4. Railway will auto-detect the `Procfile` and run:
   - `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 180`
5. After deploy, open the generated Railway domain URL.

Notes:
- `data/styles.csv` and `data/images.csv` must be present in the deployed repo.
- First boot can be slower because MediaPipe and dataset loading happen at startup.

## Usage Guide

1. **Step 1 - Profile**: Enter your name and select your gender
2. **Step 2 - Photo**: Either capture a photo using your camera or upload an existing one
3. **Step 3 - Analysis**: View your detected skin tone and recommended color palette
4. **Step 4 - Recommendations**: Browse outfit suggestions filtered by occasion and season
5. **Chatbot**: Click the chat icon (bottom-right) to ask fashion questions anytime!

## Project Structure

```
SmartFashionAdvisor/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── templates/
│   └── index.html        # Frontend UI
└── static/
    ├── css/              # Stylesheets (inline in index.html)
    └── js/               # JavaScript (inline in index.html)
```

## Data Source

The outfit recommendations use the fashion dataset from `../FashionAdvisor_kaggle/styles.csv` which contains:
- 44,000+ fashion items
- Categories: Apparel, Footwear
- Attributes: Gender, Color, Season, Usage/Occasion

## Technologies Used

- **Backend**: Flask (Python)
- **Face Detection**: MediaPipe Face Mesh
- **Image Processing**: OpenCV
- **Data Processing**: Pandas, NumPy
- **AI Chatbot**: Google Gemini API (optional)
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)

## Tips for Best Results

1. **Lighting**: Use good, even lighting when taking photos
2. **Face Position**: Center your face in the camera frame
3. **Clear View**: Ensure your face is fully visible without obstructions
4. **Photo Quality**: Use clear, well-lit photos for upload

## Chatbot Sample Questions

- "What colors suit me best?"
- "What should I wear to a wedding?"
- "Suggest a party outfit"
- "What to wear for an interview?"
- "How do I pair my navy blue shirt?"
- "Summer outfit ideas"
- "Help me with date night outfit"

## Troubleshooting

- **Camera not working**: Allow camera permissions in your browser
- **No face detected**: Ensure good lighting and face visibility
- **Slow loading**: The first load may take time as MediaPipe initializes

Enjoy your personalized fashion experience!
