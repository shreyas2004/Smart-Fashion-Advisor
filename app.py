"""
Smart Fashion Advisor - Combined Application
Integrates skin tone detection, outfit recommendations, and AI chatbot
"""

import io
import os
import base64
import numpy as np
import cv2
import pandas as pd
from flask import Flask, render_template, request, jsonify, session
import mediapipe as mp
from datetime import datetime
import random

app = Flask(__name__)
app.secret_key = 'smart_fashion_advisor_2024'

# Try to import Gemini AI (optional)
try:
    import google.generativeai as genai
    GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        GEMINI_AVAILABLE = True
    else:
        GEMINI_AVAILABLE = False
except ImportError:
    GEMINI_AVAILABLE = False
    GEMINI_API_KEY = ''

# Load fashion dataset
DATASET_PATH = os.path.join(os.path.dirname(__file__), 'data', 'styles.csv')
IMAGES_PATH = os.path.join(os.path.dirname(__file__), 'data', 'images.csv')

# Load image URLs mapping
image_urls = {}
try:
    images_df = pd.read_csv(IMAGES_PATH)
    for _, row in images_df.iterrows():
        filename = str(row['filename']).replace('.jpg', '')
        # Convert http to https for better browser compatibility
        link = str(row['link'])
        if link.startswith('http://'):
            link = 'https://' + link[7:]
        image_urls[filename] = link
    print(f"Loaded {len(image_urls)} image URLs")
except Exception as e:
    print(f"Error loading images: {e}")

try:
    fashion_data = pd.read_csv(DATASET_PATH, on_bad_lines='skip')
    columns_to_keep = ['id', 'gender', 'masterCategory', 'subCategory', 'articleType', 'baseColour', 'season', 'usage', 'productDisplayName']
    fashion_data = fashion_data[[col for col in columns_to_keep if col in fashion_data.columns]]
    fashion_data.columns = [col.strip().lower() for col in fashion_data.columns]
    fashion_data = fashion_data[fashion_data['gender'].isin(['Men', 'Women'])]
    # Include more categories: Apparel, Footwear, and Accessories
    fashion_data = fashion_data[fashion_data['mastercategory'].isin(['Apparel', 'Footwear', 'Accessories'])]
    # Exclude inappropriate categories
    fashion_data = fashion_data[~fashion_data['subcategory'].isin(['Innerwear', 'Loungewear and Nightwear', 'Socks'])]
    fashion_data.dropna(subset=['gender', 'mastercategory', 'basecolour'], inplace=True)
    print(f"Loaded {len(fashion_data)} fashion items")
    # Print available subcategories for debugging
    print(f"Available subcategories: {fashion_data['subcategory'].unique().tolist()[:20]}...")
except Exception as e:
    print(f"Error loading dataset: {e}")
    fashion_data = pd.DataFrame()

# Enhanced Color Theory Based on Skin Tone Analysis
# Using seasonal color analysis and complementary color theory
SKIN_TONE_COLORS = {
    'Fair': {
        # Cool undertone - Winter/Summer palette
        # High contrast colors work best, jewel tones create elegance
        'best': ['Navy Blue', 'Burgundy', 'Emerald', 'Plum', 'Charcoal', 'Teal', 'Rose', 'Ruby Red', 'Cobalt Blue', 'Forest Green'],
        'good': ['Blue', 'Red', 'Green', 'Purple', 'Grey', 'Pink', 'Maroon', 'Black', 'White', 'Silver'],
        'avoid': ['Orange', 'Yellow', 'Mustard', 'Gold', 'Camel', 'Peach'],
        'theory': 'Fair skin has cool undertones. Jewel tones and high-contrast colors like navy, burgundy, and emerald create stunning contrast. Cool colors complement your natural pink undertones.',
        'undertone': 'Cool',
        'season': 'Winter'
    },
    'Light': {
        # Light warm undertone - Spring palette
        # Soft, warm colors enhance the gentle complexion
        'best': ['Coral', 'Peach', 'Aqua', 'Soft Pink', 'Light Blue', 'Lavender', 'Mint', 'Warm Red', 'Turquoise Blue', 'Cream'],
        'good': ['Pink', 'Blue', 'Green', 'Purple', 'Turquoise', 'Cream', 'Ivory', 'Camel', 'Light Grey'],
        'avoid': ['Black', 'Dark Brown', 'Neon', 'Harsh Yellow', 'Orange', 'Olive'],
        'theory': 'Light skin with warm undertones glows in soft, delicate colors. Spring palette colors like coral, peach, and aqua enhance your warm glow without overwhelming your complexion.',
        'undertone': 'Warm',
        'season': 'Spring'
    },
    'Medium': {
        # Neutral to warm undertone - Autumn palette
        # Earth tones and warm colors create harmony
        'best': ['Olive', 'Rust', 'Mustard', 'Teal', 'Terracotta', 'Burnt Orange', 'Khaki', 'Bronze', 'Copper', 'Chocolate Brown'],
        'good': ['Orange', 'Yellow', 'Green', 'Brown', 'Gold', 'Copper', 'Beige', 'Camel', 'Warm Red', 'Forest Green'],
        'avoid': ['Pale Pink', 'Pale Yellow', 'Pastel Blue', 'Silver', 'Black', 'Pure White'],
        'theory': 'Medium skin tones have golden undertones that harmonize beautifully with earth tones. Autumn palette colors like rust, olive, and mustard bring out your natural warmth.',
        'undertone': 'Warm/Neutral',
        'season': 'Autumn'
    },
    'Olive': {
        # Warm with green undertone - Deep Autumn palette
        # Rich warm colors neutralize the green undertone
        'best': ['Orange', 'Gold', 'Amber', 'Rust', 'Coral', 'Turquoise', 'Purple', 'Magenta', 'Tomato Red', 'Bright Yellow'],
        'good': ['Red', 'Green', 'Brown', 'Cream', 'Bronze', 'Copper', 'Teal', 'Fuchsia', 'Peach'],
        'avoid': ['Grey', 'Pale Blue', 'Pastel Pink', 'Silver', 'Muted Colors'],
        'theory': 'Olive skin has yellow-green undertones. Warm, saturated colors like orange, coral, and gold neutralize the green and make your skin glow. Purple creates beautiful complementary contrast.',
        'undertone': 'Warm/Green',
        'season': 'Deep Autumn'
    },
    'Brown': {
        # Warm rich undertone - Warm palette
        # Bright warm colors and earth tones create radiance
        'best': ['White', 'Cream', 'Gold', 'Orange', 'Coral', 'Hot Pink', 'Yellow', 'Royal Blue', 'Emerald', 'Red'],
        'good': ['Red', 'Purple', 'Green', 'Teal', 'Turquoise', 'Fuchsia', 'Mustard', 'Copper', 'Bronze'],
        'avoid': ['Brown', 'Tan', 'Beige', 'Muted Pastels', 'Olive', 'Khaki'],
        'theory': 'Brown skin radiates warmth and richness. Bright, bold colors like white, gold, and royal blue create beautiful contrast. Avoid colors too close to your skin tone as they can appear washed out.',
        'undertone': 'Warm',
        'season': 'Warm Autumn'
    },
    'Dark': {
        # Deep cool/warm undertone - Deep Winter palette
        # High contrast bright colors create striking looks
        'best': ['White', 'Bright Yellow', 'Orange', 'Fuchsia', 'Cobalt Blue', 'Emerald', 'Ruby Red', 'Hot Pink', 'Electric Blue', 'Lime Green'],
        'good': ['Red', 'Pink', 'Purple', 'Gold', 'Silver', 'Cream', 'Turquoise', 'Coral', 'Bright Green'],
        'avoid': ['Dark Brown', 'Navy', 'Dark Grey', 'Black', 'Muted Earth Tones', 'Pastels'],
        'theory': 'Deep skin tones look stunning in bold, vibrant colors. High contrast colors like white, bright yellow, and fuchsia create dramatic, beautiful effects. Your skin can carry intense colors that might overwhelm lighter complexions.',
        'undertone': 'Neutral/Deep',
        'season': 'Deep Winter'
    }
}

# Color wheel complementary pairings for outfit coordination
COLOR_PAIRINGS = {
    'Navy Blue': ['White', 'Cream', 'Coral', 'Gold', 'Burgundy', 'Tan'],
    'White': ['Navy Blue', 'Black', 'Red', 'Blue', 'Any Color'],
    'Black': ['White', 'Red', 'Pink', 'Gold', 'Silver', 'Cream'],
    'Red': ['White', 'Black', 'Navy Blue', 'Cream', 'Grey', 'Denim Blue'],
    'Blue': ['White', 'Cream', 'Brown', 'Orange', 'Coral', 'Grey'],
    'Green': ['White', 'Cream', 'Brown', 'Navy Blue', 'Pink', 'Gold'],
    'Pink': ['Grey', 'Navy Blue', 'White', 'Black', 'Cream', 'Blue'],
    'Grey': ['Pink', 'Blue', 'Purple', 'Yellow', 'Red', 'Black'],
    'Brown': ['White', 'Cream', 'Blue', 'Green', 'Orange', 'Rust'],
    'Burgundy': ['White', 'Cream', 'Grey', 'Navy Blue', 'Pink', 'Camel'],
    'Purple': ['Grey', 'White', 'Silver', 'Pink', 'Navy Blue', 'Yellow'],
    'Orange': ['Navy Blue', 'White', 'Blue', 'Cream', 'Brown', 'Teal'],
    'Yellow': ['Navy Blue', 'Grey', 'White', 'Black', 'Brown', 'Purple'],
    'Teal': ['White', 'Cream', 'Coral', 'Brown', 'Navy Blue', 'Gold'],
    'Coral': ['Navy Blue', 'White', 'Teal', 'Grey', 'Cream', 'Gold']
}

# Occasion-based style mapping
OCCASION_STYLES = {
    'casual': ['Casual', 'Smart Casual'],
    'formal': ['Formal'],
    'party': ['Party', 'Smart Casual'],
    'sports': ['Sports'],
    'ethnic': ['Ethnic'],
    'travel': ['Travel', 'Casual'],
    'work': ['Formal', 'Smart Casual']
}


class HeadlessSkinToneDetector:
    """Skin tone detector that works with uploaded images"""
    
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6)
        
        self.skin_tone_categories = {
            'Fair': {'h_min': 0, 'h_max': 20, 's_min': 20, 's_max': 60, 'v_min': 80, 'v_max': 200},
            'Light': {'h_min': 0, 'h_max': 25, 's_min': 30, 's_max': 80, 'v_min': 70, 'v_max': 180},
            'Medium': {'h_min': 0, 'h_max': 25, 's_min': 40, 's_max': 100, 'v_min': 60, 'v_max': 160},
            'Olive': {'h_min': 15, 'h_max': 35, 's_min': 50, 's_max': 120, 'v_min': 50, 'v_max': 140},
            'Brown': {'h_min': 0, 'h_max': 25, 's_min': 60, 's_max': 140, 'v_min': 40, 'v_max': 120},
            'Dark': {'h_min': 0, 'h_max': 25, 's_min': 80, 's_max': 180, 'v_min': 20, 'v_max': 80}
        }

    def normalize_lighting(self, frame):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab_normalized = cv2.merge([l, a, b])
        return cv2.cvtColor(lab_normalized, cv2.COLOR_LAB2BGR)

    def check_lighting_quality(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        std_brightness = np.std(gray)
        return 30 <= mean_brightness <= 240 and std_brightness > 10

    def get_skin_regions(self, landmarks, frame_shape):
        h, w = frame_shape[:2]
        points = np.array([(int(lm.x * w), int(lm.y * h)) for lm in landmarks.landmark])
        
        left_cheek_indices = [116, 117, 118, 119, 120, 121, 126, 142, 36, 37, 38, 39, 40, 41]
        right_cheek_indices = [345, 346, 347, 348, 349, 350, 355, 371, 266, 267, 268, 269, 270, 271]
        forehead_indices = [10, 151, 9, 8, 107, 55, 65, 52, 53, 46]
        
        return {
            'left_cheek': points[left_cheek_indices],
            'right_cheek': points[right_cheek_indices],
            'forehead': points[forehead_indices]
        }

    def extract_skin_color(self, frame, skin_regions):
        normalized_frame = self.normalize_lighting(frame)
        hsv_frame = cv2.cvtColor(normalized_frame, cv2.COLOR_BGR2HSV)
        
        all_skin_pixels = []
        all_skin_pixels_hsv = []
        
        for region_name, points in skin_regions.items():
            x, y, w, h = cv2.boundingRect(points.astype(np.int32))
            padding = 15
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(frame.shape[1] - x, w + 2 * padding)
            h = min(frame.shape[0] - y, h + 2 * padding)
            
            region_bgr = normalized_frame[y:y+h, x:x+w]
            region_hsv = hsv_frame[y:y+h, x:x+w]
            
            lower_skin = np.array([0, 30, 40], dtype=np.uint8)
            upper_skin = np.array([25, 150, 220], dtype=np.uint8)
            
            skin_mask = cv2.inRange(region_hsv, lower_skin, upper_skin)
            kernel = np.ones((3, 3), np.uint8)
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
            
            skin_pixels_bgr = region_bgr[skin_mask > 0]
            skin_pixels_hsv = region_hsv[skin_mask > 0]
            
            if len(skin_pixels_bgr) > 0:
                all_skin_pixels.extend(skin_pixels_bgr)
                all_skin_pixels_hsv.extend(skin_pixels_hsv)
        
        if len(all_skin_pixels) > 0:
            avg_bgr = np.mean(all_skin_pixels, axis=0)
            avg_rgb = avg_bgr[::-1]
            avg_hsv = np.mean(all_skin_pixels_hsv, axis=0)
            return avg_rgb, avg_hsv
        return None, None

    def classify_skin_tone(self, hsv_values):
        if hsv_values is None:
            return "Medium"
        
        h, s, v = hsv_values
        category_scores = {}
        
        for category, thresholds in self.skin_tone_categories.items():
            h_score = 1.0 if thresholds['h_min'] <= h <= thresholds['h_max'] else 0.0
            s_score = 1.0 if thresholds['s_min'] <= s <= thresholds['s_max'] else 0.0
            v_score = 1.0 if thresholds['v_min'] <= v <= thresholds['v_max'] else 0.0
            total_score = (h_score * 0.4) + (s_score * 0.4) + (v_score * 0.2)
            category_scores[category] = total_score
        
        best_category = max(category_scores, key=category_scores.get)
        best_score = category_scores[best_category]
        
        if best_score < 0.6:
            if h <= 15 and s <= 50:
                return "Fair"
            elif h <= 20 and s <= 70:
                return "Light"
            elif h <= 25 and s <= 100:
                return "Medium"
            elif 15 <= h <= 35 and s >= 50:
                return "Olive"
            elif h <= 25 and s >= 80:
                return "Brown"
            else:
                return "Dark"
        return best_category

    def process_image(self, bgr_image):
        lighting_ok = self.check_lighting_quality(bgr_image)
        if not lighting_ok:
            return {"success": False, "message": "Lighting conditions are not optimal. Please try with better lighting."}
        
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)
        
        if not results.multi_face_landmarks:
            return {"success": False, "message": "No face detected. Please ensure your face is clearly visible."}
        
        landmarks = results.multi_face_landmarks[0]
        skin_regions = self.get_skin_regions(landmarks, bgr_image.shape)
        rgb_vals, hsv_vals = self.extract_skin_color(bgr_image, skin_regions)
        
        if rgb_vals is None:
            return {"success": False, "message": "Could not extract skin tone. Please try with a clearer photo."}
        
        skin_tone = self.classify_skin_tone(hsv_vals)
        rgb_list = [int(x) for x in np.round(rgb_vals).tolist()]
        
        return {
            "success": True,
            "skin_tone": skin_tone,
            "rgb": rgb_list,
            "color_hex": "#{:02x}{:02x}{:02x}".format(*rgb_list)
        }


# Initialize detector
detector = HeadlessSkinToneDetector()


def get_recommended_outfits(skin_tone, gender, occasion=None, season=None, limit=20):
    """Get outfit recommendations based on skin tone and preferences"""
    if fashion_data.empty:
        return {'topwear': [], 'bottomwear': [], 'footwear': [], 'dress': []}
    
    recommended_colors = SKIN_TONE_COLORS.get(skin_tone, SKIN_TONE_COLORS['Medium'])
    
    # Filter by gender
    gender_map = {'male': 'Men', 'female': 'Women'}
    target_gender = gender_map.get(gender.lower(), 'Men')
    filtered_data = fashion_data[fashion_data['gender'] == target_gender]
    
    # Filter by occasion/usage
    if occasion and occasion in OCCASION_STYLES:
        target_usages = OCCASION_STYLES[occasion]
        filtered_data = filtered_data[filtered_data['usage'].isin(target_usages)]
    
    # Filter by season
    if season:
        filtered_data = filtered_data[filtered_data['season'].str.lower() == season.lower()]
    
    # Score items by color match
    def color_score(color):
        if pd.isna(color):
            return 0
        color = str(color).lower()
        for rec in recommended_colors['best']:
            if rec.lower() in color or color in rec.lower():
                return 3
        for rec in recommended_colors['good']:
            if rec.lower() in color or color in rec.lower():
                return 2
        for avoid in recommended_colors['avoid']:
            if avoid.lower() in color or color in avoid.lower():
                return 0
        return 1
    
    filtered_data = filtered_data.copy()
    filtered_data['color_score'] = filtered_data['basecolour'].apply(color_score)
    filtered_data = filtered_data[filtered_data['color_score'] > 0]
    filtered_data = filtered_data.sort_values('color_score', ascending=False)
    
    # Expanded categories for diverse recommendations
    results = {
        'topwear': [],
        'bottomwear': [],
        'footwear': [],
        'dress': [],
        'traditional': [],
        'outerwear': [],
        'accessories': []
    }
    
    # Category keywords mapping
    category_keywords = {
        'topwear': ['top', 'shirt', 'tshirt', 't-shirt', 'blouse', 'polo', 'sweater', 'sweatshirt', 'tank', 'tunic', 'kurta', 'kurti'],
        'bottomwear': ['bottom', 'jeans', 'trouser', 'pant', 'short', 'skirt', 'legging', 'capri', 'palazzos', 'churidar'],
        'footwear': ['shoe', 'flip', 'sandal', 'sneaker', 'boot', 'heel', 'loafer', 'slipper', 'mojari', 'kolhapuri'],
        'dress': ['dress', 'gown', 'jumpsuit', 'romper', 'maxi'],
        'traditional': ['saree', 'sari', 'lehenga', 'kurta', 'sherwani', 'salwar', 'dupatta', 'ethnic', 'nehru jacket', 'bandhgala', 'anarkali', 'palazzo'],
        'outerwear': ['jacket', 'blazer', 'coat', 'cardigan', 'hoodie', 'shrug', 'waistcoat', 'vest'],
        'accessories': ['watch', 'belt', 'bag', 'wallet', 'scarf', 'tie', 'cap', 'hat', 'sunglasses', 'jewellery']
    }
    
    for _, row in filtered_data.iterrows():
        subcategory = str(row.get('subcategory', '')).lower()
        articletype = str(row.get('articletype', '')).lower()
        item_id = int(row['id']) if pd.notna(row['id']) else 0
        
        # Get image URL from the images mapping
        img_url = image_urls.get(str(item_id), '')
        
        # Generate Google Image search URL as fallback if no image
        product_name = str(row.get('productdisplayname', row.get('articletype', 'Fashion Item')))
        color = str(row.get('basecolour', 'Unknown'))
        
        if not img_url or img_url == 'nan':
            # Create Google Image search fallback URL
            search_query = f"{product_name} {color}".replace(' ', '+')
            img_url = f"https://source.unsplash.com/400x500/?{search_query.replace('+', ',')},fashion,clothing"
        
        color_match = 'Best Match' if row.get('color_score', 0) == 3 else 'Good Match' if row.get('color_score', 0) == 2 else 'Compatible'
        
        item = {
            'id': item_id,
            'name': product_name,
            'type': str(row.get('articletype', 'Unknown')),
            'color': color,
            'category': subcategory,
            'usage': str(row.get('usage', 'Casual')),
            'season': str(row.get('season', 'All Season')),
            'score': int(row.get('color_score', 1)),
            'image': img_url,
            'color_match': color_match
        }
        
        # Categorize based on keywords in subcategory and articletype
        combined_text = f"{subcategory} {articletype}"
        categorized = False
        
        # Check traditional/ethnic first (priority)
        for keyword in category_keywords['traditional']:
            if keyword in combined_text and len(results['traditional']) < limit:
                results['traditional'].append(item)
                categorized = True
                break
        
        if not categorized:
            for keyword in category_keywords['outerwear']:
                if keyword in combined_text and len(results['outerwear']) < limit:
                    results['outerwear'].append(item)
                    categorized = True
                    break
        
        if not categorized:
            for keyword in category_keywords['topwear']:
                if keyword in combined_text and len(results['topwear']) < limit:
                    results['topwear'].append(item)
                    categorized = True
                    break
        
        if not categorized:
            for keyword in category_keywords['bottomwear']:
                if keyword in combined_text and len(results['bottomwear']) < limit:
                    results['bottomwear'].append(item)
                    categorized = True
                    break
        
        if not categorized:
            for keyword in category_keywords['footwear']:
                if keyword in combined_text and len(results['footwear']) < limit:
                    results['footwear'].append(item)
                    categorized = True
                    break
        
        if not categorized:
            for keyword in category_keywords['dress']:
                if keyword in combined_text and len(results['dress']) < limit:
                    results['dress'].append(item)
                    categorized = True
                    break
        
        if not categorized:
            for keyword in category_keywords['accessories']:
                if keyword in combined_text and len(results['accessories']) < limit:
                    results['accessories'].append(item)
                    categorized = True
                    break
    
    return results


def get_ai_response(user_message, context):
    """Get AI chatbot response using Gemini or fallback"""
    if GEMINI_AVAILABLE and GEMINI_API_KEY:
        try:
            model = genai.GenerativeModel('gemini-pro')
            
            system_prompt = f"""You are a friendly and knowledgeable fashion advisor chatbot. 
            The user's details:
            - Name: {context.get('name', 'Friend')}
            - Gender: {context.get('gender', 'Not specified')}
            - Skin Tone: {context.get('skin_tone', 'Not analyzed yet')}
            
            Best colors for their skin tone: {', '.join(SKIN_TONE_COLORS.get(context.get('skin_tone', 'Medium'), SKIN_TONE_COLORS['Medium'])['best'])}
            
            Provide helpful, personalized fashion advice. Be conversational, friendly, and specific.
            If they ask about outfits for an occasion, suggest specific combinations.
            If they ask what colors suit them, explain based on their skin tone.
            Keep responses concise but informative (2-3 paragraphs max)."""
            
            full_prompt = f"{system_prompt}\n\nUser: {user_message}"
            response = model.generate_content(full_prompt)
            return response.text
        except Exception as e:
            print(f"Gemini API error: {e}")
    
    return get_fallback_response(user_message, context)


def get_fallback_response(user_message, context):
    """Provide intelligent fallback responses when API is unavailable"""
    user_message = user_message.lower()
    skin_tone = context.get('skin_tone', 'Medium')
    name = context.get('name', 'there')
    gender = context.get('gender', 'male')
    
    colors = SKIN_TONE_COLORS.get(skin_tone, SKIN_TONE_COLORS['Medium'])
    
    # Occasion-based responses
    occasions = {
        'wedding': f"For a wedding, I'd recommend elegant pieces in {colors['best'][0]} or {colors['best'][1]}. "
                   f"A well-fitted {'suit or sherwani' if gender == 'male' else 'saree or lehenga'} in these colors would complement your {skin_tone.lower()} skin beautifully!",
        'party': f"Party time, {name}! Go bold with {colors['best'][2]} or {colors['best'][0]}. "
                 f"{'A crisp shirt' if gender == 'male' else 'A stylish dress'} in these shades will make you stand out!",
        'interview': f"For interviews, stick to professional colors like {colors['good'][0]} or {colors['good'][1]}. "
                     f"A {'well-tailored blazer' if gender == 'male' else 'structured blouse'} paired with neutral bottoms works great!",
        'casual': f"For casual outings, you can't go wrong with {colors['best'][0]} or {colors['good'][0]}. "
                  f"{'A comfortable polo or t-shirt' if gender == 'male' else 'A breezy top or casual dress'} would be perfect!",
        'date': f"Date night? {colors['best'][1]} or {colors['best'][2]} would look amazing on your {skin_tone.lower()} skin! "
                f"{'A nice button-down shirt' if gender == 'male' else 'An elegant blouse or dress'} will surely impress.",
        'office': f"For the office, I suggest {colors['good'][0]} or {colors['good'][1]}. "
                  f"Professional yet stylish - perfect for your {skin_tone.lower()} complexion!",
        'gym': f"For the gym, comfort is key! Go with breathable fabrics in {colors['good'][0]} or Black. "
               f"Sports wear in these colors will keep you looking fresh while working out!",
        'beach': f"Beach vibes! Light colors like {colors['best'][0]} or White would be perfect. "
                 f"Pair with comfortable shorts and sandals for that perfect beach look!"
    }
    
    for occasion, response in occasions.items():
        if occasion in user_message:
            return response
    
    # Color-related queries
    if any(word in user_message for word in ['color', 'colour', 'shade', 'suit me', 'look good']):
        return (f"Based on your {skin_tone.lower()} skin tone, {name}, these colors will look amazing on you: "
                f"{', '.join(colors['best'][:4])}. Try to avoid {', '.join(colors['avoid'][:2])} as they might wash you out.")
    
    # Pairing queries
    if any(word in user_message for word in ['pair', 'match', 'combine', 'go with', 'wear with']):
        return (f"Great question! For your {skin_tone.lower()} skin, I'd suggest pairing {colors['best'][0]} tops with "
                f"neutral bottoms like black, grey, or khaki. For a bolder look, try {colors['best'][1]} with "
                f"{colors['good'][0]} - this combination will really make your complexion glow!")
    
    # Season queries
    if 'summer' in user_message:
        return f"For summer, light and breathable colors work best! Try {colors['best'][0]} or {colors['good'][0]} in cotton or linen fabrics. Stay cool and stylish!"
    if 'winter' in user_message:
        return f"Winter calls for deeper, richer shades! {colors['best'][1]} and {colors['best'][2]} in warm fabrics like wool or cashmere would be perfect for you."
    if 'spring' in user_message:
        return f"Spring is perfect for fresh colors! Try {colors['best'][0]} or pastel versions of {colors['good'][0]}. Light layers work great for this season!"
    if 'fall' in user_message or 'autumn' in user_message:
        return f"Fall fashion is all about warm, earthy tones! {colors['best'][1]} and {colors['good'][0]} would complement the season and your {skin_tone.lower()} skin perfectly."
    
    # Greeting responses
    if any(word in user_message for word in ['hi', 'hello', 'hey', 'good morning', 'good evening']):
        return (f"Hello {name}! I'm your personal fashion advisor. With your {skin_tone.lower()} skin tone, "
                f"you have so many great color options! Feel free to ask me about outfit suggestions for any occasion, "
                f"color combinations, or fashion tips. How can I help you today?")
    
    # Help/capabilities
    if any(word in user_message for word in ['help', 'what can you do', 'capabilities']):
        return (f"I'm here to help you look your best, {name}! Here's what I can do:\n"
                f"- Suggest colors that complement your {skin_tone.lower()} skin tone\n"
                f"- Recommend outfits for different occasions (wedding, party, office, etc.)\n"
                f"- Help you pair clothes together\n"
                f"- Give season-specific fashion advice\n"
                f"Just ask away!")
    
    # Default helpful response
    return (f"Hi {name}! With your {skin_tone.lower()} skin tone, you have great options! "
            f"Your best colors are {', '.join(colors['best'][:3])}. "
            f"Feel free to ask me about specific occasions, color combinations, or outfit pairings! "
            f"For example, try asking 'What should I wear to a wedding?' or 'Which colors suit me best?'")


# Flask Routes
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    """Analyze uploaded image for skin tone"""
    if 'image' not in request.files and 'image_data' not in request.form:
        return jsonify({'success': False, 'message': 'No image provided'}), 400
    
    try:
        if 'image' in request.files:
            file = request.files['image']
            in_memory = file.read()
            arr = np.frombuffer(in_memory, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        else:
            # Handle base64 image from camera capture
            image_data = request.form['image_data']
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            img_bytes = base64.b64decode(image_data)
            arr = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'success': False, 'message': 'Could not decode image'}), 400
        
        result = detector.process_image(img)
        
        if result['success']:
            session['skin_tone'] = result['skin_tone']
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error processing image: {str(e)}'}), 500


def get_outfit_combinations(skin_tone):
    """Generate outfit color combinations based on skin tone and color theory"""
    color_palette = SKIN_TONE_COLORS.get(skin_tone, SKIN_TONE_COLORS['Medium'])
    best_colors = color_palette['best']
    good_colors = color_palette['good']
    
    combinations = []
    
    # Classic combinations based on color theory
    classic_combos = [
        {'top': 'White', 'bottom': 'Navy Blue', 'style': 'Classic Professional', 'occasion': 'Office / Formal'},
        {'top': 'Black', 'bottom': 'Grey', 'style': 'Modern Minimal', 'occasion': 'Any Occasion'},
        {'top': 'Navy Blue', 'bottom': 'Khaki', 'style': 'Smart Casual', 'occasion': 'Weekend / Casual'},
        {'top': 'White', 'bottom': 'Black', 'style': 'Timeless Contrast', 'occasion': 'Any Occasion'},
        {'top': 'Grey', 'bottom': 'Blue', 'style': 'Cool Tones', 'occasion': 'Office / Casual'},
    ]
    
    # Add personalized combinations based on user's best colors
    for i, top_color in enumerate(best_colors[:4]):
        # Find matching bottom colors from COLOR_PAIRINGS
        matching_bottoms = COLOR_PAIRINGS.get(top_color, ['Black', 'Grey', 'Navy Blue', 'White'])
        
        # Filter to include neutrals and complementary colors
        for bottom in matching_bottoms[:2]:
            combo = {
                'top': top_color,
                'bottom': bottom,
                'style': 'Personalized Match',
                'occasion': 'Recommended for You',
                'reason': f'{top_color} complements your {skin_tone.lower()} skin tone'
            }
            combinations.append(combo)
    
    # Add some classic combos
    for combo in classic_combos[:3]:
        combo['reason'] = 'Classic color theory combination'
        combinations.append(combo)
    
    # Create contrast and monochrome suggestions
    if 'Navy Blue' in best_colors or 'Blue' in best_colors:
        combinations.append({
            'top': 'Light Blue',
            'bottom': 'Navy Blue',
            'style': 'Tonal Blue',
            'occasion': 'Office / Casual',
            'reason': 'Monochromatic blue creates a sophisticated look'
        })
    
    if 'Burgundy' in best_colors or 'Maroon' in best_colors:
        combinations.append({
            'top': 'Burgundy',
            'bottom': 'Black',
            'style': 'Bold Elegance',
            'occasion': 'Party / Evening',
            'reason': 'Deep burgundy pairs beautifully with black for evening wear'
        })
    
    # Ethnic/Traditional combinations
    if skin_tone in ['Medium', 'Olive', 'Brown', 'Dark']:
        combinations.append({
            'top': 'Gold',
            'bottom': 'Maroon',
            'style': 'Traditional Elegance',
            'occasion': 'Festive / Wedding',
            'reason': 'Rich jewel tones enhance warm skin tones'
        })
        combinations.append({
            'top': 'Royal Blue',
            'bottom': 'Gold',
            'style': 'Regal Traditional',
            'occasion': 'Festive / Wedding',
            'reason': 'Royal blue with gold accents is a timeless festive combination'
        })
    
    if skin_tone in ['Fair', 'Light']:
        combinations.append({
            'top': 'Pastel Pink',
            'bottom': 'Grey',
            'style': 'Soft Elegance',
            'occasion': 'Casual / Brunch',
            'reason': 'Soft pastels complement fair skin beautifully'
        })
        combinations.append({
            'top': 'Emerald',
            'bottom': 'Black',
            'style': 'Jewel Tone Pop',
            'occasion': 'Party / Evening',
            'reason': 'Emerald creates stunning contrast with fair skin'
        })
    
    return combinations[:10]  # Return top 10 combinations


@app.route('/api/recommendations', methods=['POST'])
def get_recommendations():
    """Get outfit recommendations"""
    data = request.json
    skin_tone = data.get('skin_tone', session.get('skin_tone', 'Medium'))
    gender = data.get('gender', 'male')
    occasion = data.get('occasion')
    season = data.get('season')
    
    session['gender'] = gender
    if skin_tone:
        session['skin_tone'] = skin_tone
    
    recommendations = get_recommended_outfits(skin_tone, gender, occasion, season)
    color_palette = SKIN_TONE_COLORS.get(skin_tone, SKIN_TONE_COLORS['Medium'])
    outfit_combinations = get_outfit_combinations(skin_tone)
    
    return jsonify({
        'success': True,
        'skin_tone': skin_tone,
        'recommended_colors': color_palette,
        'outfit_combinations': outfit_combinations,
        'outfits': recommendations
    })


@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chatbot messages"""
    data = request.json
    user_message = data.get('message', '')
    
    context = {
        'name': data.get('name', session.get('name', 'Friend')),
        'gender': data.get('gender', session.get('gender', 'male')),
        'skin_tone': data.get('skin_tone', session.get('skin_tone', 'Medium'))
    }
    
    session['name'] = context['name']
    
    response = get_ai_response(user_message, context)
    
    return jsonify({
        'success': True,
        'response': response
    })


@app.route('/api/save_profile', methods=['POST'])
def save_profile():
    """Save user profile to session"""
    data = request.json
    session['name'] = data.get('name', '')
    session['gender'] = data.get('gender', 'male')
    return jsonify({'success': True})


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
