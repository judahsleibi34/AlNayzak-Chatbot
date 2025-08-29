from __future__ import annotations
import os, re, io, json, math, uuid, argparse, logging
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
from PIL import Image
import cv2

# Vector text (if available)
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

# OCR
try:
    import pytesseract
except Exception:
    pytesseract = None

# Optional Arabic display
try:
    import arabic_reshaper
    from bidi.algorithm import get_display
except Exception:
    arabic_reshaper = None
    get_display = None

# -----------------------------
# === CONFIGURATION ===
# -----------------------------

# --- CHANGE THESE PATHS ---
INPUT_PDF_PATH = "Data_pdf.pdf"  # <-- CHANGE THIS to your PDF path
OUTPUT_DIR     = "out"           # <-- Output directory

# --- OCR CONFIGURATION ---
TESSERACT_LANG = "ara+eng"
TESSERACT_PSM = 6  # Page segmentation mode (6: Uniform block of text)
OCR_DPI = 400      # Higher DPI for better OCR on complex layouts

# --- PROCESSING FLAGS ---
DEBUG_MODE = False  # Save intermediate images for debugging (set to False for speed)
# --- KEY CHANGE: Prioritize vector text ---
USE_OCR_ALWAYS = False  # Run OCR only if vector text is NOT found.
                        # This will give you clean text from the PDF layer,
                        # but will skip table structure detection via OCR.

# -----------------------------
# Logging Setup
# -----------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -----------------------------
# Arabic normalization utilities (Enhanced)
# -----------------------------

_ARABIC_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u06D6-\u06ED]")
_NOISE_CHARS = re.compile(r"[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFFa-zA-Z0-9\s\.\,\!\?\:\;\-\(\)\[\]\{\}\"\'\/\\@#\$%\^&\*\+\=\_\|\~\`]")

def strip_diacritics(s: str) -> str:
    return _ARABIC_DIACRITICS.sub("", s)

def strip_tatweel(s: str) -> str:
    return s.replace("ـ", "")

def clean_ocr_noise(s: str) -> str:
    """Remove OCR noise and artifacts"""
    if not s:
        return s
    # Remove common OCR artifacts
    s = re.sub(r'[|\\/_\-]{3,}', ' ', s)  # Long sequences of separators
    s = re.sub(r'\s+', ' ', s)  # Multiple spaces
    s = re.sub(r'^\s*[|\\/_\-]+\s*$', '', s)  # Lines with only separators
    s = _NOISE_CHARS.sub('', s)  # Remove non-Arabic/Latin noise
    return s.strip()

def normalize_arabic(s: str, map_ta_tarbuta_to_ha: bool=False) -> str:
    """Enhanced Arabic normalization for retrieval"""
    if not s:
        return s
    s = clean_ocr_noise(s)
    s = strip_diacritics(strip_tatweel(s))
    s = (s.replace("أ", "ا")
           .replace("إ", "ا")
           .replace("آ", "ا")
           .replace("ٱ", "ا")
           .replace("ى", "ي"))
    if map_ta_tarbuta_to_ha:
        s = s.replace("ة", "ه")
    return s

def to_human_rtl(s: str) -> str:
    """Optional: reshape/BiDi for display only"""
    if not s:
        return s
    if arabic_reshaper and get_display:
        try:
            return get_display(arabic_reshaper.reshape(s))
        except Exception:
            return s
    return s

# -----------------------------
# Utility to convert NumPy types for JSON serialization
# -----------------------------

def convert_numpy_types(obj):
    """Recursively convert NumPy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist() # Convert arrays to lists
    elif isinstance(obj, (np.bool_, bool)): # Handle boolean types
        return bool(obj)
    else:
        return obj

# -----------------------------
# Data models (unchanged)
# -----------------------------

def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"

@dataclass
class Node:
    node_id: str
    type: str  # heading, paragraph, list_item, kv, table, cell
    page: int
    bbox: Tuple[float, float, float, float]
    order: int
    confidence: float
    text_raw: str = ""
    text_norm: str = ""
    level: Optional[int] = None  # headings only
    key: Optional[Dict[str, Any]] = None   # for kv
    value: Optional[Dict[str, Any]] = None # for kv
    cells: Optional[List[Dict[str, Any]]] = None  # for table
    parent_id: Optional[str] = None
    path: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Filter out None, empty strings, lists, dicts
        filtered_d = {k: v for k, v in d.items() if v not in (None, "", [], {})}
        # Ensure all values are JSON serializable
        return convert_numpy_types(filtered_d)

@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    route: str
    route_tokens: List[str]
    node_type: str
    text_raw: str
    text_norm: str
    page: int
    bbox: Tuple[float, float, float, float]
    confidence: float
    source_ref: str

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Ensure all values are JSON serializable
        return convert_numpy_types(d)


# -----------------------------
# Enhanced imaging helpers for complex layouts
# -----------------------------

def pil_to_cv(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def cv_to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

# --- DESKREWING HAS BEEN REMOVED ---
# The deskewing was causing severe distortion on your document.
# This function is no longer needed.
# def deskew_gray(gray: np.ndarray) -> np.ndarray:
#     ... (removed) ...

def preprocess_for_ocr(bgr: np.ndarray, enhance_contrast: bool = True) -> np.ndarray:
    """Enhanced preprocessing for better OCR results on complex layouts.
    Deskewing is disabled to prevent destructive rotations."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Noise reduction - more aggressive for complex layouts
    gray = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)

    # Enhance contrast if needed
    if enhance_contrast:
        # CLAHE can help with local contrast in complex layouts
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)

    # --- Multi-thresholding approach for complex layouts ---
    # Try different thresholds and select the best one based on content
    thresh_methods = [
        # Standard adaptive thresholding
        cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2),
        cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10),
        cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 8),
        # Otsu's thresholding (global)
        cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        # Simple binary threshold (adjust value as needed)
        cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]
    ]

    # Select the threshold with best balance (not too much black/white)
    best_thresh = thresh_methods[0]
    best_score = float('inf')

    for thresh in thresh_methods:
        black_ratio = np.sum(thresh == 0) / thresh.size
        # Target a wider range of black pixel ratio for complex layouts
        score = abs(black_ratio - 0.2) # Target ~20% black pixels
        if score < best_score:
            best_score = score
            best_thresh = thresh

    # Invert if background is dark
    if np.mean(best_thresh) < 127:
        best_thresh = 255 - best_thresh

    # Morphological operations to clean up and separate text blocks
    # Use a smaller kernel for fine details in complex layouts
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    # Opening removes noise
    best_thresh = cv2.morphologyEx(best_thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    # Closing connects nearby components
    best_thresh = cv2.morphologyEx(best_thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    return best_thresh

def render_pdf_page(doc, pno: int, dpi: int=400) -> Image.Image:
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    page = doc.load_page(pno)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return img


# -----------------------------
# Enhanced OCR with Tesseract for Complex Layouts
# -----------------------------
# This section will only run if USE_OCR_ALWAYS is True or if there's no vector text.

def ocr_tesseract_words(img_bgr: np.ndarray, lang: str="ara+eng", psm: int=6, min_conf: float=30) -> List[Dict[str, Any]]:
    """Enhanced word extraction with better confidence filtering"""
    if not pytesseract:
        logger.warning("pytesseract not available, skipping OCR word extraction.")
        return []
    h, w = img_bgr.shape[:2]

    # --- Simplified config for better Arabic handling ---
    # Removed whitelist as it can interfere with complex scripts
    config = f'--oem 1 --psm {psm}'

    try:
        df = pytesseract.image_to_data(img_bgr, lang=lang, config=config, output_type=pytesseract.Output.DATAFRAME)
    except Exception as e:
        logger.error(f"Tesseract error: {e}")
        return []

    words = []
    if df is None or len(df) == 0:
        return words

    for _, row in df.iterrows():
        try:
            conf_val = float(row.get("conf", -1))
        except Exception:
            conf_val = -1

        if conf_val < min_conf:
            continue

        text = str(row.get("text") or "").strip()
        text = clean_ocr_noise(text)

        if not text or len(text) < 1:
            continue

        x, y, ww, hh = int(row["left"]), int(row["top"]), int(row["width"]), int(row["height"])

        # Filter out very small or very large bounding boxes (likely noise)
        if ww < 3 or hh < 3 or ww > w * 0.9 or hh > h * 0.8:
            continue

        conf = conf_val / 100.0
        words.append({
            "text": text,
            "bbox": (x, y, x+ww, y+hh),
            "conf": conf
        })
    return words

def words_to_lines(words: List[Dict[str, Any]], y_tol: int=8) -> List[Dict[str, Any]]:
    """Enhanced line grouping with better sorting"""
    if not words:
        return []

    # Filter out very low confidence words
    words = [w for w in words if w["conf"] > 0.3]

    # Sort by y-coordinate first, then x-coordinate
    words_sorted = sorted(words, key=lambda w: ((w["bbox"][1] + w["bbox"][3]) / 2.0, w["bbox"][0]))

    lines = []
    current = [words_sorted[0]]

    def cy(w):
        return (w["bbox"][1] + w["bbox"][3]) / 2.0

    for w in words_sorted[1:]:
        if abs(cy(w) - cy(current[-1])) <= y_tol:
            current.append(w)
        else:
            lines.append(current)
            current = [w]
    lines.append(current)

    out = []
    for idx, line_words in enumerate(lines):
        if not line_words:
            continue

        # Sort words in line by x-coordinate
        line_words = sorted(line_words, key=lambda w: w["bbox"][0])

        xs = [w["bbox"][0] for w in line_words]
        ys = [w["bbox"][1] for w in line_words]
        xe = [w["bbox"][2] for w in line_words]
        ye = [w["bbox"][3] for w in line_words]
        bbox = (min(xs), min(ys), max(xe), max(ye))

        # Join text with appropriate spacing
        text_parts = []
        for i, w in enumerate(line_words):
            if i > 0:
                # Add space if there's significant gap between words
                prev_x2 = line_words[i-1]["bbox"][2]
                curr_x1 = w["bbox"][0]
                if curr_x1 - prev_x2 > 10:  # Gap threshold
                    text_parts.append(" ")
            text_parts.append(w["text"])

        text = "".join(text_parts)
        text = clean_ocr_noise(text)

        if not text:
            continue

        conf = float(np.mean([w["conf"] for w in line_words]))
        out.append({
            "text": text,
            "bbox": bbox,
            "conf": conf,
            "words": line_words
        })
    return out


# -----------------------------
# Enhanced Table detection for Complex Layouts
# -----------------------------
# This section will only run if USE_OCR_ALWAYS is True.

def detect_lines(img: np.ndarray, direction: str = "horizontal") -> np.ndarray:
    """Detect horizontal or vertical lines in image, adjusted for complex layouts"""
    if direction == "horizontal":
        # Adjust kernel size based on image width
        kernel_width = max(40, img.shape[1]//50)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1))
    else:  # vertical
        # Adjust kernel size based on image height
        kernel_height = max(40, img.shape[0]//50)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_height))

    inv = 255 - img
    # Use morphological operations to enhance line detection
    lines = cv2.morphologyEx(inv, cv2.MORPH_OPEN, kernel) # Open to remove noise
    lines = cv2.morphologyEx(lines, cv2.MORPH_CLOSE, kernel) # Close to connect lines
    return lines

def detect_table_regions(bin_img: np.ndarray, min_area: int = 50000) -> List[Tuple[int,int,int,int]]:
    """Enhanced table region detection for complex layouts.
    Increased min_area to reduce false positives."""
    try:
        # Detect both horizontal and vertical lines
        h_lines = detect_lines(bin_img, "horizontal")
        v_lines = detect_lines(bin_img, "vertical")

        # Combine lines to form grid
        # Using bitwise_or can sometimes be more effective than bitwise_and for noisy lines
        grid = cv2.bitwise_or(h_lines, v_lines)

        # Dilate to connect nearby intersections, use a slightly larger kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        grid = cv2.dilate(grid, kernel, iterations=2)

        # Find contours of the combined grid
        contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        table_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h

            # Filter by area and aspect ratio, adjusted for complex layouts
            # Allow for more variation in table shapes
            # SIGNIFICANTLY INCREASED min_area
            if area > min_area and w > 100 and h > 50: # Increased min width/height too
                # Add padding to the detected region
                pad = 5
                x0, y0 = max(0, x-pad), max(0, y-pad)
                x1, y1 = min(bin_img.shape[1], x+w+pad), min(bin_img.shape[0], y+h+pad)
                table_regions.append((x0, y0, x1, y1)) # Store padded bbox

        logger.debug(f"Detected {len(table_regions)} potential table regions.")
        return table_regions
    except Exception as e:
        logger.error(f"Table detection error: {e}")
        return []

def detect_table_cells_enhanced(bin_img: np.ndarray, table_region: Tuple[int,int,int,int]) -> List[Tuple[int,int,int,int]]:
    """Enhanced cell detection within table region"""
    try:
        x0, y0, x1, y1 = table_region
        # Ensure ROI is within image bounds
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(bin_img.shape[1], x1), min(bin_img.shape[0], y1)
        roi = bin_img[y0:y1, x0:x1]

        if roi.size == 0:
            return []

        # Detect grid lines in ROI
        h_lines = detect_lines(roi, "horizontal")
        v_lines = detect_lines(roi, "vertical")

        # --- Improved line position detection ---
        # Use HoughLinesP for more robust line detection in complex tables
        lines_h = cv2.HoughLinesP(h_lines, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)
        lines_v = cv2.HoughLinesP(v_lines, 1, np.pi / 180, threshold=100, minLineLength=30, maxLineGap=10)

        h_positions = [y0] # Start with top of table
        v_positions = [x0] # Start with left of table

        if lines_h is not None:
            for line in lines_h:
                x1_l, y1_l, x2_l, y2_l = line[0]
                # Average y position for horizontal line
                h_positions.append(y0 + (y1_l + y2_l) // 2)

        if lines_v is not None:
            for line in lines_v:
                x1_l, y1_l, x2_l, y2_l = line[0]
                 # Average x position for vertical line
                v_positions.append(x0 + (x1_l + x2_l) // 2)

        # Add bottom and right edges of the table region
        h_positions.append(y1)
        v_positions.append(x1)

        # Sort and remove duplicates
        h_positions = sorted(set(h_positions))
        v_positions = sorted(set(v_positions))

        # Create cells from line intersections
        cells = []
        for i in range(len(h_positions) - 1):
            for j in range(len(v_positions) - 1):
                cell_y0, cell_y1 = h_positions[i], h_positions[i + 1]
                cell_x0, cell_x1 = v_positions[j], v_positions[j + 1]

                # Add some padding, ensuring bounds
                pad = 2
                cell_x0_p, cell_y0_p = max(x0, cell_x0 + pad), max(y0, cell_y0 + pad)
                cell_x1_p, cell_y1_p = min(x1, cell_x1 - pad), min(y1, cell_y1 - pad)

                if cell_x1_p > cell_x0_p and cell_y1_p > cell_y0_p:
                     # Store cell bbox in absolute coordinates
                    cells.append((cell_x0_p, cell_y0_p, cell_x1_p, cell_y1_p))

        logger.debug(f"Detected {len(cells)} cells in table region {table_region}.")
        return cells
    except Exception as e:
        logger.error(f"Cell detection error: {e}")
        return []

def extract_cell_text(img_bgr: np.ndarray, cell_bbox: Tuple[int,int,int,int], lang: str = "ara+eng") -> str:
    """Extract text from individual cell with enhanced preprocessing"""
    try:
        x0, y0, x1, y1 = cell_bbox

        # Extract cell ROI with padding, ensuring bounds
        padding = 5
        roi_y0, roi_y1 = max(0, y0-padding), min(img_bgr.shape[0], y1+padding)
        roi_x0, roi_x1 = max(0, x0-padding), min(img_bgr.shape[1], x1+padding)
        roi = img_bgr[roi_y0:roi_y1, roi_x0:roi_x1]

        if roi.size == 0:
            return ""

        # Enhanced preprocessing for cell - might need different settings
        roi_bin = preprocess_for_ocr(roi, enhance_contrast=True)

        # --- Simplified config for cell OCR ---
        config = f'--oem 1 --psm 6' # PSM 6 for block of text, removed whitelist

        try:
            text = pytesseract.image_to_string(cv2.cvtColor(roi_bin, cv2.COLOR_GRAY2BGR),
                                             lang=lang, config=config)
            text = clean_ocr_noise(text.strip())

            # Remove newlines and excessive whitespace
            text = re.sub(r'\s+', ' ', text)
            return text

        except Exception as e:
            logger.warning(f"OCR error for cell ({cell_bbox}): {e}")
            return ""

    except Exception as e:
        logger.error(f"Cell text extraction error: {e}")
        return ""

def cells_to_structured_grid(cells: List[Tuple[int,int,int,int]], tolerance: int = 15) -> Tuple[List[List[Dict]], int,int]:
    """Convert cell bounding boxes to structured grid"""
    if not cells:
        return [], 0, 0

    # Get unique row and column positions based on center points
    y_centers = [(y0 + y1) // 2 for x0, y0, x1, y1 in cells]
    x_centers = [(x0 + x1) // 2 for x0, y0, x1, y1 in cells]

    # Cluster Y positions (rows) using a simple distance-based approach
    unique_rows = []
    for y in sorted(set(y_centers)):
        if not unique_rows or abs(y - unique_rows[-1]) > tolerance:
            unique_rows.append(y)

    # Cluster X positions (columns)
    unique_cols = []
    for x in sorted(set(x_centers)):
        if not unique_cols or abs(x - unique_cols[-1]) > tolerance:
            unique_cols.append(x)

    # Create grid
    grid = [[None for _ in range(len(unique_cols))] for _ in range(len(unique_rows))]

    # Place cells in grid
    for i, (x0, y0, x1, y1) in enumerate(cells):
        y_center = (y0 + y1) // 2
        x_center = (x0 + x1) // 2

        # Find closest row and column
        row_idx = min(range(len(unique_rows)), key=lambda i: abs(unique_rows[i] - y_center))
        col_idx = min(range(len(unique_cols)), key=lambda i: abs(unique_cols[i] - x_center))

        if grid[row_idx][col_idx] is None:
            grid[row_idx][col_idx] = {
                "bbox": (x0, y0, x1, y1),
                "row": row_idx,
                "col": col_idx
            }

    return grid, len(unique_rows), len(unique_cols)


# -----------------------------
# Vector text extraction (Improved for Complex Layouts)
# -----------------------------

def extract_vector_blocks(page) -> List[Dict[str, Any]]:
    """Extract blocks with detailed layout info using PyMuPDF's dict output"""
    out = []
    try:
        # Use 'dict' to get detailed layout information
        text_dict = page.get_text("dict")
    except Exception as e:
        logger.warning(f"Failed to extract vector blocks: {e}")
        return out

    order = 0
    for b in text_dict.get("blocks", []):
        # Check if it's a text block (not an image)
        if "lines" not in b:
            continue
        block_bbox = tuple(b.get("bbox", (0,0,0,0)))
        lines_text = []
        spans_meta = []
        for ln in b["lines"]:
            for sp in ln.get("spans", []):
                t = sp.get("text", "")
                if t.strip():
                    lines_text.append(t)
                spans_meta.append({
                    "size": sp.get("size", 0.0),
                    "flags": sp.get("flags", 0),
                    "font": sp.get("font", ""),
                    "color": sp.get("color", 0),
                    "text": sp.get("text","")
                })
        if not lines_text:
            continue
        out.append({
            "bbox": block_bbox,
            "text": "\n".join(lines_text),
            "spans": spans_meta,
            "order": order
        })
        order += 1
    return out

def vector_font_stats(blocks: List[Dict[str, Any]]) -> Tuple[float, float]:
    sizes = []
    for b in blocks:
        for sp in b.get("spans", []):
            if sp.get("text", "").strip():
                sizes.append(sp.get("size",0.0))
    if not sizes:
        return (0.0, 0.0)
    sizes = sorted(sizes)
    median = sizes[len(sizes)//2]
    p90 = sizes[int(0.9*len(sizes))]
    return median, p90


# -----------------------------
# Heading detection & hierarchy (unchanged)
# -----------------------------

HEADING_RE = re.compile(r"^\s*(\d+(?:\.\d+){0,5})\s*[)\-:\u061F\u060C\u066A]?\s*(.+)$")

def is_bold(flags: int) -> bool:
    return bool(flags & 2)

def classify_heading_vector(block: Dict[str, Any], median: float, p90: float) -> Tuple[bool, Optional[int], Optional[str]]:
    text = block["text"].strip().replace("\n", " ")
    m = HEADING_RE.match(text)
    if m:
        numbering = m.group(1)
        level = numbering.count(".") + 1
        return True, level, numbering
    sizes = [sp["size"] for sp in block.get("spans", []) if sp.get("text","").strip()]
    bolds = [is_bold(sp.get("flags",0)) for sp in block.get("spans", []) if sp.get("text","").strip()]
    if not sizes:
        return False, None,None
    avg_size = sum(sizes)/len(sizes)
    # More flexible heading detection for complex layouts
    if avg_size >= max(p90, median + 1.5) or (any(bolds) and avg_size >= median):
        return True, None,None
    return False, None, None

def classify_heading_ocr(line: Dict[str, Any], line_height_median: float) -> Tuple[bool, Optional[int], Optional[str]]:
    text = line["text"].strip()
    m = HEADING_RE.match(text)
    if m:
        numbering = m.group(1)
        level = numbering.count(".") + 1
        return True, level, numbering
    _,y0,_,y1 = line["bbox"]
    h = y1 - y0
    # Adjust threshold for complex layouts
    if h >= line_height_median * 1.3 and len(text) <= 100: # Slightly lowered height threshold
        return True, None, None
    return False, None, None

def update_heading_path(current_path: List[str], numbering: Optional[str], level: Optional[int]) -> List[str]:
    if numbering:
        parts = numbering.split(".")
        return parts
    if level is None:
        if not current_path:
            return ["1"]
        try:
            last = int(current_path[-1])
            new = current_path[:-1] + [str(last+1)]
        except:
            new = current_path + ["1"]
        return new
    if level <= len(current_path):
        new = current_path[:]
        try:
            new[level-1] = str(int(new[level-1]) + 1)
        except:
            pass
        return new[:level]
    else:
        return current_path + ["1"]

def route_str(path: List[str]) -> str:
    return " > ".join(path)


# -----------------------------
# Node builders (updated for enhanced tables)
# -----------------------------

def make_heading_node(text: str, page: int, bbox, order: int, conf: float, level: Optional[int], path: List[str]) -> Node:
    return Node(
        node_id=new_id("h"),
        type="heading",
        page=page,
        bbox=tuple(map(float, bbox)),
        order=order,
        confidence=conf,
        text_raw=text,
        text_norm=normalize_arabic(text),
        level=level,
        parent_id=None,
        path=path
    )

def make_para_node(text: str, page: int, bbox, order: int, conf: float, parent_id: Optional[str], path: List[str]) -> Node:
    return Node(
        node_id=new_id("p"),
        type="paragraph",
        page=page,
        bbox=tuple(map(float, bbox)),
        order=order,
        confidence=conf,
        text_raw=text,
        text_norm=normalize_arabic(text),
        parent_id=parent_id,
        path=path
    )

def make_list_node(text: str, page: int, bbox, order: int, conf: float, parent_id: Optional[str], path: List[str]) -> Node:
    return Node(
        node_id=new_id("li"),
        type="list_item",
        page=page,
        bbox=tuple(map(float, bbox)),
        order=order,
        confidence=conf,
        text_raw=text,
        text_norm=normalize_arabic(text),
        parent_id=parent_id,
        path=path
    )

def make_kv_node(key: str, val: str, page: int, bbox, order: int, conf: float, parent_id: Optional[str], path: List[str]) -> Node:
    return Node(
        node_id=new_id("kv"),
        type="kv",
        page=page,
        bbox=tuple(map(float, bbox)),
        order=order,
        confidence=conf,
        key={"text": key},
        value={"text": val},
        text_raw=f"{key}: {val}",
        text_norm=normalize_arabic(f"{key} {val}"),
        parent_id=parent_id,
        path=path
    )

def make_table_node(page: int, bbox, order: int, conf: float, parent_id: Optional[str], path: List[str], cells: List[Dict[str, Any]]) -> Node:
    # Ensure cell data is also converted for serialization
    converted_cells = [convert_numpy_types(cell) for cell in cells]
    return Node(
        node_id=new_id("tbl"),
        type="table",
        page=page,
        bbox=tuple(map(float, bbox)),
        order=order,
        confidence=conf,
        cells=converted_cells,
        parent_id=parent_id,
        path=path
    )


# -----------------------------
# Enhanced Chunker
# -----------------------------

def chunks_from_nodes(document_id: str, nodes: List[Node]) -> List[Chunk]:
    chunks: List[Chunk] = []
    heading_texts: Dict[str, str] = {}

    # Build heading texts dictionary
    for n in nodes:
        if n.type == "heading":
            route_key = route_str(n.path) if n.path else "1"
            heading_texts[route_key] = n.text_norm

    for n in nodes:
        if n.type in ("paragraph", "list_item", "kv"):
            text_norm = n.text_norm or ""
            text_raw = n.text_raw or ""

            if n.type == "kv" and n.key and n.value:
                key_text = n.key.get('text', '') if isinstance(n.key, dict) else str(n.key)
                val_text = n.value.get('text', '') if isinstance(n.value, dict) else str(n.value)
                text_norm = normalize_arabic(f"{key_text} {val_text}")
                text_raw = f"{key_text}: {val_text}"

            # Handle empty paths
            route = route_str(n.path) if n.path else "1"
            route_tokens = n.path[:] if n.path else ["1"]

            if route in heading_texts and heading_texts[route]:
                route_tokens.append(heading_texts[route])

            chunk = Chunk(
                chunk_id=f"{document_id}|{route}|{n.node_id}",
                document_id=document_id,
                route=route,
                route_tokens=route_tokens,
                node_type=n.type,
                text_raw=text_raw,
                text_norm=text_norm,
                page=n.page,
                bbox=n.bbox,
                confidence=n.confidence,
                source_ref=f"{document_id}.pdf#page={n.page}"
            )
            chunks.append(chunk)
        elif n.type == "table" and n.cells:
            # Enhanced table chunking - create chunks for rows and individual cells
            rows: Dict[int, List[Dict[str,Any]]] = {}
            for c in n.cells:
                row_idx = int(c.get("row", c.get("r", 0)))
                rows.setdefault(row_idx, []).append(c)

            # Create row-level chunks
            for r_idx, cells in rows.items():
                cells_sorted = sorted(cells, key=lambda c: int(c.get("col", c.get("c", 0))))

                # Filter out empty cells and clean text
                non_empty_cells = []
                for c in cells_sorted:
                    text = c.get("text_raw", c.get("text", "")).strip()
                    if text and text not in ["|", "-", "_", " ", ""]:
                        non_empty_cells.append(text)

                if not non_empty_cells:
                    continue

                # Join cells with proper separator
                txt_raw = " | ".join(non_empty_cells)
                txt_norm = normalize_arabic(" ".join(non_empty_cells))

                # Skip if the row is just noise
                if len(txt_norm.strip()) < 3:
                    continue

                route = route_str(n.path) if n.path else "1"
                route_tokens = n.path[:] if n.path else ["1"]

                chunk = Chunk(
                    chunk_id=f"{document_id}|{route}|{n.node_id}|row{r_idx}",
                    document_id=document_id,
                    route=route,
                    route_tokens=route_tokens,
                    node_type="table_row",
                    text_raw=txt_raw,
                    text_norm=txt_norm,
                    page=n.page,
                    bbox=n.bbox, # Use table bbox for row chunk
                    confidence=n.confidence,
                    source_ref=f"{document_id}.pdf#page={n.page}"
                )
                chunks.append(chunk)

            # Also create individual cell chunks for better granular search
            for i, c in enumerate(n.cells):
                text = c.get("text_raw", c.get("text", "")).strip()
                if text and len(text) > 2 and text not in ["|", "-", "_", " "]:
                    route = route_str(n.path) if n.path else "1"
                    route_tokens = n.path[:] if n.path else ["1"]

                    chunk = Chunk(
                        chunk_id=f"{document_id}|{route}|{n.node_id}|cell{i}",
                        document_id=document_id,
                        route=route,
                        route_tokens=route_tokens,
                        node_type="table_cell",
                        text_raw=text,
                        text_norm=normalize_arabic(text),
                        page=n.page,
                        bbox=tuple(c.get("bbox", n.bbox)), # Use cell bbox
                        confidence=float(c.get("conf", n.confidence)), # Ensure float
                        source_ref=f"{document_id}.pdf#page={n.page}"
                    )
                    chunks.append(chunk)

    return chunks


# -----------------------------
# Page-level extraction (Enhanced OCR version)
# -----------------------------

BULLET_RE = re.compile(r"^\s*[\-\u2022\u00B7\u25CF\u25A0\u25E6\u2219]+")

def detect_list_or_kv(line_text: str) -> Tuple[str, Optional[Tuple[str,str]]]:
    t = line_text.strip()
    if not t:
        return "paragraph", None
    if ":" in t or "：" in t or "؛" in t:
        parts = re.split(r"[:：؛]", t, maxsplit=1)
        if len(parts) == 2 and parts[0].strip() and parts[1].strip():
            return "kv", (parts[0].strip(), parts[1].strip())
    if BULLET_RE.match(t):
        t2 = BULLET_RE.sub("", t).strip()
        return "list_item", (None, t2)
    if re.match(r"^\s*[\(\[]?\d+[\)\.\-]\s+", t):
        return "list_item", (None, t)
    return "paragraph", None

def page_has_vector_text(page) -> bool:
    try:
        plain = page.get_text("text").strip()
        return len(plain) > 0
    except Exception:
        return False

def extract_page_nodes_vector(document_id: str, page, page_num: int, order_start: int, current_heading_id: Optional[str], current_path: List[str]) -> Tuple[List[Node], Optional[str], List[str], int]:
    nodes: List[Node] = []
    blocks = extract_vector_blocks(page)
    if not blocks:
        return nodes, current_heading_id, current_path, order_start

    median, p90 = vector_font_stats(blocks)
    order = order_start
    last_heading_id = current_heading_id
    path = current_path[:]

    for b in blocks:
        text = b["text"].replace("\r", " ").strip()
        if not text:
            continue
        is_h, level, numbering = classify_heading_vector(b, median, p90)
        if is_h:
            path = update_heading_path(path, numbering, level)
            hnode = make_heading_node(text, page_num, b["bbox"], order, 0.99, level, path[:])
            nodes.append(hnode)
            last_heading_id = hnode.node_id
            order += 1
            continue

        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        if not lines:
            continue
        for ln in lines:
            typ, kv = detect_list_or_kv(ln)
            bbox = b["bbox"]
            if typ == "kv" and kv:
                node = make_kv_node(kv[0], kv[1], page_num, bbox, order, 0.98, last_heading_id, path[:])
            elif typ == "list_item" and kv:
                node = make_list_node(kv[1], page_num, bbox, order, 0.98, last_heading_id, path[:])
            else:
                node = make_para_node(ln, page_num, bbox, order, 0.98, last_heading_id, path[:])
            nodes.append(node); order += 1

    return nodes, last_heading_id, path, order

def extract_page_nodes_ocr(document_id: str, img_bgr: np.ndarray, page_num: int, order_start: int, current_heading_id: Optional[str], current_path: List[str], lang: str, psm: int, debug_dir: Optional[str]) -> Tuple[List[Node],Optional[str], List[str],int]:
    """Extract nodes using OCR. This will only run if USE_OCR_ALWAYS is True."""
    nodes: List[Node] = []
    order = order_start
    last_heading_id = current_heading_id
    path = current_path[:]

    # Enhanced preprocessing
    bin_img = preprocess_for_ocr(img_bgr, enhance_contrast=True)

    # Debug: Save preprocessed image
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        debug_pre_path = os.path.join(debug_dir, f"page{page_num:03d}_preprocessed.png")
        cv2.imwrite(debug_pre_path, bin_img)
        debug_orig_path = os.path.join(debug_dir, f"page{page_num:03d}_original.png")
        cv2.imwrite(debug_orig_path, img_bgr)
        logger.debug(f"Saved debug images for page {page_num}")

    # Extract regular text with enhanced OCR
    words = ocr_tesseract_words(cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR), lang=lang, psm=psm, min_conf=40)
    lines = words_to_lines(words, y_tol=10) # Increased tolerance for complex layouts

    if lines:
        heights = [(ln["bbox"][3]-ln["bbox"][1]) for ln in lines]
        line_height_median = float(np.median(heights)) if heights else 12.0
    else:
        line_height_median = 12.0

    # Process text lines
    for ln in lines:
        is_h, level, numbering = classify_heading_ocr(ln, line_height_median)
        if is_h:
            path = update_heading_path(path, numbering, level)
            hnode = make_heading_node(ln["text"], page_num, ln["bbox"], order, ln["conf"], level, path[:])
            nodes.append(hnode)
            last_heading_id = hnode.node_id
            order += 1
        else:
            typ, kv = detect_list_or_kv(ln["text"])
            if typ == "kv" and kv:
                node = make_kv_node(kv[0], kv[1], page_num, ln["bbox"], order, ln["conf"], last_heading_id, path[:])
            elif typ == "list_item" and kv:
                node = make_list_node(kv[1], page_num, ln["bbox"], order, ln["conf"], last_heading_id, path[:])
            else:
                node = make_para_node(ln["text"], page_num, ln["bbox"], order, ln["conf"], last_heading_id, path[:])
            nodes.append(node); order += 1

    # Enhanced table detection and extraction for complex layouts
    # SIGNIFICANTLY INCREASED min_area
    table_regions = detect_table_regions(bin_img, min_area=50000)

    for table_region in table_regions:
        logger.info(f"Processing table on page {page_num} at {table_region}")
        cells_rects = detect_table_cells_enhanced(bin_img, table_region)

        if len(cells_rects) < 4:  # Need at least 4 cells for a meaningful table
            logger.debug("Skipping table: Less than 4 cells detected.")
            continue

        # Convert to structured grid
        grid, num_rows, num_cols = cells_to_structured_grid(cells_rects, tolerance=20) # Increased tolerance

        if num_rows < 2 or num_cols < 2:
            logger.debug("Skipping table: Less than 2x2 grid.")
            continue

        # Extract text from each cell
        cells_payload = []
        total_conf = 0
        valid_cells = 0

        for row_idx, row in enumerate(grid):
            for col_idx, cell_info in enumerate(row):
                if cell_info is None:
                    continue

                cell_bbox = cell_info["bbox"]
                cell_text = extract_cell_text(img_bgr, cell_bbox, lang) # Use original image for cell OCR

                # Only include cells with meaningful content
                if cell_text and len(cell_text.strip()) > 0:
                    cell_conf = 0.8 if len(cell_text.strip()) > 2 else 0.6
                    total_conf += cell_conf
                    valid_cells += 1

                    cells_payload.append({
                        "row": row_idx,
                        "col": col_idx,
                        "bbox": list(cell_bbox),
                        "text_raw": cell_text,
                        "conf": cell_conf
                    })

        # Only create table if we have enough valid cells
        if valid_cells >= 4:
            avg_conf = total_conf / valid_cells if valid_cells > 0 else 0.7
            tnode = make_table_node(page_num, table_region, order, avg_conf, last_heading_id, path[:], cells_payload)
            nodes.append(tnode)
            order += 1
            logger.info(f"Added table node with {valid_cells} cells on page {page_num}.")
        else:
             logger.debug(f"Skipping table: Only {valid_cells} valid cells after OCR.")

    return nodes, last_heading_id, path, order


# -----------------------------
# Enhanced Orchestrator for Complex Layouts
# -----------------------------

def process_pdf(pdf_path: str, out_dir: str, dpi: int=400, psm: int=6, lang: str="ara+eng", debug: bool=True) -> Tuple[List[Node], List[Chunk]]:
    assert fitz is not None, "PyMuPDF (fitz) is required to open PDF"
    doc = fitz.open(pdf_path)
    document_id = os.path.splitext(os.path.basename(pdf_path))[0]
    all_nodes: List[Node] = []

    order = 0
    current_heading_id = None
    current_path: List[str] = []

    debug_dir = os.path.join(out_dir, "debug") if debug else None
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)

    logger.info(f"Processing {len(doc)} pages...")

    for pno in range(len(doc)):
        page_num = pno + 1
        logger.info(f"Processing page {page_num}...")

        page = doc.load_page(pno)
        has_vec = page_has_vector_text(page)

        vector_nodes = []
        if has_vec:
            # Extract vector text first
            vector_nodes, current_heading_id, current_path, order = extract_page_nodes_vector(
                document_id, page, page_num, order, current_heading_id, current_path
            )
            logger.debug(f"Extracted {len(vector_nodes)} vector nodes from page {page_num}.")
            all_nodes.extend(vector_nodes)
        else:
            logger.debug(f"No vector text found on page {page_num}.")

        # Run OCR for table detection and additional text ONLY if configured to do so
        # AND only if there was no vector text OR if vector text was found but we still want OCR.
        if USE_OCR_ALWAYS or (not has_vec):
             if not pytesseract:
                 logger.warning("OCR requested but pytesseract is not available. Skipping OCR for page {page_num}.")
                 continue
             img = render_pdf_page(doc, pno, dpi=dpi)
             img_bgr = pil_to_cv(img)
             ocr_nodes, current_heading_id, current_path, order = extract_page_nodes_ocr(
                 document_id, img_bgr, page_num, order, current_heading_id, current_path, lang, psm, debug_dir
             )
             logger.debug(f"Extracted {len(ocr_nodes)} OCR nodes from page {page_num}.")
             all_nodes.extend(ocr_nodes)

    # Improved deduplication - less aggressive, based on bbox and type
    logger.info("Deduplicating nodes...")
    deduped: List[Node] = []
    seen_keys = set()

    # Sort nodes by page and then by vertical position (top of bbox)
    all_nodes.sort(key=lambda n: (n.page, n.bbox[1]))

    for n in all_nodes:
        # Create a key based on type, page, and rounded bbox to identify likely duplicates
        # This avoids text-based deduplication which can fail with OCR variations
        key = (n.type, n.page, tuple(round(coord / 5.0) * 5 for coord in n.bbox)) # Round to 5px grid

        # Special handling for different node types
        if n.type == "table":
            # Tables are less likely to be duplicated by vector/OCR, but check bbox
            table_key = (n.type, n.page, tuple(round(coord / 10.0) * 10 for coord in n.bbox), len(n.cells or []))
            if table_key not in seen_keys:
                seen_keys.add(table_key)
                deduped.append(n)
        elif key not in seen_keys:
            seen_keys.add(key)
            deduped.append(n)
        # else: Duplicate node skipped

    logger.info(f"Kept {len(deduped)} nodes after deduplication (from {len(all_nodes)}).")

    logger.info(f"Creating chunks from {len(deduped)} nodes...")
    chunks = chunks_from_nodes(document_id, deduped)

    # Filter out very short or noisy chunks
    filtered_chunks = []
    for chunk in chunks:
        text = chunk.text_norm.strip()
        # Skip very short chunks or chunks with mostly symbols
        if len(text) >= 3 and not re.match(r'^[\|\-_\s]*$', text):
            filtered_chunks.append(chunk)
        # else: Chunk filtered out

    logger.info(f"Created {len(filtered_chunks)} chunks (from {len(chunks)}).")
    return deduped, filtered_chunks

def save_jsonl(path: str, items: List[Dict[str,Any]]):
    """Save items to a JSON file, ensuring NumPy types are handled."""
    try:
        # Convert all items using our utility before saving
        converted_items = [convert_numpy_types(item) for item in items]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(converted_items, f, ensure_ascii=False, indent=2)
        logger.info(f"Successfully saved {len(converted_items)} items to {path}")
    except Exception as e:
        logger.error(f"Failed to save JSON to {path}: {e}")
        raise # Re-raise the exception after logging

def main():
    # Use the top-level variables
    pdf_path = INPUT_PDF_PATH
    out_dir = OUTPUT_DIR

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    os.makedirs(out_dir, exist_ok=True)

    nodes, chunks = process_pdf(
        pdf_path=pdf_path,
        out_dir=out_dir,
        dpi=OCR_DPI,
        psm=TESSERACT_PSM,
        lang=TESSERACT_LANG,
        debug=DEBUG_MODE
    )

    # Save the results using the fixed save function
    save_jsonl(os.path.join(out_dir, "nodes.json"), [n.to_dict() for n in nodes])
    save_jsonl(os.path.join(out_dir, "chunks.json"), [c.to_dict() for c in chunks])

    # Print statistics
    node_types = {}
    chunk_types = {}
    for n in nodes:
        node_types[n.type] = node_types.get(n.type, 0) + 1
    for c in chunks:
        chunk_types[c.node_type] = chunk_types.get(c.node_type, 0) + 1

    logger.info(f"\n✅ Extraction Complete!")
    logger.info(f"📊 Node Statistics: {dict(node_types)}")
    logger.info(f"📊 Chunk Statistics: {dict(chunk_types)}")
    logger.info(f"📁 Output saved to: {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    main()