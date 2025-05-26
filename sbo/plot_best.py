import fitz  # PyMuPDF
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
import io
import os
import glob
# Parameters
pdf_files = glob.glob('/home/pdat/EI-UCB/sbo/best_results/*')  # Replace with your actual file names

# Convert PDFs to images
images = []
for path in pdf_files:
    doc = fitz.open(path)
    page = doc.load_page(0)
    pix = page.get_pixmap(dpi=300)
    img_bytes = pix.tobytes("png")
    img = Image.open(io.BytesIO(img_bytes))
    images.append(img)
    doc.close()

# Determine max width and height for uniform cell size
max_width = max(img.width for img in images)
max_height = max(img.height for img in images)


grid_rows, grid_cols = 3, 3
canvas_width = grid_cols * max_width
canvas_height = grid_rows * max_height
canvas = Image.new("RGB", (canvas_width, canvas_height), "white")

# Paste each image into its correct location
for idx, img in enumerate(images):
    row = idx // grid_cols
    col = idx % grid_cols

    # Center each image in its cell
    x_offset = col * max_width + (max_width - img.width) // 2
    y_offset = row * max_height + (max_height - img.height) // 2

    canvas.paste(img, (x_offset, y_offset))

# Save as a single-page PDF
canvas.save("combined_grid.pdf", "PDF", resolution=150.0)
