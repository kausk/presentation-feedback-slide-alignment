from PIL import Image
from pdf2image import convert_from_path



def save_pdf_as_pngs(SLIDES_PATH, SLIDES_SAVE):
    pages = convert_from_path(SLIDES_PATH, 500)
    for count, page in enumerate(pages):
        page.save(SLIDES_SAVE + str(count) + '-slide.png', 'PNG')