from PIL import Image
from pdf2image import convert_from_path

SLIDES_PATH = '/Users/ksk/img/metrics_newdata/slides.pdf'

SLIDES_SAVE = '/Users/ksk/img/metrics_newdata/new_slides/'

pages = convert_from_path(SLIDES_PATH, 500)

for count, page in enumerate(pages):
    page.save(SLIDES_SAVE + str(count) + '-slide.png', 'PNG')