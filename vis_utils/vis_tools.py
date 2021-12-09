import numpy as np
import cv2


def create_black_image(img_shape: tuple):
    """Create a black image of given shape"""
    w, h, c = img_shape
    img = np.zeros([w, h, c], dtype=np.uint8)
    return img


def draw_sticker(img, point, thresh=240, replace_alpha=True, sticker_path=None, sticker_shape=(400, 400)):
    if replace_alpha and sticker_path is not None:
        sticker = _replace_alpha(sticker_path)
    elif sticker_path is not None:
        sticker = cv2.imread(sticker_path)
    else:
        return

    sticker = cv2.resize(sticker, sticker_shape, interpolation=cv2.INTER_LINEAR)
    w, h = sticker_shape  # width and height of sticker

    w_center = int(point[0])
    h_center = int(point[1])

    back = img[h_center - int(h / 2):h_center + h - int(h / 2), w_center - int(w / 2):w_center + w - int(w / 2)]
    # sticker = cv2.resize(sticker, (w, h))

    try:
        inserted = np.where(sticker > thresh, back, sticker)
        img[h_center - int(h / 2):h_center + h - int(h / 2), w_center - int(w / 2):w_center + w - int(w / 2)] = inserted
    except Exception as e:
        print(f"{type(e)}: {e}")

    return img


def _increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


def _replace_alpha(img_path):
    """Loads image with alpha channel and replaces it with white background"""
    image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

    try:
        # make mask of where the transparent bits are
        trans_mask = image[:, :, 3] == 0
        # replace areas of transparency with white and not transparent
        image[trans_mask] = [255, 255, 255, 255]

    except Exception as e:
        print(f"{type(e)}: {e}")

    # new image without alpha channel...
    new_img = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    return new_img


def generate_russian_text(text="текст на РУССКОМ", level=0, one_line=False, shape=(800, 200),
                          back_color=(255, 255, 255), font_size=54, font_color=(0, 0, 0),
                          outfile="text.jpg"):
    from PIL import Image, ImageDraw, ImageFont
    width, height = shape

    challenge = "это слово" if level == 0 else "эту фразу"
    unicode_text0 = f"Попробуй произнести {challenge}"
    unicode_text1 = u"\u2605" + u"\u2606" + f"{text}" + u"\u2606" + u"\u2605"

    im = Image.new("RGB", (width, height), back_color)
    draw = ImageDraw.Draw(im)

    if one_line:
        unicode_font = ImageFont.truetype("DejaVuSans.ttf", font_size)
        draw.text((42, 80), f"{text}", font=unicode_font, fill=font_color)
    else:
        unicode_font = ImageFont.truetype("DejaVuSans.ttf", font_size - 10)
        draw.text((42, 10), unicode_text0, font=unicode_font, fill=font_color)
        unicode_font = ImageFont.truetype("DejaVuSans.ttf", font_size)
        draw.text((42, 80), unicode_text1, font=unicode_font, fill=font_color)

    im.save(outfile)


generate_russian_text("Тихо...", one_line=True, shape=(400, 200), font_size=42, outfile="asr.jpg")
