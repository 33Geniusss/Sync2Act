"""Generate the Sync2Act Windows icon and a PNG preview."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
CANVAS = 1024


def _font(size: int) -> ImageFont.FreeTypeFont:
    candidates = (
        Path("C:/Windows/Fonts/segoeuib.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf"),
    )
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default(size=size)


def build_icon() -> Image.Image:
    image = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))

    shadow = Image.new("RGBA", image.size, (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow)
    shadow_draw.rounded_rectangle((74, 86, 950, 962), radius=218, fill=(19, 33, 66, 105))
    image.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(30)))

    tile_mask = Image.new("L", image.size, 0)
    ImageDraw.Draw(tile_mask).rounded_rectangle((62, 62, 962, 962), radius=220, fill=255)
    gradient = Image.new("RGBA", image.size)
    pixels = gradient.load()
    top = (108, 135, 255)
    bottom = (43, 73, 190)
    for y in range(CANVAS):
        ratio = y / (CANVAS - 1)
        color = tuple(round(a + (b - a) * ratio) for a, b in zip(top, bottom, strict=True))
        for x in range(CANVAS):
            pixels[x, y] = (*color, 255)
    gradient.putalpha(tile_mask)
    image.alpha_composite(gradient)

    shine = Image.new("RGBA", image.size, (0, 0, 0, 0))
    shine_draw = ImageDraw.Draw(shine)
    shine_draw.ellipse((-180, -330, 850, 610), fill=(255, 255, 255, 24))
    shine_draw.arc((132, 145, 892, 905), 205, 337, fill=(196, 218, 255, 95), width=22)
    shine_draw.line((210, 764, 815, 238), fill=(221, 232, 255, 48), width=18)
    for x, y, radius in ((216, 759, 22), (510, 506, 17), (812, 242, 22)):
        shine_draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=(229, 238, 255, 135),
        )
    shine.putalpha(Image.composite(shine.getchannel("A"), Image.new("L", image.size), tile_mask))
    image.alpha_composite(shine)

    draw = ImageDraw.Draw(image)
    font = _font(282)
    label = "S2A"
    bounds = draw.textbbox((0, 0), label, font=font, stroke_width=2)
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    position = ((CANVAS - width) / 2, (CANVAS - height) / 2 - bounds[1] - 5)
    draw.text(
        position,
        label,
        font=font,
        fill=(255, 255, 255, 255),
        stroke_width=2,
        stroke_fill=(255, 255, 255, 255),
    )
    return image


def main() -> int:
    ASSETS.mkdir(parents=True, exist_ok=True)
    image = build_icon()
    png_path = ASSETS / "sync2act-icon.png"
    ico_path = ASSETS / "sync2act.ico"
    image.resize((512, 512), Image.Resampling.LANCZOS).save(png_path)
    image.save(
        ico_path,
        format="ICO",
        sizes=[
            (16, 16),
            (20, 20),
            (24, 24),
            (32, 32),
            (40, 40),
            (48, 48),
            (64, 64),
            (128, 128),
            (256, 256),
        ],
    )
    print(png_path)
    print(ico_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
