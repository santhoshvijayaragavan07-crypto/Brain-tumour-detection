from pathlib import Path
from PIL import Image, ImageOps, ImageDraw, ImageFont

BASE = Path("outputs/final_explainability")
INDIVIDUAL = BASE / "individual"
MONTAGES = BASE / "montages"

MONTAGES.mkdir(parents=True, exist_ok=True)

# Samples grouped by class
groups = {
    "Glioma": [1, 2],
    "Meningioma": [3, 4],
    "No Tumor": [5, 6],
    "Pituitary": [7, 8],
}

def load_image(path):
    img = Image.open(path).convert("RGB")
    return img

def add_title(img, title, width):
    canvas = Image.new("RGB", (width, img.height + 45), "white")
    canvas.paste(img, (0, 45))

    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("arial.ttf", 22)
    except:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), title, font=font)
    text_width = bbox[2] - bbox[0]

    draw.text(
        ((width - text_width) // 2, 10),
        title,
        fill="black",
        font=font
    )

    return canvas


for class_name, indices in groups.items():

    rows = []

    for idx in indices:

        gradcam_file = list(
            INDIVIDUAL.glob(f"{idx:02d}_*_gradcam.png")
        )[0]

        gradcam_pp_file = list(
            INDIVIDUAL.glob(f"{idx:02d}_*_gradcam_plus_plus.png")
        )[0]

        gradcam = load_image(gradcam_file)
        gradcam_pp = load_image(gradcam_pp_file)

        # Resize both images to same size
        target_size = (500, 500)

        gradcam = ImageOps.contain(
            gradcam,
            target_size
        )

        gradcam_pp = ImageOps.contain(
            gradcam_pp,
            target_size
        )

        gradcam = add_title(
            gradcam,
            f"Sample {idx:02d} - Grad-CAM",
            500
        )

        gradcam_pp = add_title(
            gradcam_pp,
            f"Sample {idx:02d} - Grad-CAM++",
            500
        )

        row_width = 1000
        row_height = max(
            gradcam.height,
            gradcam_pp.height
        )

        row = Image.new(
            "RGB",
            (row_width, row_height),
            "white"
        )

        row.paste(gradcam, (0, 0))
        row.paste(gradcam_pp, (500, 0))

        rows.append(row)

    # Final montage
    width = 1000
    height = sum(row.height for row in rows) + 80

    montage = Image.new(
        "RGB",
        (width, height),
        "white"
    )

    draw = ImageDraw.Draw(montage)

    try:
        title_font = ImageFont.truetype("arial.ttf", 30)
    except:
        title_font = ImageFont.load_default()

    title = f"{class_name} - Grad-CAM and Grad-CAM++"

    bbox = draw.textbbox((0, 0), title, font=title_font)
    title_width = bbox[2] - bbox[0]

    draw.text(
        ((width - title_width) // 2, 20),
        title,
        fill="black",
        font=title_font
    )

    y = 80

    for row in rows:
        montage.paste(row, (0, y))
        y += row.height

    output = MONTAGES / (
        class_name.lower().replace(" ", "_")
        + "_explainability_montage.png"
    )

    montage.save(
        output,
        quality=95
    )

    print(f"Saved: {output}")


print("\nAll montages created successfully.")