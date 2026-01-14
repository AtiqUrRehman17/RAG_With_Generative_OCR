import os
from pdf2image import convert_from_path
from langchain_core.documents import Document
from utils.image_utils import image_to_base64


def generative_ocr(bot, image, page_number):
    image_b64 = image_to_base64(image)

    ocr_prompt = (
        "Extract ONLY the text that is CLEARLY visible in this image.\n\n"
        "Rules:\n"
        "- DO NOT guess missing text\n"
        "- DO NOT infer names, dates, or addresses\n"
        "- If text is unreadable, write: [UNCLEAR]\n"
        "- If a field is empty, write: [BLANK]\n"
        "- Preserve original wording exactly\n"
    )

    response = bot.ocr_llm.invoke(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": ocr_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_b64}"
                        },
                    },
                ],
            }
        ]
    )

    text = response.content.strip()
    unclear_count = text.count("[UNCLEAR]")
    confidence = "low" if unclear_count > 8 else "high"

    return Document(
        page_content=text,
        metadata={
            "page": page_number,
            "pdf_id": bot.pdf_id,
            "source_file": os.path.basename(bot.pdf_path),
            "ocr_confidence": confidence,
        },
    )


def load_and_ocr_pdf(bot):
    print("Converting PDF to images...")
    images = convert_from_path(bot.pdf_path, dpi=300)

    documents = []
    for i, image in enumerate(images, start=1):
        print(f"Running Generative OCR on page {i}")
        documents.append(generative_ocr(bot, image, i))

    return documents
