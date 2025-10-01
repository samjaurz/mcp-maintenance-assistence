import fitz  # PyMuPDF
import os

# --------------------------------------------------------------------------------
# --- CONFIGURACIÓN CRÍTICA ---
# Estas variables globales ya NO controlan la ruta de GUARDADO,
# pero sí controlan la URL que se devuelve al frontend.
# --------------------------------------------------------------------------------
BASE_MANUALS_DIR = "/Users/sam/Desktop/github/mpc_maintenance_assistence/manuals"
SERVER_BASE_URL = "http://localhost:3000/images"


# --------------------------------------------------------------------------------

def render_page_as_image(input_pdf_path: str, page_number_1_based: int, output_dir: str, dpi: int = 300) -> str | None:
    # ... (Cuerpo de la función render_page_as_image se mantiene igual, pero usa output_dir) ...
    if not os.path.exists(input_pdf_path):
        print(f"Error: Archivo no encontrado en la ruta: {input_pdf_path}")
        return None

    # El output_dir será la ruta absoluta que definamos en get_rendered_page_url
    os.makedirs(output_dir, exist_ok=True)

    try:
        doc = fitz.open(input_pdf_path)
        page_index = page_number_1_based - 1

        if not (0 <= page_index < len(doc)):
            print(f"Error: La página {page_number_1_based} está fuera del rango (1 a {len(doc)}).")
            doc.close()
            return None

        page = doc[page_index]
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix, alpha=False)

        base_name = os.path.splitext(os.path.basename(input_pdf_path))[0]
        image_filename = f"{base_name}_page_{page_number_1_based}.png"
        output_filename = os.path.join(output_dir, image_filename)  # Usa la ruta absoluta aquí

        pix.save(output_filename)
        doc.close()

        # Devolvemos solo el nombre del archivo para construir la URL pública
        return image_filename

    except Exception as e:
        print(f"Error al intentar renderizar la página a imagen: {e}")
        return None


def get_rendered_page_url(pdf_filename: str, page_number: int) -> str | None:
    """
    Función principal que orquesta el proceso, guarda en la ruta absoluta y devuelve la URL HTTP.
    """

    # --------------------------------------------------------------------------------
    # --- RUTA DE GUARDADO ABSOLUTA Y EXPLÍCITA (Corregida) ---
    # Python intentará guardar la imagen en esta ruta de tu sistema:
    # --------------------------------------------------------------------------------
    ABSOLUTE_OUTPUT_PATH = "/Users/sam/Desktop/github/mpc_maintenance_assistence/client/public/images"

    # 1. CONSTRUCCIÓN DE LA RUTA DE ENTRADA
    input_pdf_path = os.path.join(BASE_MANUALS_DIR, pdf_filename)

    # 2. EJECUCIÓN DEL RENDERIZADO, usando la ruta ABSOLUTA
    image_filename = render_page_as_image(
        input_pdf_path=input_pdf_path,
        page_number_1_based=page_number,
        output_dir=ABSOLUTE_OUTPUT_PATH,  # <-- ¡La clave! Usamos la ruta absoluta aquí.
        dpi=300
    )

    if not image_filename:
        return None

    # 3. CONSTRUCCIÓN DE LA URL ACCESIBLE AL FRONTEND (Ruta HTTP)
    # Se debe unir la URL base (que es HTTP) con el nombre del archivo.
    final_url = f"{SERVER_BASE_URL.rstrip('/')}/{image_filename}"

    return final_url


# --- EJEMPLO DE USO DESDE EL BACKEND ---
if __name__ == "__main__":
    TEST_PDF_NAME = "mitsubishi-a800-manual.pdf"
    TEST_PAGE = 30

    url_result = get_rendered_page_url(TEST_PDF_NAME, TEST_PAGE)

    if url_result:
        # Se imprime la ruta donde se intentó guardar y la URL devuelta al frontend

        print(f"✅ URL FINAL para el Frontend: {url_result}")
    else:
        print(f"\n❌ Fallo al generar la URL.")