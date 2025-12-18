def test_upload_pdf_pages(client, real_pdf):
    files = {"file": ("variador_frecuencia_fr-f800.pdf", real_pdf, "application/pdf")}

    response = client.post("/manuals/upload", files=files)
    text = response.json()
    assert response.status_code == 200



if __name__ == "__main__":
    test_upload_pdf_pages()
