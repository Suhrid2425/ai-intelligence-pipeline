from google import genai
from src.config import get_settings


def test_gemini_connection():
    settings = get_settings()

    assert settings.gemini_api_key, "Gemini API key is missing"

    client = genai.Client(api_key=settings.gemini_api_key)

    response = client.models.generate_content(
        model=settings.gemini_model,
        contents="Reply with exactly: GEMINI_CONNECTION_OK"
    )

    print("\nGemini response:", response.text)

    assert response.text is not None