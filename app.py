from flask import Flask, render_template, request, Response, stream_with_context, jsonify
import anthropic
import os

app = Flask(__name__)

MAX_INPUT_CHARS = 8000  # ~2000 words, safe limit before token issues

BASE_SYSTEM_PROMPT = """Anda adalah pakar bahasa Melayu akademik. Tugas anda adalah untuk menulis semula teks akademik Bahasa Malaysia yang diberikan dengan frasa dan struktur ayat yang berbeza, sambil mengekalkan makna asal sepenuhnya.

Panduan:
- Gunakan sinonim yang sesuai untuk perkataan biasa
- Susun semula struktur ayat tanpa mengubah maksud
- Kekalkan laras bahasa akademik dan formal
- Pastikan teks masih kedengaran semula jadi dalam Bahasa Malaysia
- Jangan terjemahkan ke bahasa lain — output MESTI dalam Bahasa Malaysia
- Hanya kembalikan teks yang telah ditulis semula, tanpa sebarang ulasan atau penjelasan"""

STYLE_ADDON = """
Selain itu, anda telah diberikan contoh gaya penulisan seseorang. Tiru gaya penulisan tersebut — termasuk cara pembinaan ayat, pilihan perkataan, panjang ayat, dan nada penulisan — tetapi JANGAN ubah maksud teks asal."""

PROTECTED_ADDON = """
PENTING — Perkataan dan frasa berikut MESTI dikekalkan SAMA PERSIS seperti asal, jangan ubah langsung:
{protected_list}"""


def get_client():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None, "ANTHROPIC_API_KEY tidak dijumpai. Sila tetapkan pemboleh ubah persekitaran."
    try:
        return anthropic.Anthropic(api_key=api_key), None
    except Exception as e:
        return None, str(e)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    _, err = get_client()
    if err:
        return jsonify({"status": "error", "message": err}), 500
    return jsonify({"status": "ok"})


@app.route("/paraphrase", methods=["POST"])
def paraphrase():
    # Validate request body
    if not request.is_json:
        return jsonify({"error": "Permintaan tidak sah."}), 400

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Data tidak diterima."}), 400

    text = data.get("text", "").strip()
    style_sample = data.get("style_sample", "").strip()
    protected_words = data.get("protected_words", [])

    # Validate input
    if not text:
        return jsonify({"error": "Teks kosong. Sila masukkan teks untuk diparafrasa."}), 400

    if len(text) > MAX_INPUT_CHARS:
        return jsonify({"error": f"Teks terlalu panjang ({len(text)} aksara). Had maksimum ialah {MAX_INPUT_CHARS} aksara (~2000 patah perkataan). Sila bahagikan teks kepada bahagian yang lebih kecil."}), 400

    # Get client
    client, err = get_client()
    if err:
        return jsonify({"error": err}), 500

    # Build system prompt
    system_prompt = BASE_SYSTEM_PROMPT

    if protected_words:
        protected_list = "\n".join(f"- {w}" for w in protected_words[:50])  # cap at 50
        system_prompt += PROTECTED_ADDON.format(protected_list=protected_list)

    if style_sample:
        system_prompt += STYLE_ADDON

    # Build user message
    user_message = text
    if style_sample:
        user_message = f"Contoh gaya penulisan:\n{style_sample}\n\n---\n\nTeks untuk ditulis semula:\n{text}"

    def generate():
        try:
            with client.messages.stream(
                model="claude-opus-4-6",
                max_tokens=4096,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            ) as stream:
                for chunk in stream.text_stream:
                    yield chunk
        except anthropic.AuthenticationError:
            yield "\n\n[RALAT: API key tidak sah. Sila semak ANTHROPIC_API_KEY anda.]"
        except anthropic.RateLimitError:
            yield "\n\n[RALAT: Had kadar API telah dicapai. Sila cuba sebentar lagi.]"
        except anthropic.APIConnectionError:
            yield "\n\n[RALAT: Gagal menyambung ke API. Sila semak sambungan internet anda.]"
        except anthropic.APIStatusError as e:
            yield f"\n\n[RALAT API {e.status_code}: {e.message}]"
        except Exception as e:
            yield f"\n\n[RALAT: {str(e)}]"

    return Response(stream_with_context(generate()), mimetype="text/plain")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
