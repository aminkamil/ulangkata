from flask import Flask, render_template, request, Response, stream_with_context, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import anthropic
import os

app = Flask(__name__)

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["30 per hour", "5 per minute"],
    storage_uri="memory://"
)

MAX_INPUT_CHARS = 8000  # ~2000 words, safe limit before token issues

PROMPTS = {
    "formal": """Anda adalah penulis akademik Bahasa Malaysia yang berpengalaman. Tugas anda adalah menulis semula teks yang diberikan dalam laras bahasa akademik formal yang tinggi, dengan frasa dan struktur ayat yang berbeza, tetapi mengekalkan makna asal sepenuhnya.

Panduan:
- Utamakan kelancaran pembacaan — teks mesti kedengaran semula jadi, bukan dipaksa
- Gunakan laras bahasa tinggi dan baku sepanjang masa
- Ubah struktur ayat: aktif-pasif, gabung atau pecah ayat panjang, susun semula urutan maklumat
- Gantikan perkataan dengan sinonim yang LAZIM dalam penulisan akademik Melayu — elakkan perkataan janggal atau jarang digunakan
- Istilah teknikal, nama, angka, dan kata pinjaman — kekalkan sahaja
- Output MESTI dalam Bahasa Malaysia sahaja
- Hanya kembalikan teks yang telah ditulis semula, tanpa sebarang ulasan atau penjelasan""",

    "semiformal": """Anda adalah penulis profesional Bahasa Malaysia. Tugas anda adalah menulis semula teks yang diberikan dalam gaya semi-formal — profesional dan jelas, tetapi lebih mudah dibaca berbanding penulisan akademik berat — sambil mengekalkan makna asal sepenuhnya.

Panduan:
- Utamakan kejelasan dan kelancaran — pembaca mesti faham dengan mudah
- Gunakan ayat yang lebih ringkas dan terus daripada teks asal jika boleh
- Gantikan frasa akademik berat dengan frasa yang lebih natural tetapi masih profesional
- Gunakan ayat aktif lebih banyak daripada pasif untuk meningkatkan kejelasan
- Elakkan jargon yang tidak perlu — jika boleh dipermudah tanpa hilang makna, permudahkan
- Istilah teknikal, nama, angka — kekalkan sahaja
- Output MESTI dalam Bahasa Malaysia sahaja
- Hanya kembalikan teks yang telah ditulis semula, tanpa sebarang ulasan atau penjelasan""",

    "auto": """Anda adalah penulis Bahasa Malaysia yang mahir. Tugas anda adalah menulis semula teks yang diberikan dengan frasa dan struktur ayat yang berbeza, tetapi mengekalkan makna asal DAN gaya penulisan asal sepenuhnya.

Panduan:
- ANALISIS teks asal terlebih dahulu: perhatikan larasnya (formal/semi-formal), panjang ayat, pilihan perkataan, dan nadanya
- Kekalkan laras, nada, dan kerumitan yang SAMA seperti teks asal — jangan naikkan atau turunkan tahap formaliti
- Ubah struktur ayat dan pilihan perkataan sahaja, bukan gaya keseluruhannya
- Gantikan perkataan dengan sinonim yang sesuai dengan tahap teks asal
- Istilah teknikal, nama, angka — kekalkan sahaja
- Output MESTI dalam Bahasa Malaysia sahaja
- Hanya kembalikan teks yang telah ditulis semula, tanpa sebarang ulasan atau penjelasan"""
}

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


@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({"error": "Terlalu banyak permintaan. Sila cuba lagi sebentar."}), 429


@app.route("/paraphrase", methods=["POST"])
@limiter.limit("5 per minute; 30 per hour")
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
    mode = data.get("mode", "formal") if data.get("mode") in PROMPTS else "formal"
    context = data.get("context", "").strip()

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
    system_prompt = PROMPTS[mode]

    if protected_words:
        protected_list = "\n".join(f"- {w}" for w in protected_words[:50])  # cap at 50
        system_prompt += PROTECTED_ADDON.format(protected_list=protected_list)

    if style_sample:
        system_prompt += STYLE_ADDON

    # Build user message
    user_message = text
    if style_sample:
        user_message = f"Contoh gaya penulisan:\n{style_sample}\n\n---\n\nTeks untuk ditulis semula:\n{text}"

    if context:
        user_message = f"[Konteks perenggan sebelumnya — untuk rujukan sahaja, JANGAN parafrasa ini]:\n{context}\n\n---\n\n{user_message}"

    def generate():
        models = ["claude-opus-4-6", "claude-sonnet-4-6"]
        for i, model in enumerate(models):
            content_yielded = False
            try:
                with client.messages.stream(
                    model=model,
                    max_tokens=4096,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_message}],
                ) as stream:
                    for chunk in stream.text_stream:
                        content_yielded = True
                        yield chunk
                return
            except anthropic.AuthenticationError:
                yield "\n\n[RALAT: API key tidak sah. Sila semak ANTHROPIC_API_KEY anda.]"
                return
            except anthropic.RateLimitError:
                yield "\n\n[RALAT: Had kadar API telah dicapai. Sila cuba sebentar lagi.]"
                return
            except anthropic.APIConnectionError:
                yield "\n\n[RALAT: Gagal menyambung ke API. Sila semak sambungan internet anda.]"
                return
            except anthropic.APIStatusError as e:
                is_overloaded = e.status_code == 529 or 'overloaded' in str(e.message).lower()
                if is_overloaded and not content_yielded and i < len(models) - 1:
                    continue  # fallback to Sonnet
                if not content_yielded:
                    yield f"\n\n[RALAT API {e.status_code}: {e.message}]"
                return
            except Exception as e:
                if not content_yielded:
                    yield f"\n\n[RALAT: {str(e)}]"
                return

    return Response(stream_with_context(generate()), mimetype="text/plain")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
