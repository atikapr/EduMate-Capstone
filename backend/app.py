"""
EduMate Flask API - Integrasi sistem rekomendasi dengan REST API
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import logging
from datetime import datetime
import threading  # Tambahkan ini

# Import EduMate recommender dari modul rekomendasi_model.py
from rekomendasi_model import (
    EduMateRecommender,
    validate_api_input,
    create_sample_user_profile,
)

# Setup Flask app
app = Flask(__name__)
CORS(
    app
)  # Enable CORS untuk frontend integration, sesuaikan dengan kebutuhan produksi Anda

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global recommender instance
recommender = None
# Gunakan flag untuk memastikan inisialisasi hanya berjalan sekali
_recommender_initialized = threading.Event()


def initialize_recommender():
    """Initialize dan load model saat startup aplikasi Flask."""
    global recommender
    if _recommender_initialized.is_set():
        logger.info("Recommender sudah diinisialisasi sebelumnya, melewati.")
        return True

    try:
        # Tentukan path model. Gunakan variabel lingkungan MODEL_PATH jika ada,
        # jika tidak, default ke './models/'.
        # PASTIKAN SEMUA FILE .pkl DARI NOTEBOOK ADA DI DALAM FOLDER INI.
        model_path = os.getenv("MODEL_PATH", "./models/")
        logger.info(f"Mencoba memuat model dari: {model_path}")

        recommender = EduMateRecommender(model_path)

        if recommender.load_models():
            logger.info("✅ EduMate recommender initialized successfully")
            _recommender_initialized.set()  # Set flag bahwa inisialisasi berhasil
            return True
        else:
            logger.error(
                f"❌ Gagal memuat model rekomendasi dari {model_path}. Pastikan semua file .pkl tersedia."
            )
            return False
    except Exception as e:
        logger.error(f"❌ Error saat inisialisasi recommender: {str(e)}")
        return False


@app.before_request
def startup_load_models_once():
    """
    Fungsi ini akan dijalankan sekali sebelum request pertama untuk memuat model.
    Menggantikan @app.before_first_request yang sudah deprecated di Flask 2.3+.
    """
    if not _recommender_initialized.is_set():
        logger.info("Aplikasi Flask sedang startup, memuat model untuk pertama kali...")
        if not initialize_recommender():
            logger.critical(
                "❌ Startup gagal - recommender tidak dapat diinisialisasi. Aplikasi mungkin tidak berfungsi dengan baik."
            )


@app.route("/health", methods=["GET"])
def health_check():
    """
    Endpoint untuk health check.
    Mengembalikan status sehat jika recommender berhasil dimuat.
    """
    if recommender and recommender.is_loaded:
        return (
            jsonify(
                {
                    "status": "healthy",
                    "service": "EduMate Recommendation API",
                    "model_status": recommender.get_model_info().get(
                        "model_status", "unknown"
                    ),
                    "timestamp": datetime.now().isoformat(),
                }
            ),
            200,
        )
    else:
        logger.warning("Health check: Recommender belum dimuat atau gagal.")
        return (
            jsonify(
                {
                    "status": "unhealthy",
                    "service": "EduMate Recommendation API",
                    "error": "Recommender not loaded",
                    "timestamp": datetime.now().isoformat(),
                }
            ),
            503,
        )


@app.route("/model/info", methods=["GET"])
def get_model_info_endpoint():
    """
    Endpoint untuk mendapatkan informasi detail tentang model yang dimuat.
    """
    if not recommender or not recommender.is_loaded:
        return (
            jsonify({"error": "Recommender belum diinisialisasi atau gagal dimuat"}),
            503,
        )

    model_info = recommender.get_model_info()
    return jsonify(model_info), 200


@app.route("/recommend", methods=["POST"])
def get_recommendations_endpoint():
    """
    Main endpoint untuk mendapatkan rekomendasi pembelajaran.
    Menerima profil pengguna, dan opsional user_id, top_k, serta metode rekomendasi.
    """
    try:
        if not recommender or not recommender.is_loaded:
            logger.error(
                "Permintaan rekomendasi ditolak: Recommender belum diinisialisasi."
            )
            return (
                jsonify(
                    {"error": "Sistem rekomendasi belum siap. Silakan coba lagi nanti."}
                ),
                503,
            )

        data = request.get_json()

        if not data:
            logger.warning("Permintaan rekomendasi tanpa data JSON.")
            return (
                jsonify(
                    {"error": "Tidak ada data JSON yang disediakan dalam permintaan."}
                ),
                400,
            )

        # Validasi input menggunakan fungsi dari rekomendasi_model.py
        is_valid, error_msg = validate_api_input(data)
        if not is_valid:
            logger.warning(f"Validasi input gagal: {error_msg}")
            return jsonify({"error": error_msg}), 400

        # Ekstrak parameter dari payload JSON
        user_profile = data.get("user_profile")
        user_id = data.get("user_id")  # Opsional untuk Collaborative Filtering
        top_k = data.get("top_k", 5)  # Default 5
        method = data.get("method", "hybrid")  # Default 'hybrid'

        # Validasi top_k
        if not isinstance(top_k, int) or top_k < 1 or top_k > 20:
            logger.warning(f"top_k tidak valid: {top_k}. Harus antara 1 dan 20.")
            return (
                jsonify(
                    {
                        "error": "Parameter 'top_k' harus berupa bilangan bulat antara 1 dan 20."
                    }
                ),
                400,
            )

        # Validasi method
        valid_methods = ["content", "collaborative", "ml", "hybrid"]
        if method not in valid_methods:
            logger.warning(
                f"Metode rekomendasi tidak valid: {method}. Harus salah satu dari {valid_methods}."
            )
            return (
                jsonify(
                    {
                        "error": f"Metode rekomendasi tidak valid. Pilihan yang tersedia: {', '.join(valid_methods)}."
                    }
                ),
                400,
            )

        logger.info(
            f"Menerima permintaan rekomendasi untuk user_id: {user_id}, method: {method}, top_k: {top_k}"
        )

        # Panggil fungsi get_recommendations dari instance recommender
        recommendations = recommender.get_recommendations(
            user_profile=user_profile, user_id=user_id, top_k=top_k, method=method
        )

        if "error" in recommendations:
            logger.error(
                f"Error saat menghasilkan rekomendasi: {recommendations['error']}"
            )
            return (
                jsonify(
                    {
                        "message": "Gagal menghasilkan rekomendasi",
                        "details": recommendations["error"],
                    }
                ),
                500,
            )

        logger.info(
            f"Rekomendasi berhasil dihasilkan. Jumlah: {len(recommendations['recommendations'])}"
        )
        return jsonify(recommendations), 200

    except Exception as e:
        logger.exception("Terjadi error tak terduga pada endpoint /recommend:")
        return (
            jsonify({"error": "Terjadi kesalahan internal server.", "details": str(e)}),
            500,
        )


@app.route("/recommend/batch", methods=["POST"])
def get_batch_recommendations_endpoint():
    """
    Endpoint untuk mendapatkan rekomendasi secara batch untuk beberapa pengguna.
    Menerima list objek permintaan rekomendasi.
    """
    try:
        if not recommender or not recommender.is_loaded:
            logger.error(
                "Permintaan batch rekomendasi ditolak: Recommender belum diinisialisasi."
            )
            return (
                jsonify(
                    {"error": "Sistem rekomendasi belum siap. Silakan coba lagi nanti."}
                ),
                503,
            )

        data = request.get_json()
        if not data or "requests" not in data:
            logger.warning("Permintaan batch tanpa array 'requests' yang valid.")
            return (
                jsonify(
                    {
                        "error": "Payload JSON harus berisi kunci 'requests' berupa array."
                    }
                ),
                400,
            )

        requests_list = data["requests"]
        if not isinstance(requests_list, list) or len(requests_list) == 0:
            logger.warning("Array 'requests' kosong atau tidak dalam format list.")
            return (
                jsonify(
                    {
                        "error": "Kunci 'requests' harus berisi daftar permintaan rekomendasi yang tidak kosong."
                    }
                ),
                400,
            )

        if len(requests_list) > 10:  # Batasi ukuran batch untuk mencegah overload
            logger.warning(
                f"Ukuran batch melebihi batas (10). Diterima: {len(requests_list)}."
            )
            return (
                jsonify({"error": "Maksimum 10 permintaan per batch diizinkan."}),
                400,
            )

        results = []
        for i, req in enumerate(requests_list):
            try:
                # Validasi setiap permintaan dalam batch
                is_valid, error_msg = validate_api_input(req)
                if not is_valid:
                    results.append(
                        {"request_index": i, "status": "failed", "error": error_msg}
                    )
                    continue

                # Ekstrak parameter untuk setiap permintaan
                user_profile = req.get("user_profile")
                user_id = req.get("user_id")
                top_k = req.get("top_k", 5)
                method = req.get("method", "hybrid")

                # Tambahan validasi top_k dan method untuk setiap request di batch
                if not isinstance(top_k, int) or top_k < 1 or top_k > 20:
                    results.append(
                        {
                            "request_index": i,
                            "status": "failed",
                            "error": "Parameter 'top_k' harus berupa bilangan bulat antara 1 dan 20.",
                        }
                    )
                    continue
                valid_methods = ["content", "collaborative", "ml", "hybrid"]
                if method not in valid_methods:
                    results.append(
                        {
                            "request_index": i,
                            "status": "failed",
                            "error": f"Metode rekomendasi tidak valid. Pilihan yang tersedia: {', '.join(valid_methods)}.",
                        }
                    )
                    continue

                # Panggil fungsi rekomendasi
                recommendations = recommender.get_recommendations(
                    user_profile=user_profile,
                    user_id=user_id,
                    top_k=top_k,
                    method=method,
                )

                results.append(recommendations)

            except Exception as e:
                logger.error(f"Error memproses permintaan batch ke-{i}: {str(e)}")
                results.append(
                    {
                        "request_index": i,
                        "status": "failed",
                        "error": f"Kesalahan internal saat memproses permintaan ini: {str(e)}",
                    }
                )

        return jsonify({"results": results}), 200

    except Exception as e:
        logger.exception("Terjadi error tak terduga pada endpoint /recommend/batch:")
        return (
            jsonify(
                {
                    "error": "Terjadi kesalahan internal server saat memproses batch.",
                    "details": str(e),
                }
            ),
            500,
        )


@app.route("/user/profile/sample", methods=["GET"])
def get_sample_profile_endpoint():
    """
    Endpoint untuk mendapatkan contoh profil pengguna. Berguna untuk pengujian.
    """
    sample_profile = create_sample_user_profile()
    return (
        jsonify(
            {
                "sample_user_profile": sample_profile,
                "description": "Contoh profil pengguna untuk pengujian sistem rekomendasi.",
            }
        ),
        200,
    )


@app.route("/content/subjects", methods=["GET"])
def get_available_subjects_endpoint():
    """
    Endpoint untuk mendapatkan daftar mata kuliah yang tersedia dalam data konten.
    """
    try:
        if not recommender or not recommender.is_loaded:
            return (
                jsonify(
                    {"error": "Recommender belum diinisialisasi atau gagal dimuat"}
                ),
                503,
            )

        # Akses content_profiles dari instance recommender
        content_profiles = recommender.models["content_profiles"]
        subjects = content_profiles["mata_kuliah"].unique().tolist()

        return jsonify({"subjects": subjects, "total_subjects": len(subjects)}), 200

    except Exception as e:
        logger.exception("Error saat mendapatkan daftar mata kuliah:")
        return (
            jsonify({"error": "Terjadi kesalahan internal server.", "details": str(e)}),
            500,
        )


@app.route("/stats", methods=["GET"])
def get_system_stats_endpoint():
    """
    Endpoint untuk mendapatkan statistik umum tentang sistem rekomendasi dan data konten.
    """
    try:
        if not recommender or not recommender.is_loaded:
            return (
                jsonify(
                    {"error": "Recommender belum diinisialisasi atau gagal dimuat"}
                ),
                503,
            )

        model_info = recommender.get_model_info()

        # Tambahan statistik dari content_profiles
        content_profiles = recommender.models["content_profiles"]
        stats = {
            **model_info,  # Gabungkan info model dasar
            "content_by_difficulty": content_profiles["kesulitan"]
            .value_counts()
            .to_dict(),
            "content_by_format": content_profiles["format"].value_counts().to_dict(),
            "content_by_platform": content_profiles["platform"]
            .value_counts()
            .to_dict(),
            "average_content_rating": content_profiles["rating_pengguna"].mean(),
            "average_content_duration": content_profiles["durasi"].mean(),
        }

        return jsonify(stats), 200

    except Exception as e:
        logger.exception("Error saat mendapatkan statistik sistem:")
        return (
            jsonify({"error": "Terjadi kesalahan internal server.", "details": str(e)}),
            500,
        )


# Penanganan error global untuk respons JSON yang konsisten
@app.errorhandler(404)
def not_found_error(error):
    logger.warning(f"404 Not Found: {request.url}")
    return (
        jsonify(
            {
                "error": "Endpoint tidak ditemukan. Pastikan URL dan metode permintaan sudah benar."
            }
        ),
        404,
    )


@app.errorhandler(405)
def method_not_allowed_error(error):
    logger.warning(f"405 Method Not Allowed: {request.method} {request.url}")
    return (
        jsonify({"error": "Metode permintaan tidak diizinkan untuk endpoint ini."}),
        405,
    )


@app.errorhandler(500)
def internal_server_error(error):
    logger.exception("Terjadi kesalahan internal server yang tidak tertangani:")
    return (
        jsonify(
            {
                "error": "Terjadi kesalahan internal server. Silakan coba lagi nanti atau hubungi administrator."
            }
        ),
        500,
    )


if __name__ == "__main__":
    # Pastikan direktori 'models' ada jika Anda menyimpannya di sana.
    # Anda perlu menjalankan edumate_ml_notebook.ipynb terlebih dahulu
    # untuk membuat file-file model .pkl
    if not os.path.exists("./models"):
        os.makedirs("./models")
        logger.warning(
            "Direktori './models/' tidak ditemukan. Pastikan Anda telah menjalankan notebook pelatihan dan menyimpan model di sini."
        )

    # Jalankan server Flask
    # Gunakan variabel lingkungan untuk konfigurasi yang fleksibel
    # HOST: default ke '0.0.0.0' (dapat diakses dari luar localhost)
    # PORT: default ke 5000
    # DEBUG: default ke False. Set ke 'True' untuk development (memberikan traceback detail)
    app.run(
        host=os.getenv("FLASK_HOST", "0.0.0.0"),
        port=int(os.getenv("FLASK_PORT", 5000)),
        debug=os.getenv("FLASK_DEBUG", "False").lower() == "true",
    )
