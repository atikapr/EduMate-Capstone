"""
EduMate - Sistem Rekomendasi Pembelajaran
Production-ready recommendation model untuk integrasi dengan REST API
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple, Optional
import logging
import os
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EduMateRecommender:
    """
    Sistem rekomendasi hybrid untuk EduMate
    Menggabungkan Content-Based, Collaborative Filtering, dan Machine Learning
    """

    def __init__(self, model_path: str = "./"):
        """
        Initialize recommender dengan loading semua model dan data

        Args:
            model_path (str): Path ke folder berisi file model .pkl
        """
        self.model_path = model_path
        self.models = {}
        self.is_loaded = False

    def load_models(self) -> bool:
        """
        Load semua model dan data yang diperlukan

        Returns:
            bool: True jika berhasil load semua model
        """
        try:
            logger.info("🔄 Loading EduMate recommendation models...")

            required_files = [
                "rf_model.pkl",
                "user_item_matrix.pkl",
                "user_similarity_df.pkl",
                "item_similarity_df.pkl",
                "label_encoders.pkl",
                "scaler_user.pkl",
                "scaler_content.pkl",
                "user_profiles.pkl",
                "content_profiles.pkl",
                "hybrid_features.pkl",
                "user_features.pkl",
                "content_features.pkl",
            ]

            for file_name in required_files:
                file_path = os.path.join(self.model_path, file_name)
                if not os.path.exists(file_path):
                    logger.error(
                        f"❌ File {file_name} tidak ditemukan di {self.model_path}"
                    )
                    return False

                with open(file_path, "rb") as f:
                    model_name = file_name.replace(".pkl", "")
                    self.models[model_name] = pickle.load(f)

            logger.info(f"✅ Successfully loaded {len(required_files)} model files")
            self.is_loaded = True
            return True

        except Exception as e:
            logger.error(f"❌ Error loading models: {str(e)}")
            return False

    def _validate_user_profile(self, user_profile: Dict) -> bool:
        """
        Validasi input user profile

        Args:
            user_profile (Dict): Profile user dengan semua field yang diperlukan

        Returns:
            bool: True jika valid
        """
        required_fields = [
            "jurusan",
            "learning_style",
            "goal",
            "ketersediaan_belajar",
            "device_preference",
            "ipk_terakhir",
            "waktu_belajar_per_hari",
        ]

        for field in required_fields:
            if field not in user_profile:
                logger.error(f"❌ Missing required field: {field}")
                return False

        return True

    def _encode_user_profile(self, user_profile: Dict) -> np.ndarray:
        """
        Encode user profile menggunakan label encoders yang sudah dilatih

        Args:
            user_profile (Dict): Raw user profile

        Returns:
            np.ndarray: Encoded user profile features
        """
        try:
            encoded_profile = {}
            label_encoders = self.models["label_encoders"]

            # Encode categorical features
            categorical_mappings = {
                "jurusan": "jurusan_encoded",
                "learning_style": "learning_style_encoded",
                "goal": "goal_encoded",
                "ketersediaan_belajar": "ketersediaan_belajar_encoded",
                "device_preference": "device_preference_encoded",
            }

            for original_field, encoded_field in categorical_mappings.items():
                if original_field in label_encoders:
                    try:
                        # Handle unseen categories
                        value = str(user_profile[original_field])
                        if value in label_encoders[original_field].classes_:
                            encoded_profile[encoded_field] = label_encoders[
                                original_field
                            ].transform([value])[0]
                        else:
                            # Use most common class for unseen categories
                            encoded_profile[encoded_field] = 0
                            logger.warning(
                                f"⚠️ Unseen category '{value}' for {original_field}, using default"
                            )
                    except Exception as e:
                        encoded_profile[encoded_field] = 0
                        logger.warning(f"⚠️ Error encoding {original_field}: {str(e)}")

            # Add numerical features
            encoded_profile["ipk_terakhir"] = float(user_profile["ipk_terakhir"])
            encoded_profile["waktu_belajar_per_hari"] = float(
                user_profile["waktu_belajar_per_hari"]
            )

            # Arrange according to user_features order
            user_features = self.models["user_features"]
            profile_array = np.array(
                [encoded_profile.get(feature, 0) for feature in user_features]
            )

            return profile_array.reshape(1, -1)

        except Exception as e:
            logger.error(f"❌ Error encoding user profile: {str(e)}")
            return np.zeros((1, len(self.models["user_features"])))

    def _content_based_recommendations(
        self, user_profile: Dict, top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Generate recommendations menggunakan content-based filtering

        Args:
            user_profile (Dict): User profile
            top_k (int): Jumlah rekomendasi yang diinginkan

        Returns:
            List[Tuple[str, float]]: List of (content_id, similarity_score)
        """
        try:
            # Encode user profile
            user_encoded = self._encode_user_profile(user_profile)

            # Scale user profile
            scaler_user = self.models["scaler_user"]
            user_scaled = scaler_user.transform(user_encoded)

            # Get content profiles
            content_profiles = self.models["content_profiles"]
            content_features = self.models["content_features"]

            # Scale content profiles
            scaler_content = self.models["scaler_content"]
            content_scaled = scaler_content.transform(
                content_profiles[content_features]
            )

            # Calculate cosine similarity
            similarities = cosine_similarity(user_scaled, content_scaled)[0]

            # Get top recommendations
            content_ids = content_profiles["id_konten"].values
            recommendations = list(zip(content_ids, similarities))
            recommendations.sort(key=lambda x: x[1], reverse=True)

            return recommendations[:top_k]

        except Exception as e:
            logger.error(f"❌ Error in content-based filtering: {str(e)}")
            return []

    def _collaborative_filtering(
        self, user_id: str, top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Generate recommendations menggunakan collaborative filtering

        Args:
            user_id (str): ID user dalam sistem
            top_k (int): Jumlah rekomendasi yang diinginkan

        Returns:
            List[Tuple[str, float]]: List of (content_id, predicted_rating)
        """
        try:
            user_item_matrix = self.models["user_item_matrix"]
            user_similarity_df = self.models["user_similarity_df"]

            # Check if user exists in matrix
            if user_id not in user_item_matrix.index:
                logger.warning(f"⚠️ User {user_id} not found in interaction history")
                return []

            # Get user's ratings
            user_ratings = user_item_matrix.loc[user_id]
            unrated_items = user_ratings[user_ratings == 0].index

            if len(unrated_items) == 0:
                logger.info(f"ℹ️ User {user_id} has rated all available content")
                return []

            # Find similar users
            similar_users = (
                user_similarity_df.loc[user_id].nlargest(11).index[1:]
            )  # Exclude self

            recommendations = []
            for item_id in unrated_items:
                # Calculate weighted average rating from similar users
                numerator = 0
                denominator = 0

                for sim_user in similar_users:
                    if user_item_matrix.loc[sim_user, item_id] > 0:
                        similarity = user_similarity_df.loc[user_id, sim_user]
                        rating = user_item_matrix.loc[sim_user, item_id]
                        numerator += similarity * rating
                        denominator += abs(similarity)

                if denominator > 0:
                    predicted_rating = numerator / denominator
                    recommendations.append((item_id, predicted_rating))

            # Sort by predicted rating
            recommendations.sort(key=lambda x: x[1], reverse=True)
            return recommendations[:top_k]

        except Exception as e:
            logger.error(f"❌ Error in collaborative filtering: {str(e)}")
            return []

    def _hybrid_ml_recommendations(
        self, user_profile: Dict, top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Generate recommendations menggunakan hybrid machine learning model

        Args:
            user_profile (Dict): User profile
            top_k (int): Jumlah rekomendasi yang diinginkan

        Returns:
            List[Tuple[str, float]]: List of (content_id, predicted_score)
        """
        try:
            rf_model = self.models["rf_model"]
            content_profiles = self.models["content_profiles"]
            hybrid_features = self.models["hybrid_features"]

            # Encode user profile
            user_encoded_dict = {}
            user_encoded = self._encode_user_profile(user_profile)[0]
            user_features = self.models["user_features"]

            for i, feature in enumerate(user_features):
                user_encoded_dict[feature] = user_encoded[i]

            recommendations = []

            # Predict score untuk setiap content
            for _, content in content_profiles.iterrows():
                # Combine user and content features
                feature_vector = {}

                # Add user features
                for feature in user_features:
                    feature_vector[feature] = user_encoded_dict.get(feature, 0)

                # Add content features
                content_features = self.models["content_features"]
                for feature in content_features:
                    if feature in content:
                        feature_vector[feature] = content[feature]
                    else:
                        feature_vector[feature] = 0

                # Add dummy interaction feature (average for new users)
                feature_vector["watch_ratio"] = 0.5  # Default assumption

                # Create feature array in correct order
                X = np.array(
                    [feature_vector.get(feature, 0) for feature in hybrid_features]
                ).reshape(1, -1)

                # Predict
                predicted_score = rf_model.predict(X)[0]
                recommendations.append((content["id_konten"], predicted_score))

            # Sort by predicted score
            recommendations.sort(key=lambda x: x[1], reverse=True)
            return recommendations[:top_k]

        except Exception as e:
            logger.error(f"❌ Error in hybrid ML recommendations: {str(e)}")
            return []

    def get_recommendations(
        self,
        user_profile: Dict,
        user_id: Optional[str] = None,
        top_k: int = 5,
        method: str = "hybrid",
    ) -> Dict:
        """
        Main function untuk mendapatkan rekomendasi

        Args:
            user_profile (Dict): Profile lengkap user
            user_id (Optional[str]): ID user jika ada (untuk collaborative filtering)
            top_k (int): Jumlah rekomendasi yang diinginkan
            method (str): 'content', 'collaborative', 'ml', atau 'hybrid'

        Returns:
            Dict: Hasil rekomendasi dengan metadata
        """
        if not self.is_loaded:
            logger.error("❌ Models not loaded. Call load_models() first.")
            return {"error": "Models not loaded"}

        if not self._validate_user_profile(user_profile):
            return {"error": "Invalid user profile"}

        start_time = datetime.now()
        result = {
            "user_profile": user_profile,
            "user_id": user_id,
            "method": method,
            "top_k": top_k,
            "recommendations": [],
            "metadata": {},
        }

        try:
            if method == "content":
                # Content-based only
                recs = self._content_based_recommendations(user_profile, top_k)
                result["recommendations"] = self._format_recommendations(
                    recs, "content_similarity"
                )

            elif method == "collaborative" and user_id:
                # Collaborative filtering only
                recs = self._collaborative_filtering(user_id, top_k)
                result["recommendations"] = self._format_recommendations(
                    recs, "collaborative_score"
                )

            elif method == "ml":
                # Machine learning only
                recs = self._hybrid_ml_recommendations(user_profile, top_k)
                result["recommendations"] = self._format_recommendations(
                    recs, "ml_score"
                )

            else:
                # Hybrid approach (default)
                content_recs = self._content_based_recommendations(
                    user_profile, top_k * 2
                )
                ml_recs = self._hybrid_ml_recommendations(user_profile, top_k * 2)

                # Combine and weight recommendations
                combined_scores = {}

                # Weight: 40% content-based, 60% ML
                for content_id, score in content_recs:
                    combined_scores[content_id] = combined_scores.get(content_id, 0) + (
                        score * 0.4
                    )

                for content_id, score in ml_recs:
                    combined_scores[content_id] = combined_scores.get(content_id, 0) + (
                        score * 0.6
                    )

                # Add collaborative if user_id available
                if user_id:
                    collab_recs = self._collaborative_filtering(user_id, top_k)
                    for content_id, score in collab_recs:
                        # Boost score if also recommended by collaborative
                        combined_scores[content_id] = combined_scores.get(
                            content_id, 0
                        ) + (score * 0.2)

                # Sort and take top_k
                final_recs = sorted(
                    combined_scores.items(), key=lambda x: x[1], reverse=True
                )[:top_k]
                result["recommendations"] = self._format_recommendations(
                    final_recs, "hybrid_score"
                )

            # Add metadata
            processing_time = (datetime.now() - start_time).total_seconds()
            result["metadata"] = {
                "processing_time_seconds": processing_time,
                "total_recommendations": len(result["recommendations"]),
                "timestamp": datetime.now().isoformat(),
            }

            logger.info(
                f"✅ Generated {len(result['recommendations'])} recommendations in {processing_time:.3f}s"
            )
            return result

        except Exception as e:
            logger.error(f"❌ Error generating recommendations: {str(e)}")
            return {"error": str(e)}

    def _format_recommendations(
        self, recommendations: List[Tuple[str, float]], score_type: str
    ) -> List[Dict]:
        """
        Format recommendations dengan informasi content yang lengkap

        Args:
            recommendations (List[Tuple[str, float]]): List of (content_id, score)
            score_type (str): Type of score untuk labeling

        Returns:
            List[Dict]: Formatted recommendations dengan content details
        """
        try:
            content_profiles = self.models["content_profiles"]
            formatted_recs = []

            for i, (content_id, score) in enumerate(recommendations, 1):
                # Find content details
                content_info = content_profiles[
                    content_profiles["id_konten"] == content_id
                ]

                if not content_info.empty:
                    content_detail = content_info.iloc[0]
                    formatted_recs.append(
                        {
                            "rank": i,
                            "content_id": content_id,
                            "title": content_detail.get("judul", "Unknown"),
                            "subject": content_detail.get("mata_kuliah", "Unknown"),
                            "platform": content_detail.get("platform", "Unknown"),
                            "format": content_detail.get("format", "Unknown"),
                            "duration": int(content_detail.get("durasi", 0)),
                            "difficulty": content_detail.get("kesulitan", "Unknown"),
                            "rating": float(content_detail.get("rating_pengguna", 0)),
                            score_type: round(float(score), 4),
                        }
                    )
                else:
                    # Fallback jika content tidak ditemukan
                    formatted_recs.append(
                        {
                            "rank": i,
                            "content_id": content_id,
                            "title": "Content Not Found",
                            "subject": "Unknown",
                            "platform": "Unknown",
                            "format": "Unknown",
                            "duration": 0,
                            "difficulty": "Unknown",
                            "rating": 0,
                            score_type: round(float(score), 4),
                        }
                    )

            return formatted_recs

        except Exception as e:
            logger.error(f"❌ Error formatting recommendations: {str(e)}")
            return []

    def retrain_model(self, new_data_path: str) -> bool:
        """
        Retrain model dengan data baru

        Args:
            new_data_path (str): Path ke folder dengan data CSV baru

        Returns:
            bool: True jika berhasil retrain
        """
        try:
            logger.info("🔄 Starting model retraining...")

            # Import training functions (you can modularize the notebook code here)
            # For now, this is a placeholder - implement training pipeline

            logger.warning(
                "⚠️ Retrain function not fully implemented. Please run notebook untuk retrain."
            )
            return False

        except Exception as e:
            logger.error(f"❌ Error retraining model: {str(e)}")
            return False

    def get_model_info(self) -> Dict:
        """
        Get informasi tentang model yang dimuat

        Returns:
            Dict: Model information
        """
        if not self.is_loaded:
            return {"error": "Models not loaded"}

        try:
            user_item_matrix = self.models["user_item_matrix"]
            content_profiles = self.models["content_profiles"]

            return {
                "model_status": "loaded",
                "total_users": len(user_item_matrix.index),
                "total_content": len(content_profiles),
                "total_interactions": user_item_matrix.values.sum(),
                "model_components": list(self.models.keys()),
                "last_loaded": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"❌ Error getting model info: {str(e)}")
            return {"error": str(e)}


# ====== UTILITY FUNCTIONS ======


def create_sample_user_profile() -> Dict:
    """
    Create sample user profile untuk testing

    Returns:
        Dict: Sample user profile
    """
    return {
        "jurusan": "Informatika",
        "learning_style": "Visual",
        "goal": "Pahami Materi",
        "ketersediaan_belajar": "Malam",
        "device_preference": "Laptop",
        "ipk_terakhir": 3.5,
        "waktu_belajar_per_hari": 3,
    }


def validate_api_input(data: Dict) -> Tuple[bool, str]:
    """
    Validate input dari API request

    Args:
        data (Dict): Request data dari API

    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    required_fields = ["user_profile"]

    for field in required_fields:
        if field not in data:
            return False, f"Missing required field: {field}"

    user_profile = data["user_profile"]
    required_profile_fields = [
        "jurusan",
        "learning_style",
        "goal",
        "ketersediaan_belajar",
        "device_preference",
        "ipk_terakhir",
        "waktu_belajar_per_hari",
    ]

    for field in required_profile_fields:
        if field not in user_profile:
            return False, f"Missing user profile field: {field}"

    # Validate data types
    try:
        float(user_profile["ipk_terakhir"])
        int(user_profile["waktu_belajar_per_hari"])
    except (ValueError, TypeError):
        return False, "Invalid data type for numeric fields"

    return True, ""


# ====== MAIN EXECUTION & TESTING ======

if __name__ == "__main__":
    """
    Testing dan demo penggunaan EduMateRecommender
    """
    print("🚀 EduMate Recommendation System - Testing")

    # Initialize recommender
    recommender = EduMateRecommender(model_path="./")

    # Load models
    if not recommender.load_models():
        print("❌ Failed to load models. Make sure all .pkl files are available.")
        exit(1)

    # Get model info
    model_info = recommender.get_model_info()
    print(f"📊 Model Info: {model_info}")

    # Create sample user
    sample_user = create_sample_user_profile()
    print(f"👤 Sample User Profile: {sample_user}")

    # Test different recommendation methods
    methods = ["content", "ml", "hybrid"]

    for method in methods:
        print(f"\n🎯 Testing {method.upper()} method:")

        recommendations = recommender.get_recommendations(
            user_profile=sample_user, user_id=None, top_k=5, method=method  # New user
        )

        if "error" in recommendations:
            print(f"❌ Error: {recommendations['error']}")
            continue

        print(
            f"⏱️ Processing time: {recommendations['metadata']['processing_time_seconds']:.3f}s"
        )
        print("📝 Recommendations:")

        for rec in recommendations["recommendations"]:
            score_key = [k for k in rec.keys() if "score" in k or "similarity" in k][0]
            print(f"   {rec['rank']}. {rec['title'][:40]}...")
            print(
                f"      📚 {rec['subject']} | 🎯 {rec['difficulty']} | ⭐ {rec['rating']:.2f} | 📊 {rec[score_key]:.3f}"
            )

    # Test dengan user yang ada di system (collaborative filtering)
    print(f"\n👥 Testing COLLABORATIVE method:")

    # Ambil user ID yang ada di system
    user_item_matrix = recommender.models["user_item_matrix"]
    if len(user_item_matrix.index) > 0:
        existing_user_id = user_item_matrix.index[0]

        collab_recommendations = recommender.get_recommendations(
            user_profile=sample_user,
            user_id=existing_user_id,
            top_k=5,
            method="collaborative",
        )

        if "error" not in collab_recommendations:
            print(
                f"⏱️ Processing time: {collab_recommendations['metadata']['processing_time_seconds']:.3f}s"
            )
            print("📝 Collaborative Recommendations:")

            for rec in collab_recommendations["recommendations"]:
                print(f"   {rec['rank']}. {rec['title'][:40]}...")
                print(
                    f"      📚 {rec['subject']} | 🎯 {rec['difficulty']} | ⭐ {rec['rating']:.2f} | 📊 {rec['collaborative_score']:.3f}"
                )
        else:
            print(
                f"❌ Collaborative filtering error: {collab_recommendations['error']}"
            )

    print("\n✅ Testing completed!")
    print("\n📋 Usage untuk Flask/FastAPI:")
    print(
        """
    # Initialize
    recommender = EduMateRecommender('./models/')
    recommender.load_models()
    
    # API endpoint example
    @app.post('/recommend')
    def get_recommendations(request_data: dict):
        is_valid, error_msg = validate_api_input(request_data)
        if not is_valid:
            return {'error': error_msg}
            
        recommendations = recommender.get_recommendations(
            user_profile=request_data['user_profile'],
            user_id=request_data.get('user_id'),
            top_k=request_data.get('top_k', 5),
            method=request_data.get('method', 'hybrid')
        )
        
        return recommendations
    """
    )
