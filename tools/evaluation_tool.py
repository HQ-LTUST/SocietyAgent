import logging
import numpy as np
from typing import List, Dict, Union
from nltk.sentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from scipy.spatial import distance
import torch
import nltk
from dataclasses import dataclass

# 配置日志
logger = logging.getLogger("websocietysimulator")


@dataclass
class EvaluationMetrics:
    """存储评估指标的数据类"""
    preference_estimation: float  # 偏好估计（与第一版保持一致）
    review_generation: float  # 评论生成总分
    overall_quality: float  # 整体质量
    sentiment_error: float  # 情感分析误差
    emotion_error: float  # 情绪分类误差
    topic_error: float  # 主题相似度误差


class UserModelingEvaluator:
    def __init__(self, device: str = "auto"):
        """初始化评估器
        Args:
            device: 使用的设备，可选 "auto", "cpu", "gpu"
        """
        self.device = self._get_device(device)
        st_device = "cuda" if self.device == 0 else "cpu"
        pipeline_device = self.device

        # 初始化评估模型（与第一版保持一致）
        self.sia = SentimentIntensityAnalyzer()
        self.emotion_classifier = pipeline(
            "text-classification",
            model="cardiffnlp/twitter-roberta-base-emotion",
            top_k=5,
            device=pipeline_device
        )
        self.topic_model = SentenceTransformer('paraphrase-MiniLM-L6-v2', device=st_device)

        # 确保下载必要的NLTK数据
        try:
            nltk.data.find('sentiment/vader_lexicon.zip')
        except LookupError:
            nltk.download('vader_lexicon', quiet=True)

    def _get_device(self, device: str) -> int:
        """确定运行设备"""
        if device == "gpu":
            return 0 if torch.cuda.is_available() else -1
        elif device == "cpu":
            return -1
        elif device == "auto":
            return 0 if torch.cuda.is_available() else -1
        else:
            raise ValueError("Device must be 'cpu', 'gpu' or 'auto'")

    def calculate_preference_estimation(
            self,
            simulated_ratings: List[float],
            real_ratings: List[float]
    ) -> float:
        """计算偏好估计（与第一版保持一致）"""
        star_error = 0
        for sim_star, real_star in zip(simulated_ratings, real_ratings):
            # 确保评分在0-5范围内
            sim_star = np.clip(sim_star, 0, 5)
            star_error += abs(sim_star - real_star) / 5

        star_error = star_error / len(real_ratings)
        return 1 - star_error

    def _calculate_emotion_error(
            self,
            emotions1: List[Dict],
            emotions2: List[Dict]
    ) -> float:
        """计算情绪分布的相似度（与第一版保持一致）"""
        emotion_dict1 = {e['label']: e['score'] for e in emotions1}
        emotion_dict2 = {e['label']: e['score'] for e in emotions2}

        all_emotions = set(emotion_dict1.keys()) | set(emotion_dict2.keys())

        vec1 = np.array([emotion_dict1.get(e, 0) for e in all_emotions])
        vec2 = np.array([emotion_dict2.get(e, 0) for e in all_emotions])

        return float(np.mean(np.abs(vec1 - vec2)))

    def calculate_review_metrics(
            self,
            simulated_reviews: List[str],
            real_reviews: List[str]
    ) -> Dict[str, float]:
        """计算评论指标（与第一版保持一致）"""
        sentiment_error = []
        emotion_error = []
        topic_error = []

        # 处理情绪分析（截断过长的评论）
        truncated_sim_reviews = [
            review[:300] if len(review) > 300 else review
            for review in simulated_reviews
        ]
        truncated_real_reviews = [
            review[:300] if len(review) > 300 else review
            for review in real_reviews
        ]

        simulated_emotions = self.emotion_classifier(truncated_sim_reviews)
        real_emotions = self.emotion_classifier(truncated_real_reviews)

        for i, (sim_review, real_review) in enumerate(zip(simulated_reviews, real_reviews)):
            # 情感分析
            sentiment1 = self.sia.polarity_scores(sim_review)['compound']
            sentiment2 = self.sia.polarity_scores(real_review)['compound']
            sentiment_error.append(abs(sentiment1 - sentiment2) / 2)

            # 主题分析
            embeddings = self.topic_model.encode([sim_review, real_review])
            topic_error.append(distance.cosine(embeddings[0], embeddings[1]) / 2)

            # 情绪分析
            emotion_error_single = self._calculate_emotion_error(
                simulated_emotions[i],
                real_emotions[i]
            )
            emotion_error.append(emotion_error_single)

        return {
            'sentiment_error': float(np.mean(sentiment_error)),
            'emotion_error': float(np.mean(emotion_error)),
            'topic_error': float(np.mean(topic_error))
        }

    def evaluate(
            self,
            simulated_data: List[Dict],
            real_data: List[Dict]
    ) -> EvaluationMetrics:
        """评估整体性能"""
        try:
            # 提取评分和评论
            simulated_ratings = [item['stars'] for item in simulated_data]
            real_ratings = [item['stars'] for item in real_data]
            simulated_reviews = [item['review'] for item in simulated_data]
            real_reviews = [item['review'] for item in real_data]

            # 1. 计算偏好估计
            preference_estimation = self.calculate_preference_estimation(
                simulated_ratings,
                real_ratings
            )

            # 2. 计算评论相关指标
            review_metrics = self.calculate_review_metrics(
                simulated_reviews,
                real_reviews
            )

            # 3. 计算评论生成得分（与第一版保持一致）
            review_generation = 1 - (
                    review_metrics['sentiment_error'] * 0.25 +
                    review_metrics['emotion_error'] * 0.25 +
                    review_metrics['topic_error'] * 0.5
            )

            # 4. 计算整体质量分数
            overall_quality = (preference_estimation + review_generation) / 2

            return EvaluationMetrics(
                preference_estimation=preference_estimation,
                review_generation=review_generation,
                overall_quality=overall_quality,
                sentiment_error=review_metrics['sentiment_error'],
                emotion_error=review_metrics['emotion_error'],
                topic_error=review_metrics['topic_error']
            )

        except Exception as e:
            logger.error(f"Evaluation error: {str(e)}")
            raise


# 使用示例
if __name__ == "__main__":
    evaluator = UserModelingEvaluator()
    # 示例数据
    simulated_data = [
        {"stars": 4.0, "review": "Great product, really enjoyed it!"},
        {"stars": 3.5, "review": "Pretty good but could be better."}
    ]
    real_data = [
        {"stars": 4.5, "review": "Excellent product, highly recommend!"},
        {"stars": 3.0, "review": "It's okay, not the best."}
    ]

    metrics = evaluator.evaluate(simulated_data, real_data)
    print(f"Preference Estimation: {metrics.preference_estimation:.4f}")
    print(f"Review Generation Score: {metrics.review_generation:.4f}")
    print(f"Overall Quality: {metrics.overall_quality:.4f}")
    print(f"Sentiment Error: {metrics.sentiment_error:.4f}")
    print(f"Emotion Error: {metrics.emotion_error:.4f}")
    print(f"Topic Error: {metrics.topic_error:.4f}")