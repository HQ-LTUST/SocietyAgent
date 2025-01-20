from openai import OpenAI
from websocietysimulator.agent import SimulationAgent
from websocietysimulator.llm import LLMBase
from websocietysimulator.agent.modules.planning_modules import PlanningBase
from websocietysimulator.agent.modules.reasoning_modules import ReasoningBase
from websocietysimulator.agent.modules.memory_modules import MemoryDILU, MemoryGenerative, MemoryTP, MemoryVoyager
from websocietysimulator.tools.interaction_tool import InteractionTool
from tools.evaluation_tool import SimulationEvaluator  # 更新导入路径更改为新的导入路径
from typing import Dict, List, Any, Optional, Union, Tuple
from tools.cache_interaction_tool import CacheInteractionTool  # 假设文件名是 cache_interaction_tool.py
import logging
import os
import json
import numpy as np
from sklearn import neighbors, svm
from textblob import TextBlob
from nltk import sent_tokenize, word_tokenize
import textstat
import re
from collections import Counter, defaultdict
import logging
from collections import defaultdict, deque
from sklearn.preprocessing import StandardScaler
STOP_WORDS = {'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
              'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
              'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her',
              'she', 'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there',
              'their', 'what', 'so', 'up', 'out', 'if', 'about', 'who', 'get',
              'which', 'go', 'me', 'when', 'make', 'can', 'like', 'time', 'no',
              'just', 'him', 'know', 'take', 'people', 'into', 'year', 'your',
              'good', 'some', 'could', 'them', 'see', 'other', 'than', 'then',
              'now', 'look', 'only', 'come', 'its', 'over', 'think', 'also',
              'back', 'after', 'use', 'two', 'how', 'our', 'work', 'first',
              'well', 'way', 'even', 'new', 'want', 'because', 'any', 'these',
              'give', 'day', 'most', 'us'}

logger = logging.getLogger(__name__)

POSITIVE_WORDS = {'great', 'good', 'excellent', 'amazing', 'wonderful', 'fantastic'}
NEGATIVE_WORDS = {'bad', 'poor', 'terrible', 'horrible', 'awful', 'disappointing'}

class OpenAIWrapper(LLMBase):
    """OpenAI wrapper that inherits from LLMBase"""

    def __init__(self, api_key: str, base_url: str, model: str = "gpt-3.5-turbo"):
        """Initialize OpenAI wrapper"""
        super().__init__(model=model)
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.api_key = api_key
        self.base_url = base_url
        self._embedding_model = None

    def __call__(self, messages: List[Dict[str, str]],
                 model: Optional[str] = None,
                 temperature: float = 0.0,
                 max_tokens: int = 500,
                 stop_strs: Optional[List[str]] = None,
                 n: int = 1) -> Union[str, List[str]]:
        """Call OpenAI API to get response"""
        try:
            response = self.client.chat.completions.create(
                model=model or self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop_strs,
                n=n
            )

            if n == 1:
                return response.choices[0].message.content
            else:
                return [choice.message.content for choice in response.choices]
        except Exception as e:
            logger.error(f"Error in OpenAI API call: {e}")
            raise

    def get_embedding_model(self):
        """Get the embedding model"""
        if self._embedding_model is None:
            self._embedding_model = OpenAIEmbeddings(
                api_key=self.api_key,
                base_url=self.base_url
            )
        return self._embedding_model


class OpenAIEmbeddings:
    """OpenAI Embeddings to replace InfinigenceEmbeddings"""

    def __init__(self, api_key: str, base_url: str, model: str = "text-embedding-ada-002"):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model = model

    def get_embeddings(self, text: str) -> List[float]:
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error in OpenAI Embeddings API call: {e}")
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """添加兼容原始接口的方法"""
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=texts
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"Error in batch embedding: {e}")
            raise

    def embed_query(self, text: str) -> List[float]:
        """添加用于查询的嵌入方法"""
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error in query embedding: {e}")
            raise


from sklearn.ensemble import IsolationForest
import numpy as np


class MemoryManager:
    """Memory module manager with ML preprocessing"""

    def __init__(self, llm):
        """
        初始化 MemoryManager
        Args:
            llm: 语言模型实例
        """
        self.llm = llm
        # 初始化所有记忆模块
        self.memories = {
            'dilu': MemoryDILU(llm=llm),
            'generative': MemoryGenerative(llm=llm),
            'tp': MemoryTP(llm=llm),
            'voyager': MemoryVoyager(llm=llm)
        }
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.review_stats = []
        self.is_trained = False
        self.min_training_samples = 3  # 最小训练样本数

    def preprocess_review(self, review_data):
        """
        预处理评论数据，提取特征
        Args:
            review_data: 评论数据字典
        Returns:
            numpy.array: 处理后的特征数组
        """
        try:
            # 提取数值特征
            features = [
                float(review_data.get('stars', 3.0)),
                int(review_data.get('useful', 0)),
                int(review_data.get('funny', 0)),
                int(review_data.get('cool', 0)),
                len(str(review_data.get('text', '')))
            ]
            return np.array(features).reshape(1, -1)
        except Exception as e:
            logger.error(f"Error in preprocess_review: {e}")
            # 返回默认特征
            return np.array([3.0, 0, 0, 0, 0]).reshape(1, -1)

    def train_anomaly_detector(self, review_data_list):
        """
        训练异常检测器
        Args:
            review_data_list: 评论数据列表
        """
        try:
            if len(review_data_list) < self.min_training_samples:
                logger.warning(f"Insufficient training data: {len(review_data_list)} samples")
                return False

            features_list = []
            for review in review_data_list:
                features = self.preprocess_review(review)
                features_list.append(features[0])

            if len(features_list) >= self.min_training_samples:
                self.isolation_forest.fit(features_list)
                self.is_trained = True
                logger.info(f"Anomaly detector trained with {len(features_list)} samples")
                return True
            return False
        except Exception as e:
            logger.error(f"Error in train_anomaly_detector: {e}")
            return False

    def select_memory_mode(self, review_data, business_data):
        """选择最适合的记忆模式"""
        try:
            # 如果 review_data 是字符串，需要先解析
            if isinstance(review_data, str):
                # 假设是 JSON 字符串，尝试解析
                try:
                    import json
                    review_data = json.loads(review_data)
                except:
                    # 如果解析失败，返回默认模式
                    logger.warning("Unable to parse review_data string, using default mode")
                    return self.memories['dilu']

            features = self.preprocess_review(review_data)

            if not self.is_trained:
                logger.info("Using default DILU memory mode (model not trained)")
                return self.memories['dilu']

            try:
                is_anomaly = self.isolation_forest.predict(features)[0] == -1
            except Exception as e:
                logger.error(f"Prediction error: {e}")
                is_anomaly = False

            if is_anomaly:
                logger.info("Anomaly detected, using Generative memory mode")
                return self.memories['generative']

            text = str(review_data.get('text', '')) if isinstance(review_data, dict) else str(review_data)
            if len(text) > 500:
                logger.info("Long text detected, using Voyager memory mode")
                return self.memories['voyager']

            if isinstance(business_data, dict) and business_data.get('review_count', 0) > 1000:
                logger.info("Popular business detected, using TP memory mode")
                return self.memories['tp']

            logger.info("Using default DILU memory mode")
            return self.memories['dilu']

        except Exception as e:
            logger.error(f"Error in select_memory_mode: {e}")
            return self.memories['dilu']

    def add_memory(self, memory_type, current_situation):
        """
        添加记忆到指定类型的记忆模块
        Args:
            memory_type: 记忆模块类型
            current_situation: 当前场景
        """
        try:
            if memory_type in self.memories:
                self.memories[memory_type].addMemory(current_situation)
            else:
                logger.error(f"Unknown memory type: {memory_type}")
        except Exception as e:
            logger.error(f"Error in add_memory: {e}")

    def retrieve_memory(self, memory_type, query_scenario):
        """
        从指定类型的记忆模块检索记忆
        Args:
            memory_type: 记忆模块类型
            query_scenario: 查询场景
        Returns:
            str: 检索到的记忆
        """
        try:
            if memory_type in self.memories:
                return self.memories[memory_type].retriveMemory(query_scenario)
            else:
                logger.error(f"Unknown memory type: {memory_type}")
                return ""
        except Exception as e:
            logger.error(f"Error in retrieve_memory: {e}")
            return ""

    def get_all_memories(self, query_scenario):
        """
        从所有记忆模块检索记忆并合并结果
        Args:
            query_scenario: 查询场景
        Returns:
            str: 合并后的记忆
        """
        try:
            all_memories = []
            for memory_type, memory_module in self.memories.items():
                try:
                    memory = memory_module.retriveMemory(query_scenario)
                    if memory:
                        all_memories.append(f"[{memory_type}] {memory}")
                except Exception as e:
                    logger.error(f"Error retrieving memory from {memory_type}: {e}")
                    continue

            return "\n\n".join(all_memories) if all_memories else ""
        except Exception as e:
            logger.error(f"Error in get_all_memories: {e}")
            return ""

class PlanningBaseline(PlanningBase):
    """Inherit from PlanningBase"""

    def __init__(self, llm):
        """Initialize the planning module"""
        super().__init__(llm=llm)

    def __call__(self, task_description):
        """Override the parent class's __call__ method"""
        self.plan = [
            {
                'description': 'First I need to find user information',
                'reasoning instruction': 'None',
                'tool use instruction': {task_description['user_id']}
            },
            {
                'description': 'Next, I need to find business information',
                'reasoning instruction': 'None',
                'tool use instruction': {task_description['item_id']}
            }
        ]
        return self.plan


class ReasoningBaseline(ReasoningBase):
    """Inherit from ReasoningBase"""

    def __init__(self, profile_type_prompt, llm):
        """Initialize the reasoning module"""
        super().__init__(profile_type_prompt=profile_type_prompt, memory=None, llm=llm)

    def __call__(self, task_description: str):
        """Override the parent class's __call__ method"""
        prompt = '''
{task_description}'''
        prompt = prompt.format(task_description=task_description)

        messages = [{"role": "user", "content": prompt}]
        reasoning_result = self.llm(
            messages=messages,
            temperature=0.0,
            max_tokens=1000
        )

        return reasoning_result


class DynamicWeightAdjuster:
    """动态权重调整系统"""

    def __init__(self, history_size=50):
        self.weights = {
            'text_similarity': 0.4,
            'rating_match': 0.3,
            'sentiment_match': 0.3
        }
        self.history = deque(maxlen=history_size)
        self.learning_rate = 0.1

    def adjust_weights(self, metrics):
        """根据性能指标调整权重"""
        self.history.append(metrics)
        if len(self.history) >= 5:
            # 分析最近5次的趋势
            recent_metrics = list(self.history)[-5:]
            avg_performance = np.mean([m['overall_quality'] for m in recent_metrics])

            # 根据性能调整权重
            if avg_performance < 0.6:
                self._increase_weight('text_similarity')
            elif avg_performance < 0.7:
                self._increase_weight('rating_match')
            else:
                self._balance_weights()

    def _increase_weight(self, key):
        """增加指定权重"""
        adjustment = self.learning_rate
        self.weights[key] = min(0.5, self.weights[key] + adjustment)
        # 调整其他权重保持总和为1
        remaining = 1.0 - self.weights[key]
        other_keys = [k for k in self.weights if k != key]
        for k in other_keys:
            self.weights[k] = remaining / len(other_keys)

    def _balance_weights(self):
        """平衡所有权重"""
        for key in self.weights:
            self.weights[key] = 1.0 / len(self.weights)


class ContextAwareManager:
    def __init__(self):
        self.context_cache = None

    def _extract_topics(self, text_blob: TextBlob) -> List[str]:
        """提取文本主题"""
        try:
            # 使用词频分析提取主题词
            words = text_blob.words.lower()
            # 过滤停用词和短词
            filtered_words = [w for w in words if len(w) > 3 and w not in STOP_WORDS]
            # 返回最常见的词作为主题
            return [word for word, _ in Counter(filtered_words).most_common(5)]
        except Exception as e:
            logger.error(f"Error extracting topics: {e}")
            return []

    def _extract_keywords(self, reviews: List[Dict]) -> List[str]:
        """提取关键词"""
        try:
            # 合并所有评论文本
            all_text = ' '.join([r['text'] for r in reviews])
            blob = TextBlob(all_text)
            # 使用词频和词性标注提取关键词
            words = blob.words.lower()
            # 过滤停用词和短词
            keywords = [w for w in words if len(w) > 3 and w not in STOP_WORDS]
            return list(set([word for word, _ in Counter(keywords).most_common(10)]))
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            return []

    def analyze_context(self, user_id, business_id, reviews):
        """分析上下文信息"""
        try:
            cache_key = f"{user_id}_{business_id}"
            if cache_key in self.context_cache:
                return self.context_cache[cache_key]

            if not reviews:
                # 没有评论时使用默认值
                default_context = {
                    'user_patterns': {'avg_length': 200, 'avg_rating': 3.0},
                    'business_patterns': {'common_topics': [], 'avg_satisfaction': 3.5},
                    'keywords': []
                }
                self.context_cache[cache_key] = default_context
                return default_context

            # 分析用户评论模式
            avg_length = np.mean([len(r.get('text', '')) for r in reviews])
            avg_rating = np.mean([r.get('stars', 3.0) for r in reviews])

            # 提取关键词和主题
            blob = TextBlob(' '.join([r.get('text', '') for r in reviews]))
            common_topics = self._extract_topics(blob)
            keywords = self._extract_keywords(reviews)

            context = {
                'user_patterns': {
                    'avg_length': avg_length,
                    'avg_rating': avg_rating,
                    'sentiment_distribution': self._analyze_sentiments(reviews)
                },
                'business_patterns': {
                    'common_topics': common_topics,
                    'avg_satisfaction': avg_rating
                },
                'keywords': keywords
            }

            # 缓存管理
            if len(self.context_cache) >= self.max_cache_size:
                self.context_cache.pop(next(iter(self.context_cache)))
            self.context_cache[cache_key] = context

            return context
        except Exception as e:
            logger.error(f"Error in analyze_context: {e}")
            # 返回默认上下文
            return {
                'user_patterns': {'avg_length': 200, 'avg_rating': 3.0},
                'business_patterns': {'common_topics': [], 'avg_satisfaction': 3.5},
                'keywords': []
            }

class ReviewQualityControl:
    """评论质量控制系统"""

    def __init__(self):
        self.quality_thresholds = {
            'min_length': 50,
            'max_length': 500,
            'min_words': 10,
            'coherence_threshold': 0.5
        }

    def validate_review(self, review_text, context):
        scores = {
            'coherence': self._check_coherence(review_text),
            'specificity': self._check_specificity(review_text, context),
            'authenticity': self._check_authenticity(review_text)
        }
        return self._aggregate_scores(scores)

    def _check_coherence(self, text):
        """检查评论的连贯性"""
        if not text:
            return 0.0

        sentences = sent_tokenize(text)
        if len(sentences) < 2:
            return 0.5

        # 使用 Flesch Reading Ease 评分
        readability = textstat.flesch_reading_ease(text)
        # 归一化到0-1范围
        return min(1.0, max(0.0, readability / 100.0))

    def _check_specificity(self, text, context):
        """检查评论的具体程度"""
        if not text or not context:
            return 0.0

        # 检查是否包含上下文关键词
        keywords = set(context.get('keywords', []))
        words = set(word_tokenize(text.lower()))
        keyword_ratio = len(keywords.intersection(words)) / len(keywords) if keywords else 0

        return min(1.0, keyword_ratio)

    def _check_authenticity(self, text):
        """检查评论的真实性"""
        if not text:
            return 0.0

        # 检查是否过于重复
        words = word_tokenize(text.lower())
        word_freq = defaultdict(int)
        for word in words:
            word_freq[word] += 1

        repetition_ratio = max(word_freq.values()) / len(words) if words else 1
        authenticity_score = 1.0 - repetition_ratio

        return max(0.0, authenticity_score)

    def _aggregate_scores(self, scores):
        """聚合各项评分"""
        weights = {
            'coherence': 0.4,
            'specificity': 0.4,
            'authenticity': 0.2
        }

        final_score = sum(score * weights[metric]
                          for metric, score in scores.items())

        return min(1.0, max(0.0, final_score))


class ProgressiveLearner:
    """渐进式学习优化器"""

    def __init__(self):
        self.performance_history = []
        self.adaptation_rate = 0.1
        self.model_parameters = {
            'text_weight': 0.4,
            'rating_weight': 0.3,
            'context_weight': 0.3
        }

    def update_model(self, new_data, performance):
        """基于新数据和性能更新模型"""
        self.performance_history.append({
            'performance': performance,
            'parameters': self.model_parameters.copy()
        })

        if len(self.performance_history) >= 10:
            self._analyze_and_adjust()

    def _analyze_and_adjust(self):
        """分析性能历史并调整参数"""
        recent_performance = [p['performance'] for p in self.performance_history[-10:]]
        trend = np.polyfit(range(10), recent_performance, 1)[0]

        if trend < 0:  # 性能下降趋势
            self._adjust_parameters()

    def _adjust_parameters(self):
        """调整模型参数"""
        # 找到历史最佳性能的参数
        best_performance = max(self.performance_history,
                               key=lambda x: x['performance'])

        # 向最佳参数靠拢
        for param in self.model_parameters:
            current = self.model_parameters[param]
            target = best_performance['parameters'][param]
            self.model_parameters[param] = current + self.adaptation_rate * (target - current)


class MySimulationAgent(SimulationAgent):
    """Enhanced implementation of SimulationAgent with advanced features."""

    def __init__(self, llm: LLMBase):
        """
        初始化增强版 SimulationAgent
        Args:
            llm: 语言模型实例
        """
        super().__init__(llm=llm)
        # 基础组件
        self.planning = PlanningBaseline(llm=self.llm)
        self.reasoning = ReasoningBaseline(profile_type_prompt='', llm=self.llm)
        self.memory_manager = MemoryManager(llm=self.llm)

        # 增强组件
        self.weight_adjuster = DynamicWeightAdjuster()
        self.context_manager = ContextAwareManager()
        self.quality_control = ReviewQualityControl()
        self.progressive_learner = ProgressiveLearner()

        # 其他属性
        self.embedding = None
        self.interaction_tool = None
        self.task = None
        self.retry_attempts = 2
        self.quality_threshold = 0.5

        # 性能追踪
        self.performance_metrics = {
            'quality_scores': [],
            'generation_attempts': [],
            'success_rate': 0.0
        }

    def setup(self, interaction_tool, task, embedding_model):
        """
        设置代理所需的组件
        Args:
            interaction_tool: 交互工具
            task: 任务信息
            embedding_model: 嵌入模型
        """
        self.interaction_tool = interaction_tool
        self.task = task
        self.embedding = embedding_model

    def analyze_user(self, user_info: str, reviews_user: List[Dict]) -> str:
        """
        增强的用户资料分析
        Args:
            user_info: 用户基本信息
            reviews_user: 用户历史评论
        Returns:
            str: 分析结果
        """
        try:
            user_info_brief = str(user_info)[:1500]

            # 分析用户评论模式
            if reviews_user:
                avg_rating = np.mean([r['stars'] for r in reviews_user])
                avg_length = np.mean([len(str(r.get('text', ''))) for r in reviews_user])
                common_ratings = Counter([r['stars'] for r in reviews_user]).most_common(2)
            else:
                avg_rating = 3.0
                avg_length = 200
                common_ratings = [(3.0, 1)]

            task = f'''
            Analyze this Yelp user profile in detail. Extract:
            1. Rating patterns:
               - Average rating: {avg_rating:.1f}
               - Most common ratings: {common_ratings}
               - Average review length: {avg_length:.0f} chars
            2. Writing style (length, detail level, tone)
            3. Specific preferences and dislikes
            4. Typical review focus areas
            5. Response to different service levels

            Profile excerpt: {user_info_brief}

            Provide a concise summary of key patterns.
            '''

            messages = [{"role": "user", "content": task}]
            return self.reasoning(messages)
        except Exception as e:
            logger.error(f"Error in analyze_user: {e}")
            return "Average Yelp reviewer with balanced ratings and standard review length."

    def analyze_business(self, business_info: str, reviews_item: List[Dict]) -> str:
        """
        增强的商家资料分析
        Args:
            business_info: 商家基本信息
            reviews_item: 商家历史评论
        Returns:
            str: 分析结果
        """
        try:
            # 分析评论模式
            if reviews_item:
                avg_rating = np.mean([r['stars'] for r in reviews_item])
                recent_trend = np.mean([r['stars'] for r in reviews_item[:5]]) - avg_rating
                common_words = self._extract_common_words([r['text'] for r in reviews_item])
            else:
                avg_rating = 3.5
                recent_trend = 0.0
                common_words = []

            task = f'''
            Based on this Yelp business profile, analyze:
            1. Business type and category
            2. Price range and service level
            3. Most mentioned positive features
            4. Common customer complaints
            5. Overall service expectations

            Business profile: {business_info}

            Additional insights:
            - Average rating: {avg_rating:.1f} stars
            - Recent trend: {recent_trend:+.1f} stars
            - Common themes: {', '.join(common_words[:5])}

            Focus on aspects that influence customer reviews.
            '''

            messages = [{"role": "user", "content": task}]
            return self.reasoning(messages)
        except Exception as e:
            logger.error(f"Error in analyze_business: {e}")
            return ""

    def _extract_common_words(self, texts: List[str], top_n: int = 10) -> List[str]:
        """提取文本中的常见词汇"""
        try:
            # 合并所有文本
            combined_text = ' '.join(texts)
            # 分词并统计
            words = word_tokenize(combined_text.lower())
            # 过滤停用词和短词
            filtered_words = [w for w in words if len(w) > 3 and w not in STOP_WORDS]
            return [word for word, _ in Counter(filtered_words).most_common(top_n)]
        except Exception as e:
            logger.error(f"Error in _extract_common_words: {e}")
            return []

    def process_reviews(self, reviews: List[Dict]) -> str:
        """
        增强的评论处理
        Args:
            reviews: 评论列表
        Returns:
            str: 处理结果
        """
        if not reviews:
            return ""

        try:
            # 选择代表性样本
            review_samples = self._select_representative_samples(reviews, n=3)
            sample_texts = [r['text'][:300] for r in review_samples]

            # 提取评论特征
            avg_length = np.mean([len(r['text']) for r in reviews])
            rating_dist = Counter([r['stars'] for r in reviews])

            task = f'''
            Analyze these sample reviews to identify:
            1. Common rating justifications
            2. Writing style patterns (avg length: {avg_length:.0f} chars)
            3. Rating distribution: {dict(rating_dist)}
            4. Emotional tone trends
            5. Specific details mentioned

            Reviews:
            {' | '.join(sample_texts)}

            Provide a brief synthesis of key patterns.
            '''

            messages = [{"role": "user", "content": task}]
            return self.reasoning(messages)
        except Exception as e:
            logger.error(f"Error in process_reviews: {e}")
            return ""

    def _select_representative_samples(self, reviews: List[Dict], n: int = 3) -> List[Dict]:
        """选择代表性的评论样本"""
        try:
            if len(reviews) <= n:
                return reviews

            # 按评分分组
            grouped = defaultdict(list)
            for review in reviews:
                grouped[review['stars']].append(review)

            # 从每个评分组中选择样本
            samples = []
            for star in sorted(grouped.keys(), reverse=True):
                if len(samples) < n:
                    group_reviews = grouped[star]
                    # 选择最有用的评论
                    sorted_reviews = sorted(group_reviews,
                                            key=lambda x: x.get('useful', 0),
                                            reverse=True)
                    samples.extend(sorted_reviews[:1])

            return samples[:n]
        except Exception as e:
            logger.error(f"Error in _select_representative_samples: {e}")
            return reviews[:n]

    def workflow(self) -> Tuple[float, int, int, int, str]:
        """
        执行增强的评论模拟工作流
        Returns:
            tuple: (stars, useful, funny, cool, review_text)
        """
        try:
            # 1. 获取基础信息
            user = str(self.interaction_tool.get_user(user_id=self.task['user_id']))
            business = str(self.interaction_tool.get_item(item_id=self.task['item_id']))

            logger.info("\n====== Input Data ======")
            logger.info(f"User ID: {self.task['user_id']}")
            logger.info(f"Business ID: {self.task['item_id']}")

            # 2. 获取评论数据
            reviews_item = self.interaction_tool.get_reviews(item_id=self.task['item_id']) or []
            reviews_user = self.interaction_tool.get_reviews(user_id=self.task['user_id']) or []

            # 3. 构建全量评论数据
            all_reviews = []
            if reviews_item:
                all_reviews.extend(reviews_item[:10])  # 使用商家最近10条评论
            if reviews_user:
                all_reviews.extend(reviews_user[:5])  # 使用用户最近5条评论

            # 4. 训练异常检测器
            if all_reviews:
                logger.info("Training anomaly detector...")
                self.memory_manager.train_anomaly_detector(all_reviews)

            # 5. 内存管理
            logger.info("Managing memory...")
            if reviews_item:
                for review in reviews_item[:5]:  # 商家近期评论
                    memory_mode = self.memory_manager.select_memory_mode(review, business)
                    memory_mode.addMemory(f"business review: {review['text']}")
            if reviews_user:
                for review in reviews_user[:3]:  # 用户近期评论
                    memory_mode = self.memory_manager.select_memory_mode(review, business)
                    memory_mode.addMemory(f"user review: {review['text']}")

            # 6. 上下文分析
            context = {}
            try:
                context = {
                    'user_patterns': self._analyze_user_patterns(reviews_user),
                    'business_patterns': self._analyze_business_patterns(reviews_item),
                    'keywords': self._extract_common_words(
                        [r['text'] for r in all_reviews] if all_reviews else []
                    )
                }
            except Exception as e:
                logger.warning(f"Context analysis failed, using defaults: {e}")
                context = {
                    'user_patterns': {'avg_length': 200, 'avg_rating': 3.0},
                    'business_patterns': {'common_topics': [], 'avg_satisfaction': 3.5},
                    'keywords': []
                }

            # 7. 增强分析
            user_summary = self.analyze_user(user, reviews_user)
            business_summary = self.analyze_business(business, reviews_item)
            review_patterns = self.process_reviews(reviews_item)

            # 8. 获取所有记忆
            all_memories = self.memory_manager.get_all_memories(str(business))

            # 9. 生成评论（带重试机制）
            last_error = None
            for attempt in range(self.retry_attempts):
                try:
                    # 9.1 生成评论
                    result = self.generate_review(
                        user_summary=user_summary,
                        business_summary=business_summary,
                        review_context=f"{review_patterns}\n{all_memories}",
                        context=context,
                        retry=(attempt > 0)
                    )

                    # 9.2 解析结果
                    parsed_result = self.parse_review_result(result)

                    # 9.3 质量控制
                    quality_score = self.quality_control.validate_review(parsed_result[4], context)

                    # 9.4 如果质量达标
                    if quality_score >= self.quality_threshold:
                        # 更新性能指标
                        self._update_performance_metrics(quality_score, attempt + 1)

                        # 更新学习组件
                        self._update_learning_components(parsed_result, quality_score)

                        logger.info(f"Successfully generated review on attempt {attempt + 1}")
                        return parsed_result
                    else:
                        logger.warning(
                            f"Review quality {quality_score} below threshold {self.quality_threshold} on attempt {attempt + 1}")

                except Exception as e:
                    last_error = e
                    logger.error(f"Error on attempt {attempt + 1}: {e}")
                    continue

            # 10. 所有重试失败，返回默认值
            if last_error:
                logger.error(f"All attempts failed. Last error: {last_error}")
            else:
                logger.warning("All attempts failed to meet quality threshold")

            return 3.0, 0, 0, 0, "Unable to generate a satisfactory review."

        except Exception as e:
            logger.error(f"Workflow error: {e}")
            return 3.0, 0, 0, 0, f"Error in review generation process: {str(e)}"

    def _update_performance_metrics(self, quality_score: float, attempts: int):
        """更新性能指标"""
        try:
            # 更新质量分数历史
            self.performance_metrics['quality_scores'].append(quality_score)
            # 更新尝试次数历史
            self.performance_metrics['generation_attempts'].append(attempts)

            # 计算成功率
            total_attempts = sum(self.performance_metrics['generation_attempts'])
            total_success = len(self.performance_metrics['quality_scores'])
            success_rate = total_success / total_attempts if total_attempts > 0 else 0.0

            # 更新整体成功率
            self.performance_metrics['success_rate'] = success_rate

            logger.info(f"Updated performance metrics - Quality: {quality_score:.2f}, "
                        f"Attempts: {attempts}, Success Rate: {success_rate:.2f}")
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")

    def _update_learning_components(self, parsed_result: tuple, quality_score: float):
        """更新学习组件"""
        try:
            # 计算综合评分
            stars_score = parsed_result[0] / 5.0  # 归一化星级评分
            metrics = {
                'overall_quality': quality_score,
                'review_quality': quality_score * 0.8 + stars_score * 0.2
            }

            # 更新动态权重
            self.weight_adjuster.adjust_weights(metrics)

            # 更新渐进式学习器
            self.progressive_learner.update_model(parsed_result, quality_score)

            logger.info(f"Updated learning components with quality score: {quality_score:.2f}")
        except Exception as e:
            logger.error(f"Error updating learning components: {e}")

    def _analyze_user_patterns(self, reviews: List[Dict]) -> Dict:
        """分析用户评论模式"""
        try:
            if not reviews:
                return {'avg_length': 200, 'avg_rating': 3.0}

            avg_rating = np.mean([r.get('stars', 3.0) for r in reviews])
            avg_length = np.mean([len(str(r.get('text', ''))) for r in reviews])

            return {
                'avg_length': avg_length,
                'avg_rating': avg_rating,
            }
        except Exception as e:
            logger.error(f"Error analyzing user patterns: {e}")
            return {'avg_length': 200, 'avg_rating': 3.0}

    def _analyze_business_patterns(self, reviews: List[Dict]) -> Dict:
        """分析商家评论模式"""
        try:
            if not reviews:
                return {'common_topics': [], 'avg_satisfaction': 3.5}

            avg_rating = np.mean([r.get('stars', 3.5) for r in reviews])
            # 提取常见主题词
            all_text = ' '.join([r.get('text', '') for r in reviews])
            blob = TextBlob(all_text)
            words = [w for w in blob.words.lower() if len(w) > 3 and w not in STOP_WORDS]
            common_topics = [word for word, _ in Counter(words).most_common(5)]

            return {
                'common_topics': common_topics,
                'avg_satisfaction': avg_rating
            }
        except Exception as e:
            logger.error(f"Error analyzing business patterns: {e}")
            return {'common_topics': [], 'avg_satisfaction': 3.5}

    def generate_review(self, user_summary: str, business_summary: str,
                        review_context: str, context: dict, retry: bool = False) -> str:
        """
        增强的评论生成
        """
        try:
            weights = self.weight_adjuster.weights
            recent_patterns = context.get('user_patterns', {})
            business_patterns = context.get('business_patterns', {})

            task_description = f'''
            Write a realistic Yelp review using this context:

            USER STYLE AND PATTERNS:
            {user_summary[:500]}
            Recent patterns: 
            - Average length: {recent_patterns.get('avg_length', 200):.0f} chars
            - Typical rating: {recent_patterns.get('avg_rating', 3.0):.1f} stars

            BUSINESS CHARACTERISTICS:
            {business_summary[:500]}
            Key aspects:
            - Common topics: {', '.join(business_patterns.get('common_topics', [])[:3])}
            - Average satisfaction: {business_patterns.get('avg_satisfaction', 3.5):.1f}

            REVIEW PATTERNS:
            {review_context[:500]}

            Style weights:
            - Text similarity: {weights['text_similarity']:.2f}
            - Rating match: {weights['rating_match']:.2f}
            - Sentiment match: {weights['sentiment_match']:.2f}

            {f"Please improve quality and specificity. Focus on detailed experiences." if retry else ""}

            Requirements:
            1. Match user's typical rating and writing style
            2. Include specific business details from the list above
            3. Maintain natural tone and appropriate length
            4. Keep metrics realistic and justified
            5. Include relevant keywords: {', '.join(context.get('keywords', [])[:5])}

            EXACTLY follow this format:
            stars: [1.0-5.0]
            useful: [0-3]
            funny: [0-1]
            cool: [0-2]
            review: [2-4 concise, specific sentences]
            '''

            messages = [{"role": "user", "content": task_description}]
            return self.reasoning(messages)

        except Exception as e:
            logger.error(f"Error in generate_review: {e}")
            return "stars: 3.0\nuseful: 0\nfunny: 0\ncool: 0\nreview: Unable to generate review."

    def parse_review_result(self, result: str) -> Tuple[float, int, int, int, str]:
        """
        解析生成的评论结果
        Args:
            result: 生成的评论文本
        Returns:
            tuple: (stars, useful, funny, cool, review_text)
        """
        try:
            lines = result.strip().split('\n')
            parsed = {}
            current_field = None

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                for field in ['stars:', 'useful:', 'funny:', 'cool:', 'review:']:
                    if line.lower().startswith(field):
                        current_field = field[:-1]
                        value = line[len(field):].strip()
                        parsed[current_field] = value
                        break
                else:
                    if current_field == 'review':
                        parsed['review'] = parsed.get('review', '') + ' ' + line

            # 转换和验证值
            stars = float(parsed.get('stars', '3.0'))
            stars = max(1.0, min(5.0, stars))

            useful = int(float(parsed.get('useful', '0')))
            useful = max(0, min(3, useful))

            funny = int(float(parsed.get('funny', '0')))
            funny = max(0, min(1, funny))

            cool = int(float(parsed.get('cool', '0')))
            cool = max(0, min(2, cool))

            # 处理评论文本
            review_text = parsed.get('review', 'No review content provided.').strip()
            # 清理评论文本
            review_text = ' '.join(review_text.split())  # 规范化空白字符
            review_text = review_text[:512] if len(review_text) > 512 else review_text

            # 验证所有必需字段都存在
            if not all(key in parsed for key in ['stars', 'useful', 'funny', 'cool', 'review']):
                logger.warning("Missing required fields in review result")
                raise ValueError("Generated review missing required fields")

            # 验证评论文本的最小长度
            if len(review_text.split()) < 5:  # 至少5个单词
                logger.warning("Review text too short")
                raise ValueError("Generated review text too short")

            return stars, useful, funny, cool, review_text

        except Exception as e:
            logger.error(f"Error parsing review result: {e}")
            raise ValueError(f"Failed to parse review result: {str(e)}")

def main(batch_size: int = 50, max_tasks: int = None):
    """
    运行模拟程序
    Args:
        batch_size: 每批处理的任务数量，用于定期评估
        max_tasks: 最大处理任务数，None表示处理所有任务
    Returns:
        tuple: (all_simulated_data, all_real_data)
    """
    # 设置日志
    logging.basicConfig(level=logging.INFO)

    try:
        # 创建 OpenAI wrapper
        llm = OpenAIWrapper(
            api_key="sk-cS3K2urP0tyj470rD0Fb0a61EaC949D8AcAfC7C55eD00dCc",
            base_url="https://api.gpt.ge/v1/",
            model="gpt-3.5-turbo"
        )

        # 设置数据目录和创建交互工具
        data_directory = r"C:\Users\hq200\Desktop\User Modeling Track\files"
        interaction_tool = CacheInteractionTool(data_dir=data_directory)

        # 创建评估器
        simulation_evaluator = SimulationEvaluator(device="auto")

        # 用于存储所有模拟结果
        all_simulated_data = []
        all_real_data = []

        # 创建任务列表
        tasks = []
        logger.info("正在收集任务...")

        # 获取所有用户和商品ID
        users = interaction_tool.get_all_user_ids()
        items = interaction_tool.get_all_item_ids()

        logger.info(f"发现 {len(users)} 个用户和 {len(items)} 个商品")

        # 优化任务收集逻辑
        task_count = 0
        for user_id in users:
            user_reviews = interaction_tool.get_reviews(user_id=user_id)
            if user_reviews:
                for review in user_reviews:
                    item_id = review['item_id']
                    tasks.append({
                        "user_id": user_id,
                        "item_id": item_id
                    })
                    task_count += 1

                    if max_tasks is not None and task_count >= max_tasks:
                        break

            if max_tasks is not None and task_count >= max_tasks:
                break

        total_tasks = len(tasks)
        logger.info(f"总任务数: {total_tasks}")

        # 批量处理任务
        for i, task in enumerate(tasks, 1):
            try:
                logger.info(f"处理任务 {i}/{total_tasks}")
                logger.info(f"用户 ID: {task['user_id']}, 商品 ID: {task['item_id']}")

                # 创建和设置 agent
                agent = MySimulationAgent(llm=llm)
                agent.setup(
                    interaction_tool=interaction_tool,
                    task=task,
                    embedding_model=llm.get_embedding_model()
                )

                # 执行 workflow
                stars, useful, funny, cool, review_text = agent.workflow()

                # 收集模拟数据
                simulated_data = {
                    'stars': stars,
                    'review': review_text,
                    'user_id': task['user_id'],
                    'item_id': task['item_id'],
                    'useful': useful,
                    'funny': funny,
                    'cool': cool
                }
                all_simulated_data.append(simulated_data)

                # 获取真实数据
                real_reviews = interaction_tool.get_reviews(
                    user_id=task['user_id'],
                    item_id=task['item_id']
                )

                if real_reviews:
                    review_data = {
                        'stars': real_reviews[0].get('stars', 0.0),
                        'review': real_reviews[0].get('text', ''),
                        'user_id': task['user_id'],
                        'item_id': task['item_id'],
                        'useful': real_reviews[0].get('useful', 0),
                        'funny': real_reviews[0].get('funny', 0),
                        'cool': real_reviews[0].get('cool', 0)
                    }
                    all_real_data.append(review_data)

                # 定期计算和输出评估指标
                if i % batch_size == 0 or i == total_tasks:
                    metrics = simulation_evaluator.calculate_metrics(
                        simulated_data=all_simulated_data,
                        real_data=all_real_data
                    )
                    logger.info(f"\n中间评估指标 (已完成 {i} 个任务):")
                    logger.info(f"偏好估计: {metrics.preference_estimation:.4f}")
                    logger.info(f"评论生成: {metrics.review_generation:.4f}")
                    logger.info(f"整体质量: {metrics.overall_quality:.4f}\n")

                    # 保存中间结果
                    save_results(all_simulated_data, all_real_data, i)

            except Exception as e:
                logger.error(f"处理任务 {i} 时出错: {str(e)}")
                logger.error("错误详情:", exc_info=True)
                continue

        # 最终评估
        if all_real_data:
            final_metrics = simulation_evaluator.calculate_metrics(
                simulated_data=all_simulated_data,
                real_data=all_real_data
            )

            logger.info("\n最终评估指标:")
            logger.info(f"偏好估计: {final_metrics.preference_estimation:.4f}")
            logger.info(f"评论生成: {final_metrics.review_generation:.4f}")
            logger.info(f"整体质量: {final_metrics.overall_quality:.4f}")

            # 保存最终结果
            save_results(all_simulated_data, all_real_data, 'final')

        return all_simulated_data, all_real_data

    except Exception as e:
        logger.error(f"main 函数执行出错: {str(e)}")
        logger.error("错误详情:", exc_info=True)
        raise


def save_results(simulated_data, real_data, checkpoint):
    """保存模拟结果到文件"""
    results_dir = "simulation_results"
    os.makedirs(results_dir, exist_ok=True)

    # 保存模拟数据
    with open(f"{results_dir}/simulated_data_{checkpoint}.json", 'w', encoding='utf-8') as f:
        json.dump(simulated_data, f, ensure_ascii=False, indent=2)

    # 保存真实数据
    with open(f"{results_dir}/real_data_{checkpoint}.json", 'w', encoding='utf-8') as f:
        json.dump(real_data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # 可以通过参数控制批量大小和最大任务数
    all_simulated_data, all_real_data = main(batch_size=3, max_tasks=20)

